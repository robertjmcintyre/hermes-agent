"""Unit tests for Secondbrain memory provider plugin."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict
from unittest import mock
from unittest.mock import MagicMock, patch

import pytest

from plugins.memory.secondbrain import (
    READ_SCHEMA,
    SEARCH_SCHEMA,
    WRITE_SCHEMA,
    UPDATE_SCHEMA,
    BACKLINKS_SCHEMA,
    REFERENCES_SCHEMA,
    ALL_TOOL_SCHEMAS,
    SecondbrainClient,
    SecondbrainMemoryProvider,
    register,
)


# ============================================================================
# Test fixtures and helpers
# ============================================================================

@pytest.fixture
def temp_hermes_home():
    """Create a temporary hermes home directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mock_client():
    """Create a mock SecondbrainClient."""
    client = MagicMock(spec=SecondbrainClient)
    client.health_check.return_value = True
    client.call_tool.return_value = {"status": "healthy"}
    return client


@pytest.fixture
def provider(mock_client):
    """Create a provider with mocked client."""
    with patch("plugins.memory.secondbrain.SecondbrainClient") as MockClient:
        MockClient.return_value = mock_client
        prov = SecondbrainMemoryProvider()
        prov._client = mock_client
        prov._session_id = "test-session"
        return prov


# ============================================================================
# Test SecondbrainClient
# ============================================================================

class TestSecondbrainClient:
    """Tests for the SecondbrainClient class."""

    def test_init_with_defaults(self):
        """Test client initialization with default values."""
        client = SecondbrainClient("http://localhost:18764")
        assert client.base_url == "http://localhost:18764"
        assert client.mcp_url == "http://localhost:18764/mcp"
        assert client.api_key is None
        assert client.timeout == 30

    def test_init_with_custom_values(self):
        """Test client initialization with custom values."""
        client = SecondbrainClient(
            base_url="http://example.com:8080",
            api_key="secret-key",
            timeout=60,
        )
        assert client.base_url == "http://example.com:8080"
        assert client.mcp_url == "http://example.com:8080/mcp"
        assert client.api_key == "secret-key"
        assert client.timeout == 60

    def test_init_strips_trailing_slash(self):
        """Test that trailing slash is stripped from URL."""
        client = SecondbrainClient("http://localhost:18764/")
        assert client.base_url == "http://localhost:18764"

    def test_get_headers_without_api_key(self):
        """Test headers without API key."""
        client = SecondbrainClient("http://localhost:18764")
        headers = client._get_headers()
        assert headers["Content-Type"] == "application/json"
        assert "Authorization" not in headers

    def test_get_headers_with_api_key(self):
        """Test headers with API key."""
        client = SecondbrainClient("http://localhost:18764", api_key="my-key")
        headers = client._get_headers()
        assert headers["Authorization"] == "Bearer my-key"

    @patch.dict(os.environ, {"SECONDBRAIN_USE_SIMPLE_HTTP": "1"})
    @patch("plugins.memory.secondbrain.requests.post")
    def test_call_tool_success(self, mock_post):
        """Test successful tool call (uses fallback HTTP)."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"result": {"status": "ok"}}
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        client = SecondbrainClient("http://localhost:18764")
        result = client.call_tool("health_check")

        assert result == {"status": "ok"}
        mock_post.assert_called_once()

    @patch("mcp.client.streamable_http.streamablehttp_client")
    @patch("mcp.ClientSession")
    def test_call_tool_connection_error(self, mock_session_class, mock_client):
        """Test connection error handling."""
        # Mock the MCP client to raise connection error
        mock_client.side_effect = Exception("Connection refused")

        client = SecondbrainClient("http://localhost:18764")
        result = client.call_tool("health_check")

        assert "error" in result

    @patch("mcp.client.streamable_http.streamablehttp_client")
    @patch("mcp.ClientSession")
    def test_call_tool_timeout(self, mock_session_class, mock_client):
        """Test timeout handling."""
        # Mock the MCP client to raise timeout
        mock_client.side_effect = Exception("Request timed out")

        client = SecondbrainClient("http://localhost:18764")
        result = client.call_tool("health_check")

        assert "error" in result

    @patch("plugins.memory.secondbrain.requests.post")
    def test_health_check_success(self, mock_post):
        """Test health check when healthy (uses fallback HTTP)."""
        # The client will try MCP first, but since MCP isn't properly mocked,
        # it will fall back to simple HTTP which uses requests.post
        mock_response = MagicMock()
        mock_response.json.return_value = {"result": {"status": "healthy"}}
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        client = SecondbrainClient("http://localhost:18764")
        assert client.health_check() is True

    @patch("mcp.client.streamable_http.streamablehttp_client")
    @patch("mcp.ClientSession")
    def test_health_check_failure(self, mock_session_class, mock_client):
        """Test health check when unhealthy."""
        # Mock the MCP client
        mock_read = MagicMock()
        mock_write = MagicMock()
        mock_client.return_value.__aenter__.return_value = (mock_read, mock_write, None)
        
        # Mock the session
        mock_session = MagicMock()
        mock_session_class.return_value.__aenter__.return_value = mock_session
        
        # Mock the tool result for unhealthy
        mock_tool_result = MagicMock()
        mock_tool_result.content = [MagicMock(text='{"status": "unhealthy"}')]
        mock_session.call_tool.return_value = mock_tool_result

        client = SecondbrainClient("http://localhost:18764")
        assert client.health_check() is False


# ============================================================================
# Test SecondbrainMemoryProvider - Properties and Basic Methods
# ============================================================================

class TestSecondbrainMemoryProvider:
    """Tests for the SecondbrainMemoryProvider class."""

    def test_name_property(self):
        """Test the name property returns 'secondbrain'."""
        provider = SecondbrainMemoryProvider()
        assert provider.name == "secondbrain"

    def test_is_available_with_env_url(self):
        """Test is_available returns True when SECONDBRAIN_URL is set."""
        with mock.patch.dict(os.environ, {"SECONDBRAIN_URL": "http://localhost:18764"}):
            provider = SecondbrainMemoryProvider()
            assert provider.is_available() is True

    def test_is_available_without_env_url(self):
        """Test is_available returns False when no URL is configured."""
        with mock.patch.dict(os.environ, {}, clear=True):
            provider = SecondbrainMemoryProvider()
            assert provider.is_available() is False

    def test_is_available_with_config_file(self, temp_hermes_home):
        """Test is_available returns True when config file exists."""
        config_path = temp_hermes_home / "secondbrain.json"
        config_path.write_text(json.dumps({"url": "http://localhost:18764"}))

        with mock.patch("hermes_constants.get_hermes_home", return_value=temp_hermes_home):
            with mock.patch.dict(os.environ, {}, clear=True):
                provider = SecondbrainMemoryProvider()
                assert provider.is_available() is True

    def test_get_config_schema(self):
        """Test config schema returns expected fields."""
        provider = SecondbrainMemoryProvider()
        schema = provider.get_config_schema()

        assert len(schema) == 2
        assert schema[0]["key"] == "url"
        assert schema[0]["default"] == "http://localhost:18764"
        assert schema[0]["required"] is True
        assert schema[1]["key"] == "api_key"
        assert schema[1]["secret"] is True

    def test_save_config(self, temp_hermes_home):
        """Test save_config writes to file."""
        provider = SecondbrainMemoryProvider()
        provider.save_config(
            {"url": "http://localhost:18764"},
            str(temp_hermes_home),
        )

        config_path = temp_hermes_home / "secondbrain.json"
        assert config_path.exists()
        config = json.loads(config_path.read_text())
        assert config["url"] == "http://localhost:18764"

    def test_save_config_merges_existing(self, temp_hermes_home):
        """Test save_config merges with existing config."""
        # Create existing config
        config_path = temp_hermes_home / "secondbrain.json"
        config_path.write_text(json.dumps({"url": "http://old:1234", "extra": "value"}))

        provider = SecondbrainMemoryProvider()
        provider.save_config(
            {"url": "http://localhost:18764"},
            str(temp_hermes_home),
        )

        config = json.loads(config_path.read_text())
        assert config["url"] == "http://localhost:18764"
        assert config["extra"] == "value"


# ============================================================================
# Test SecondbrainMemoryProvider - Initialization
# ============================================================================

class TestProviderInitialization:
    """Tests for provider initialization."""

    def test_initialize_skips_cron_context(self, mock_client):
        """Test that initialize skips for cron context."""
        with patch("plugins.memory.secondbrain.SecondbrainClient") as MockClient:
            MockClient.return_value = mock_client
            provider = SecondbrainMemoryProvider()
            provider.initialize("test-session", agent_context="cron", hermes_home="/tmp")

            assert provider._client is None

    def test_initialize_skips_flush_context(self, mock_client):
        """Test that initialize skips for flush context."""
        with patch("plugins.memory.secondbrain.SecondbrainClient") as MockClient:
            MockClient.return_value = mock_client
            provider = SecondbrainMemoryProvider()
            provider.initialize("test-session", agent_context="flush", hermes_home="/tmp")

            assert provider._client is None

    def test_initialize_skips_cron_platform(self, mock_client):
        """Test that initialize skips for cron platform."""
        with patch("plugins.memory.secondbrain.SecondbrainClient") as MockClient:
            MockClient.return_value = mock_client
            provider = SecondbrainMemoryProvider()
            provider.initialize("test-session", platform="cron", hermes_home="/tmp")

            assert provider._client is None

    def test_initialize_with_env_url(self, mock_client):
        """Test initialization with URL from environment."""
        with mock.patch.dict(os.environ, {"SECONDBRAIN_URL": "http://localhost:18764"}):
            with patch("plugins.memory.secondbrain.SecondbrainClient") as MockClient:
                MockClient.return_value = mock_client
                provider = SecondbrainMemoryProvider()
                provider.initialize("test-session", hermes_home="/tmp")

                assert provider._client is not None
                assert provider._session_id == "test-session"

    def test_initialize_health_check_failure(self):
        """Test initialization fails gracefully when health check fails."""
        with mock.patch.dict(os.environ, {"SECONDBRAIN_URL": "http://localhost:18764"}):
            with patch("plugins.memory.secondbrain.SecondbrainClient") as MockClient:
                mock_client = MagicMock()
                mock_client.health_check.return_value = False
                MockClient.return_value = mock_client

                provider = SecondbrainMemoryProvider()
                provider.initialize("test-session", hermes_home="/tmp")

                assert provider._client is None


# ============================================================================
# Test SecondbrainMemoryProvider - System Prompt
# ============================================================================

class TestSystemPromptBlock:
    """Tests for system_prompt_block method."""

    def test_returns_empty_when_not_available(self):
        """Test returns empty string when client is None."""
        provider = SecondbrainMemoryProvider()
        provider._client = None

        result = provider.system_prompt_block()
        assert result == ""

    def test_returns_prompt_when_available(self, provider):
        """Test returns prompt text when client is available."""
        result = provider.system_prompt_block()

        assert "Secondbrain Memory" in result
        assert "secondbrain_read" in result
        assert "secondbrain_search" in result
        assert "secondbrain_write" in result
        assert "secondbrain_update" in result


# ============================================================================
# Test SecondbrainMemoryProvider - Prefetch
# ============================================================================

class TestPrefetch:
    """Tests for prefetch method."""

    def test_returns_empty_when_no_client(self):
        """Test returns empty when client is None."""
        provider = SecondbrainMemoryProvider()
        provider._client = None

        result = provider.prefetch("test query")
        assert result == ""

    def test_returns_empty_when_no_query(self, provider):
        """Test returns empty when query is empty."""
        result = provider.prefetch("")
        assert result == ""

    def test_returns_formatted_results(self, provider):
        """Test returns formatted search results."""
        provider._client.call_tool.return_value = {
            "results": [
                {"doc_id": "doc1", "chunk_text": "Content about test"},
                {"doc_id": "doc2", "chunk_text": "More content"},
            ]
        }

        result = provider.prefetch("test query")

        assert "Relevant memories" in result
        assert "doc1" in result
        assert "doc2" in result

    def test_returns_empty_when_no_results(self, provider):
        """Test returns empty when no results found."""
        provider._client.call_tool.return_value = {"results": []}

        result = provider.prefetch("test query")
        assert result == ""


# ============================================================================
# Test SecondbrainMemoryProvider - Queue Prefetch
# ============================================================================

class TestQueuePrefetch:
    """Tests for queue_prefetch method."""

    def test_does_nothing_when_no_client(self):
        """Test does nothing when client is None."""
        provider = SecondbrainMemoryProvider()
        provider._client = None

        # Should not raise
        provider.queue_prefetch("test query")

    def test_does_nothing_when_no_query(self, provider):
        """Test does nothing when query is empty."""
        # Should not raise
        provider.queue_prefetch("")


# ============================================================================
# Test SecondbrainMemoryProvider - Sync Turn
# ============================================================================

class TestSyncTurn:
    """Tests for sync_turn method."""

    def test_does_nothing_when_no_client(self):
        """Test does nothing when client is None."""
        provider = SecondbrainMemoryProvider()
        provider._client = None

        # Should not raise
        provider.sync_turn("user message", "assistant message")

    def test_does_nothing_when_no_session(self, provider):
        """Test does nothing when session_id is empty."""
        provider._session_id = ""

        # Should not raise
        provider.sync_turn("user message", "assistant message")

    def test_updates_existing_document(self, provider):
        """Test updates existing document."""
        provider._client.call_tool.return_value = {"success": True}

        provider.sync_turn("user message", "assistant message")

        # Wait for the sync thread to complete
        if provider._sync_thread:
            provider._sync_thread.join(timeout=5.0)

        # Should call memory_update
        provider._client.call_tool.assert_called()
        call_args = provider._client.call_tool.call_args
        assert call_args[1]["doc_id"].startswith("sessions/")

    def test_creates_document_when_not_exists(self, provider):
        """Test creates document when update fails with not found."""
        # First call (update) returns error, second call (write) succeeds
        provider._client.call_tool.side_effect = [
            {"error": "Document not found"},
            {"success": True},
        ]

        provider.sync_turn("user message", "assistant message")

        # Wait for the sync thread to complete
        if provider._sync_thread:
            provider._sync_thread.join(timeout=5.0)

        # Should have called twice - update then write
        assert provider._client.call_tool.call_count == 2


# ============================================================================
# Test SecondbrainMemoryProvider - Turn Events
# ============================================================================

class TestTurnEvents:
    """Tests for turn event handlers."""

    def test_on_turn_start_tracks_count(self, provider):
        """Test on_turn_start tracks turn count."""
        provider.on_turn_start(1, "Hello")
        assert provider._turn_count == 1

        provider.on_turn_start(2, "How are you?")
        assert provider._turn_count == 2

    def test_on_turn_start_with_prefetch_result(self, provider):
        """Test on_turn_start clears prefetch result."""
        # Set a prefetch result
        with provider._prefetch_lock:
            provider._prefetch_result = "previous result"
        
        provider.on_turn_start(1, "Hello")
        
        # Prefetch result should be cleared
        assert provider._prefetch_result == ""

    def test_on_session_end_waits_for_sync(self, provider):
        """Test on_session_end waits for sync thread."""
        mock_thread = MagicMock()
        mock_thread.is_alive.return_value = True
        mock_thread.join.return_value = None
        provider._sync_thread = mock_thread

        # Should not raise
        provider.on_session_end([])

        mock_thread.join.assert_called_once_with(timeout=10.0)


# ============================================================================
# Test SecondbrainMemoryProvider - Tool Schemas
# ============================================================================

class TestToolSchemas:
    """Tests for tool schema methods."""

    def test_get_tool_schemas_returns_empty_when_not_available(self):
        """Test returns empty list when client is None."""
        provider = SecondbrainMemoryProvider()
        provider._client = None

        schemas = provider.get_tool_schemas()
        assert schemas == []

    def test_get_tool_schemas_returns_all_tools(self, provider):
        """Test returns all tool schemas when available."""
        schemas = provider.get_tool_schemas()

        assert len(schemas) == 6
        tool_names = [s["name"] for s in schemas]
        assert "secondbrain_read" in tool_names
        assert "secondbrain_search" in tool_names
        assert "secondbrain_write" in tool_names
        assert "secondbrain_update" in tool_names
        assert "secondbrain_backlinks" in tool_names
        assert "secondbrain_references" in tool_names

    def test_read_schema_structure(self):
        """Test read tool schema has correct structure."""
        assert READ_SCHEMA["name"] == "secondbrain_read"
        assert "doc_id" in READ_SCHEMA["parameters"]["properties"]
        assert READ_SCHEMA["parameters"]["required"] == ["doc_id"]

    def test_search_schema_structure(self):
        """Test search tool schema has correct structure."""
        assert SEARCH_SCHEMA["name"] == "secondbrain_search"
        assert "query" in SEARCH_SCHEMA["parameters"]["properties"]
        assert SEARCH_SCHEMA["parameters"]["required"] == ["query"]

    def test_write_schema_structure(self):
        """Test write tool schema has correct structure."""
        assert WRITE_SCHEMA["name"] == "secondbrain_write"
        props = WRITE_SCHEMA["parameters"]["properties"]
        assert "doc_id" in props
        assert "content" in props
        assert "doc_type" in props

    def test_update_schema_structure(self):
        """Test update tool schema has correct structure."""
        assert UPDATE_SCHEMA["name"] == "secondbrain_update"
        props = UPDATE_SCHEMA["parameters"]["properties"]
        assert "doc_id" in props
        assert "content" in props
        assert "metadata" in props


# ============================================================================
# Test SecondbrainMemoryProvider - Handle Tool Call
# ============================================================================

class TestHandleToolCall:
    """Tests for handle_tool_call method."""

    def test_returns_error_when_not_available(self):
        """Test returns error when client is None."""
        provider = SecondbrainMemoryProvider()
        provider._client = None

        result = provider.handle_tool_call("secondbrain_read", {"doc_id": "test"})
        assert "error" in result.lower()

    def test_handles_secondbrain_read(self, provider):
        """Test handling of secondbrain_read tool."""
        provider._client.call_tool.return_value = {
            "doc_id": "test",
            "content": "Test content",
        }

        result = provider.handle_tool_call(
            "secondbrain_read",
            {"doc_id": "test"}
        )

        assert "test" in result
        assert "Test content" in result

    def test_handles_secondbrain_read_missing_doc_id(self, provider):
        """Test secondbrain_read returns error without doc_id."""
        result = provider.handle_tool_call("secondbrain_read", {})
        assert "error" in result.lower()

    def test_handles_secondbrain_search(self, provider):
        """Test handling of secondbrain_search tool."""
        provider._client.call_tool.return_value = {
            "results": [{"doc_id": "doc1", "chunk_text": "content"}]
        }

        result = provider.handle_tool_call(
            "secondbrain_search",
            {"query": "test", "limit": 5}
        )

        assert "doc1" in result

    def test_handles_secondbrain_search_missing_query(self, provider):
        """Test secondbrain_search returns error without query."""
        result = provider.handle_tool_call("secondbrain_search", {})
        assert "error" in result.lower()

    def test_handles_secondbrain_write(self, provider):
        """Test handling of secondbrain_write tool."""
        provider._client.call_tool.return_value = {"success": True, "doc_id": "new-doc"}

        result = provider.handle_tool_call(
            "secondbrain_write",
            {"doc_id": "new-doc", "content": "New content", "doc_type": "note"}
        )

        assert "success" in result.lower() or "new-doc" in result

    def test_handles_secondbrain_write_missing_params(self, provider):
        """Test secondbrain_write returns error without required params."""
        result = provider.handle_tool_call("secondbrain_write", {})
        assert "error" in result.lower()

    def test_handles_secondbrain_write_missing_content(self, provider):
        """Test secondbrain_write returns error when content is missing."""
        result = provider.handle_tool_call("secondbrain_write", {"doc_id": "test"})
        assert "error" in result.lower()

    def test_handles_secondbrain_update(self, provider):
        """Test handling of secondbrain_update tool."""
        provider._client.call_tool.return_value = {"success": True}

        result = provider.handle_tool_call(
            "secondbrain_update",
            {"doc_id": "existing-doc", "content": "More content"}
        )

        assert "success" in result.lower() or "existing-doc" in result

    def test_handles_secondbrain_update_missing_params(self, provider):
        """Test secondbrain_update returns error without required params."""
        result = provider.handle_tool_call("secondbrain_update", {})
        assert "error" in result.lower()

    def test_handles_secondbrain_update_missing_content_and_metadata(self, provider):
        """Test secondbrain_update returns error when both content and metadata are missing."""
        result = provider.handle_tool_call("secondbrain_update", {"doc_id": "test"})
        assert "error" in result.lower()

    def test_handles_secondbrain_backlinks(self, provider):
        """Test handling of secondbrain_backlinks tool."""
        provider._client.call_tool.return_value = {
            "doc_id": "test",
            "backlinks": ["doc1", "doc2"]
        }

        result = provider.handle_tool_call(
            "secondbrain_backlinks",
            {"doc_id": "test"}
        )

        assert "backlinks" in result

    def test_handles_secondbrain_references(self, provider):
        """Test handling of secondbrain_references tool."""
        provider._client.call_tool.return_value = {
            "doc_id": "test",
            "references": ["doc3", "doc4"]
        }

        result = provider.handle_tool_call(
            "secondbrain_references",
            {"doc_id": "test"}
        )

        assert "references" in result

    def test_handles_unknown_tool(self, provider):
        """Test handling of unknown tool."""
        result = provider.handle_tool_call("unknown_tool", {})
        assert "error" in result.lower()
        assert "Unknown tool" in result


# ============================================================================
# Test SecondbrainMemoryProvider - Shutdown
# ============================================================================

class TestShutdown:
    """Tests for shutdown method."""

    def test_waits_for_sync_thread(self, provider):
        """Test shutdown waits for sync thread."""
        mock_thread = MagicMock()
        mock_thread.is_alive.return_value = True
        provider._sync_thread = mock_thread

        provider.shutdown()

        mock_thread.join.assert_called_once_with(timeout=5.0)

    def test_waits_for_prefetch_thread(self, provider):
        """Test shutdown waits for prefetch thread."""
        mock_thread = MagicMock()
        mock_thread.is_alive.return_value = True
        provider._prefetch_thread = mock_thread

        provider.shutdown()

        mock_thread.join.assert_called_once_with(timeout=3.0)

    def test_sets_client_to_none(self, provider):
        """Test shutdown sets client to None."""
        provider.shutdown()
        assert provider._client is None


# ============================================================================
# Test Plugin Registration
# ============================================================================

class TestPluginRegistration:
    """Tests for plugin registration."""

    def test_register_creates_provider(self):
        """Test register function creates and registers provider."""
        mock_ctx = MagicMock()

        register(mock_ctx)

        mock_ctx.register_memory_provider.assert_called_once()
        registered_provider = mock_ctx.register_memory_provider.call_args[0][0]
        assert isinstance(registered_provider, SecondbrainMemoryProvider)
        assert registered_provider.name == "secondbrain"


# ============================================================================
# Test Tool Schemas List
# ============================================================================

class TestAllToolSchemas:
    """Tests for the ALL_TOOL_SCHEMAS list."""

    def test_contains_all_schemas(self):
        """Test ALL_TOOL_SCHEMAS contains all expected tools."""
        tool_names = [s["name"] for s in ALL_TOOL_SCHEMAS]
        expected = [
            "secondbrain_read",
            "secondbrain_search",
            "secondbrain_write",
            "secondbrain_update",
            "secondbrain_backlinks",
            "secondbrain_references",
        ]
        assert tool_names == expected

    def test_all_schemas_have_required_fields(self):
        """Test all schemas have required fields."""
        for schema in ALL_TOOL_SCHEMAS:
            assert "name" in schema
            assert "description" in schema
            assert "parameters" in schema
            assert "type" in schema["parameters"]
            assert schema["parameters"]["type"] == "object"


# ============================================================================
# Additional coverage tests
# ============================================================================

class TestClientErrorHandling:
    """Tests for client error handling."""

    @patch("plugins.memory.secondbrain.requests.post")
    def test_call_tool_handles_error_response(self, mock_post):
        """Test call_tool handles MCP error response."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"error": {"message": "Tool not found"}}
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        client = SecondbrainClient("http://localhost:18764")
        result = client.call_tool("unknown_tool")

        assert "error" in result

    @patch.dict(os.environ, {"SECONDBRAIN_USE_SIMPLE_HTTP": "1"})
    @patch("plugins.memory.secondbrain.requests.post")
    def test_call_tool_handles_malformed_response(self, mock_post):
        """Test call_tool handles malformed response (uses fallback HTTP)."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"unexpected": "format"}
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        client = SecondbrainClient("http://localhost:18764")
        result = client.call_tool("health_check")

        assert "unexpected" in result

    @patch("mcp.client.streamable_http.streamablehttp_client")
    @patch("mcp.ClientSession")
    def test_health_check_handles_exception(self, mock_session_class, mock_client):
        """Test health_check handles exception."""
        # Mock the MCP client to raise exception
        mock_client.side_effect = Exception("Connection failed")

        client = SecondbrainClient("http://localhost:18764")
        result = client.health_check()

        assert result is False

    @patch("mcp.client.streamable_http.streamablehttp_client")
    @patch("mcp.ClientSession")
    def test_call_tool_exception_in_call_tool(self, mock_session_class, mock_client):
        """Test call_tool exception handling directly."""
        # Mock the MCP client to raise exception
        mock_client.side_effect = Exception("Network error")

        client = SecondbrainClient("http://localhost:18764")
        result = client.call_tool("test_tool")

        assert "error" in result

    def test_health_check_exception_in_call_tool(self):
        """Test health_check catches exception from call_tool."""
        # Create a client and mock call_tool directly
        client = SecondbrainClient("http://localhost:18764")
        
        # Mock call_tool to raise an exception
        original_call_tool = client.call_tool
        def raise_exception(*args, **kwargs):
            raise Exception("Test exception")
        client.call_tool = raise_exception
        
        result = client.health_check()
        
        assert result is False


class TestProviderConfigLoading:
    """Tests for configuration loading."""

    def test_is_available_with_empty_config_file(self, temp_hermes_home):
        """Test is_available returns False when config file has no url."""
        config_path = temp_hermes_home / "secondbrain.json"
        config_path.write_text(json.dumps({}))

        with mock.patch("hermes_constants.get_hermes_home", return_value=temp_hermes_home):
            with mock.patch.dict(os.environ, {}, clear=True):
                provider = SecondbrainMemoryProvider()
                assert provider.is_available() is False

    def test_is_available_config_file_parse_error(self, temp_hermes_home):
        """Test is_available handles config file parse error."""
        config_path = temp_hermes_home / "secondbrain.json"
        config_path.write_text("invalid json{")

        with mock.patch("hermes_constants.get_hermes_home", return_value=temp_hermes_home):
            with mock.patch.dict(os.environ, {}, clear=True):
                provider = SecondbrainMemoryProvider()
                # Should return False due to parse error
                assert provider.is_available() is False

    def test_initialize_loads_config_file(self, temp_hermes_home):
        """Test initialize loads URL from config file."""
        config_path = temp_hermes_home / "secondbrain.json"
        config_path.write_text(json.dumps({
            "url": "http://config:9999",
            "api_key": "config-key"
        }))

        with mock.patch.dict(os.environ, {}, clear=True):
            with patch("plugins.memory.secondbrain.SecondbrainClient") as MockClient:
                mock_client = MagicMock()
                mock_client.health_check.return_value = True
                MockClient.return_value = mock_client

                provider = SecondbrainMemoryProvider()
                provider.initialize("test-session", hermes_home=str(temp_hermes_home))

                assert provider._client is not None
                # Verify client was created with config file URL
                call_kwargs = MockClient.call_args[1]
                assert call_kwargs["base_url"] == "http://config:9999"
                assert call_kwargs["api_key"] == "config-key"

    def test_initialize_skips_missing_config_file(self, temp_hermes_home):
        """Test initialize skips config file when it doesn't exist."""
        with mock.patch.dict(os.environ, {}, clear=True):
            with patch("plugins.memory.secondbrain.SecondbrainClient") as MockClient:
                mock_client = MagicMock()
                mock_client.health_check.return_value = True
                MockClient.return_value = mock_client

                provider = SecondbrainMemoryProvider()
                # Don't create config file - it shouldn't exist
                provider.initialize("test-session", hermes_home=str(temp_hermes_home))

                # Should have no client because no URL from env or config
                assert provider._client is None

    def test_initialize_config_file_error(self, temp_hermes_home, caplog):
        """Test initialize handles config file parse error."""
        config_path = temp_hermes_home / "secondbrain.json"
        config_path.write_text("invalid json{")

        with mock.patch.dict(os.environ, {}, clear=True):
            with patch("plugins.memory.secondbrain.SecondbrainClient") as MockClient:
                mock_client = MagicMock()
                mock_client.health_check.return_value = True
                MockClient.return_value = mock_client

                provider = SecondbrainMemoryProvider()
                provider.initialize("test-session", hermes_home=str(temp_hermes_home))

                # Should still work with env vars or defaults
                assert provider._client is None  # No URL from env, so inactive

    def test_save_config_file_read_error(self, temp_hermes_home):
        """Test save_config handles file read error."""
        config_path = temp_hermes_home / "secondbrain.json"
        
        # Create a file with valid JSON first
        config_path.write_text("{}")
        
        # Now mock Path.read_text to raise an error
        with patch('pathlib.Path.read_text') as mock_read:
            mock_read.side_effect = IOError("Cannot read file")
            
            provider = SecondbrainMemoryProvider()
            # Should not raise
            provider.save_config({"url": "http://test"}, str(temp_hermes_home))


class TestPrefetchErrorHandling:
    """Tests for prefetch error handling."""

    def test_prefetch_handles_exception(self, provider, caplog):
        """Test prefetch handles exceptions gracefully."""
        provider._client.call_tool.side_effect = Exception("Search failed")

        result = provider.prefetch("test query")

        assert result == ""


class TestQueuePrefetchThreading:
    """Tests for queue_prefetch threading."""

    def test_queue_prefetch_starts_thread(self, provider):
        """Test queue_prefetch starts background thread."""
        provider._client.call_tool.return_value = {"results": []}

        provider.queue_prefetch("test query")

        assert provider._prefetch_thread is not None
        assert provider._prefetch_thread.daemon is True

        # Clean up
        if provider._prefetch_thread:
            provider._prefetch_thread.join(timeout=1.0)

    def test_queue_prefetch_handles_exception(self, provider):
        """Test queue_prefetch handles exceptions in background thread."""
        provider._client.call_tool.side_effect = Exception("Search failed")

        # Should not raise
        provider.queue_prefetch("test query")

        # Clean up
        if provider._prefetch_thread:
            provider._prefetch_thread.join(timeout=1.0)

    def test_queue_prefetch_with_results(self, provider):
        """Test queue_prefetch stores results from successful search."""
        provider._client.call_tool.return_value = {
            "results": [
                {"doc_id": "doc1", "chunk_text": "Content"}
            ]
        }

        provider.queue_prefetch("test query")

        # Wait for thread
        if provider._prefetch_thread:
            provider._prefetch_thread.join(timeout=1.0)

        # Check prefetch result was stored
        with provider._prefetch_lock:
            assert "doc1" in provider._prefetch_result

    def test_queue_prefetch_waits_for_existing_thread(self, provider):
        """Test queue_prefetch waits for existing thread."""
        # Create a mock running thread
        mock_thread = MagicMock()
        mock_thread.is_alive.return_value = True
        mock_thread.join.return_value = None
        provider._prefetch_thread = mock_thread

        provider.queue_prefetch("test query")

        # Should have joined the existing thread
        mock_thread.join.assert_called_once_with(timeout=5.0)


class TestOnTurnStart:
    """Tests for on_turn_start."""

    def test_on_turn_start_clears_prefetch_result(self, provider):
        """Test on_turn_start clears prefetch result."""
        provider._prefetch_result = "old result"

        provider.on_turn_start(1, "test message")

        assert provider._prefetch_result == ""


class TestOnSessionEnd:
    """Tests for on_session_end."""

    def test_on_session_end_no_thread(self, provider):
        """Test on_session_end handles no thread gracefully."""
        provider._sync_thread = None

        # Should not raise
        provider.on_session_end([])

    def test_on_session_end_waits_for_thread(self, provider):
        """Test on_session_end waits for running thread."""
        mock_thread = MagicMock()
        mock_thread.is_alive.return_value = True
        provider._sync_thread = mock_thread

        provider.on_session_end([])

        mock_thread.join.assert_called_once_with(timeout=10.0)


class TestHandleToolCallUpdate:
    """Tests for handle_tool_call with update."""

    def test_handle_secondbrain_update_with_content_only(self, provider):
        """Test secondbrain_update with content only (no metadata)."""
        provider._client.call_tool.return_value = {"success": True}

        result = provider.handle_tool_call(
            "secondbrain_update",
            {"doc_id": "test-doc", "content": "new content"}
        )

        # Should have called with content
        call_args = provider._client.call_tool.call_args
        assert call_args[1]["content"] == "new content"

    def test_handle_secondbrain_update_with_metadata_only(self, provider):
        """Test secondbrain_update with metadata only (no content)."""
        provider._client.call_tool.return_value = {"success": True}

        result = provider.handle_tool_call(
            "secondbrain_update",
            {"doc_id": "test-doc", "metadata": {"tag": "important"}}
        )

        # Should have called with metadata
        call_args = provider._client.call_tool.call_args
        assert call_args[1]["metadata"] == {"tag": "important"}

    def test_handle_tool_call_exception(self, provider):
        """Test handle_tool_call handles exceptions."""
        provider._client.call_tool.side_effect = Exception("Tool failed")

        result = provider.handle_tool_call("secondbrain_read", {"doc_id": "test"})

        assert "error" in result.lower()

    def test_handle_secondbrain_backlinks(self, provider):
        """Test secondbrain_backlinks tool."""
        provider._client.call_tool.return_value = {"doc_id": "test", "backlinks": []}

        result = provider.handle_tool_call(
            "secondbrain_backlinks",
            {"doc_id": "test"}
        )

        assert "backlinks" in result

    def test_handle_secondbrain_backlinks_missing_doc_id(self, provider):
        """Test secondbrain_backlinks returns error when doc_id is missing."""
        result = provider.handle_tool_call("secondbrain_backlinks", {})
        assert "error" in result.lower()

    def test_handle_secondbrain_references(self, provider):
        """Test secondbrain_references tool."""
        provider._client.call_tool.return_value = {"doc_id": "test", "references": []}

        result = provider.handle_tool_call(
            "secondbrain_references",
            {"doc_id": "test"}
        )

        assert "references" in result

    def test_handle_secondbrain_references_missing_doc_id(self, provider):
        """Test secondbrain_references returns error when doc_id is missing."""
        result = provider.handle_tool_call("secondbrain_references", {})
        assert "error" in result.lower()


class TestShutdown:
    """Tests for shutdown."""

    def test_shutdown_no_threads(self, provider):
        """Test shutdown handles no threads gracefully."""
        provider._sync_thread = None
        provider._prefetch_thread = None

        # Should not raise
        provider.shutdown()

    def test_shutdown_waits_for_prefetch_thread(self, provider):
        """Test shutdown waits for prefetch thread."""
        mock_thread = MagicMock()
        mock_thread.is_alive.return_value = True
        provider._sync_thread = None
        provider._prefetch_thread = mock_thread

        provider.shutdown()

        mock_thread.join.assert_called_once_with(timeout=3.0)

    def test_shutdown_waits_for_sync_thread(self, provider):
        """Test shutdown waits for sync thread."""
        mock_thread = MagicMock()
        mock_thread.is_alive.return_value = True
        provider._sync_thread = mock_thread
        provider._prefetch_thread = None

        provider.shutdown()

        mock_thread.join.assert_called_once_with(timeout=5.0)


class TestProviderDirectCalls:
    """Tests that make direct calls without heavy mocking."""

    def test_provider_initialization_with_valid_url(self):
        """Test provider initialization with valid URL from env."""
        with mock.patch.dict(os.environ, {"SECONDBRAIN_URL": "http://localhost:18764"}):
            with patch("plugins.memory.secondbrain.SecondbrainClient") as MockClient:
                mock_client = MagicMock()
                mock_client.health_check.return_value = True
                MockClient.return_value = mock_client

                # Need to re-import to get fresh provider
                from plugins.memory.secondbrain import SecondbrainMemoryProvider
                provider = SecondbrainMemoryProvider()
                provider.initialize("test-session", hermes_home="/tmp")

                # This should work
                assert provider._client is not None

    def test_sync_turn_with_explicit_session_id(self, provider):
        """Test sync_turn uses explicit session_id when provided."""
        provider._client.call_tool.return_value = {"success": True}

        # Call with explicit session_id
        provider.sync_turn("user", "assistant", session_id="explicit-session")

        # Wait for thread
        if provider._sync_thread:
            provider._sync_thread.join(timeout=5.0)

        # Should have called with the explicit session
        provider._client.call_tool.assert_called()
        call_kwargs = provider._client.call_tool.call_args[1]
        assert "explicit-session" in call_kwargs["doc_id"]

    def test_sync_turn_waits_for_existing_thread(self, provider):
        """Test sync_turn waits for existing thread."""
        provider._client.call_tool.return_value = {"success": True}
        
        # Create a mock running thread
        mock_thread = MagicMock()
        mock_thread.is_alive.return_value = True
        mock_thread.join.return_value = None
        provider._sync_thread = mock_thread

        provider.sync_turn("user", "assistant")

        # Should have joined the existing thread
        mock_thread.join.assert_called()

    def test_sync_turn_handles_exception(self, provider):
        """Test sync_turn handles exceptions in background thread."""
        provider._client.call_tool.side_effect = Exception("Update failed")

        # Should not raise
        provider.sync_turn("user", "assistant")

        # Wait for thread
        if provider._sync_thread:
            provider._sync_thread.join(timeout=1.0)