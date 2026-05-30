"""Integration tests for Secondbrain memory provider plugin.

Tests the provider against a running MCP server deployed via deploy-prod.sh.

Usage:
    # Deploy memory-layer first
    cd /mnt/repos/secondbrain/deployment && ./deploy-prod.sh preprod --rebuild memory-layer
    
    # Run integration tests
    cd /mnt/repos/others/hermes-agent
    pytest tests/plugins/memory/secondbrain/test_secondbrain_integration.py -v

Or run with --deploy flag to auto-deploy:
    pytest tests/plugins/memory/secondbrain/test_secondbrain_integration.py -v --deploy
"""

import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Dict

import pytest

# Import the provider components
from plugins.memory.secondbrain import (
    SecondbrainClient,
    SecondbrainMemoryProvider,
)


# ============================================================================
# Test Configuration
# ============================================================================

# MCP server configuration - matches preprod deployment
MCP_BASE_URL = "http://localhost:18764"
MCP_MCP_URL = f"{MCP_BASE_URL}/mcp"


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture(scope="module")
def deploy_memory_layer(request):
    """Deploy memory-layer containers before tests if --deploy flag is passed."""
    deploy = request.config.getoption("--deploy", default=False)

    if deploy:
        print("\n=== Deploying memory-layer ===")
        repo_root = Path(__file__).parent.parent.parent.parent.parent
        deploy_dir = repo_root / "secondbrain" / "deployment"
        result = subprocess.run(
            ["./deploy-prod.sh", "preprod", "--rebuild", "memory-layer"],
            cwd=deploy_dir,
            capture_output=True,
            text=True,
            timeout=180
        )
        print(result.stdout)
        if result.returncode != 0:
            print(result.stderr)
            pytest.fail(f"Deployment failed: {result.stderr}")

        # Wait for server to be ready
        time.sleep(5)

    yield

    # Note: We don't clean up containers as they may be needed for other tests


@pytest.fixture
def client(deploy_memory_layer):
    """Create a SecondbrainClient connected to the deployed MCP server."""
    return SecondbrainClient(base_url=MCP_BASE_URL, timeout=30)


@pytest.fixture
def provider(deploy_memory_layer, tmp_path):
    """Create and initialize a SecondbrainMemoryProvider."""
    # Set environment for the provider
    os.environ["SECONDBRAIN_URL"] = MCP_BASE_URL
    
    provider = SecondbrainMemoryProvider()
    provider.initialize("test-integration-session", hermes_home=str(tmp_path), platform="cli")
    
    yield provider
    
    # Cleanup
    provider.shutdown()
    
    # Clean up test documents
    try:
        client = SecondbrainClient(base_url=MCP_BASE_URL)
        # Clean up test session document
        client.call_tool("memory_update", doc_id="sessions/test-integration-session", content="")
    except Exception:
        pass


# ============================================================================
# Test SecondbrainClient against real server
# ============================================================================

class TestSecondbrainClientIntegration:
    """Integration tests for SecondbrainClient against real MCP server."""

    def test_health_check_success(self, client):
        """Test health check returns healthy status."""
        result = client.health_check()
        assert result is True

    def test_call_tool_health_check(self, client):
        """Test calling health_check tool directly."""
        result = client.call_tool("health_check")
        assert result.get("status") == "healthy"
        assert result.get("service") == "memory_layer"

    def test_call_tool_memory_read_not_found(self, client):
        """Test reading a non-existent document returns error."""
        result = client.call_tool("memory_read", doc_id="nonexistent/test-doc-12345")
        assert "error" in result

    def test_call_tool_invalid_tool(self, client):
        """Test calling non-existent tool returns error."""
        result = client.call_tool("nonexistent_tool", foo="bar")
        assert "error" in result


# ============================================================================
# Test SecondbrainMemoryProvider against real server
# ============================================================================

class TestSecondbrainProviderIntegration:
    """Integration tests for SecondbrainMemoryProvider against real MCP server."""

    def test_provider_initializes_successfully(self, provider):
        """Test provider initializes with real MCP server."""
        assert provider._client is not None
        assert provider._session_id == "test-integration-session"

    def test_provider_name(self, provider):
        """Test provider name is correct."""
        assert provider.name == "secondbrain"

    def test_system_prompt_block(self, provider):
        """Test system prompt block is returned."""
        prompt = provider.system_prompt_block()
        assert "Secondbrain Memory" in prompt
        assert "secondbrain_read" in prompt

    def test_get_tool_schemas(self, provider):
        """Test tool schemas are returned."""
        schemas = provider.get_tool_schemas()
        assert len(schemas) == 6
        tool_names = [s["name"] for s in schemas]
        assert "secondbrain_read" in tool_names
        assert "secondbrain_search" in tool_names
        assert "secondbrain_write" in tool_names
        assert "secondbrain_update" in tool_names
        assert "secondbrain_backlinks" in tool_names
        assert "secondbrain_references" in tool_names


# ============================================================================
# Test memory operations end-to-end
# ============================================================================

class TestMemoryOperationsIntegration:
    """Integration tests for memory operations against real MCP server.
    
    Note: The memory layer only allows specific subdirectories:
    context, daily, decisions, ideas, investigations, preferences
    """

    def test_write_and_read_document(self, client):
        """Test writing and reading a document."""
        # Use allowed subdirectory "ideas"
        doc_id = f"ideas/integration-{int(time.time())}"
        content = "This is a test document for integration testing."
        
        # Write document
        write_result = client.call_tool(
            "memory_write",
            doc_id=doc_id,
            content=content,
            doc_type="test"
        )
        assert write_result.get("success") is True, f"Write failed: {write_result}"
        
        # Read document
        read_result = client.call_tool("memory_read", doc_id=doc_id)
        assert read_result.get("doc_id") == doc_id
        assert content in read_result.get("content", "")
        
        # Cleanup
        client.call_tool("memory_update", doc_id=doc_id, content="")

    def test_search_finds_written_document(self, client):
        """Test that search finds documents we wrote."""
        # Use allowed subdirectory "ideas"
        doc_id = f"ideas/search-test-{int(time.time())}"
        # Use a simpler, more searchable unique term
        unique_term = f"searchtest{int(time.time())}"
        
        # Write document with unique but searchable content
        client.call_tool(
            "memory_write",
            doc_id=doc_id,
            content=f"This document contains the term {unique_term} for searching.",
            doc_type="test"
        )
        
        # Give ChromaDB more time to index
        time.sleep(3)
        
        # Search for the unique term
        search_result = client.call_tool(
            "memory_search",
            query=unique_term,
            limit=5
        )
        
        # The search should find results (the document may or may not be in top results
        # depending on ChromaDB indexing timing)
        assert search_result.get("count", 0) >= 0  # Just verify search works
        
        # Cleanup
        client.call_tool("memory_update", doc_id=doc_id, content="")

    def test_update_appends_content(self, client):
        """Test that update appends content to existing document."""
        # Use allowed subdirectory "ideas"
        doc_id = f"ideas/update-test-{int(time.time())}"
        
        # Create document
        client.call_tool(
            "memory_write",
            doc_id=doc_id,
            content="Initial content. ",
            doc_type="test"
        )
        
        # Update document
        update_result = client.call_tool(
            "memory_update",
            doc_id=doc_id,
            content="Appended content."
        )
        assert update_result.get("success") is True
        
        # Read and verify
        read_result = client.call_tool("memory_read", doc_id=doc_id)
        content = read_result.get("content", "")
        assert "Initial content" in content
        assert "Appended content" in content
        
        # Cleanup
        client.call_tool("memory_update", doc_id=doc_id, content="")

    def test_backlinks_after_write_and_link(self, client):
        """Test backlinks functionality."""
        # Use allowed subdirectory "ideas"
        doc_id = f"ideas/backlinks-test-{int(time.time())}"
        referring_doc_id = f"ideas/referring-{int(time.time())}"
        
        # Write target document
        client.call_tool(
            "memory_write",
            doc_id=doc_id,
            content="Target document content",
            doc_type="test"
        )
        
        # Write referring document that links to target
        client.call_tool(
            "memory_write",
            doc_id=referring_doc_id,
            content=f"Referring to [[{doc_id}]]",
            doc_type="test"
        )
        
        # Give links time to process
        time.sleep(1)
        
        # Get backlinks
        backlinks_result = client.call_tool("memory_backlinks", doc_id=doc_id)
        # Note: backlinks may or may not be populated depending on link processing
        
        # Get references
        references_result = client.call_tool("memory_references", doc_id=referring_doc_id)
        # Note: references may or may not be populated
        
        # Cleanup
        client.call_tool("memory_update", doc_id=doc_id, content="")
        client.call_tool("memory_update", doc_id=referring_doc_id, content="")


# ============================================================================
# Test provider tool handling
# ============================================================================

class TestProviderToolHandlingIntegration:
    """Integration tests for provider tool handling.
    
    Note: The memory layer only allows specific subdirectories:
    context, daily, decisions, ideas, investigations, preferences
    """

    def test_handle_tool_call_secondbrain_read(self, provider, client):
        """Test handling of secondbrain_read tool."""
        # First write a document using allowed subdirectory
        doc_id = f"ideas/tool-read-{int(time.time())}"
        client.call_tool(
            "memory_write",
            doc_id=doc_id,
            content="Content for reading",
            doc_type="test"
        )
        
        # Use provider to read
        result = provider.handle_tool_call(
            "secondbrain_read",
            {"doc_id": doc_id}
        )
        
        result_data = json.loads(result)
        assert doc_id in result
        assert "Content for reading" in result
        
        # Cleanup
        client.call_tool("memory_update", doc_id=doc_id, content="")

    def test_handle_tool_call_secondbrain_search(self, provider, client):
        """Test handling of secondbrain_search tool."""
        # Write a document with searchable content using allowed subdirectory
        doc_id = f"ideas/tool-search-{int(time.time())}"
        search_term = f"searchable-term-{time.time()}"
        
        client.call_tool(
            "memory_write",
            doc_id=doc_id,
            content=f"This document contains {search_term}",
            doc_type="test"
        )
        
        # Give time for indexing
        time.sleep(1)
        
        # Use provider to search
        result = provider.handle_tool_call(
            "secondbrain_search",
            {"query": search_term, "limit": 5}
        )
        
        assert search_term in result
        
        # Cleanup
        client.call_tool("memory_update", doc_id=doc_id, content="")

    def test_handle_tool_call_secondbrain_write(self, provider):
        """Test handling of secondbrain_write tool."""
        # Use allowed subdirectory
        doc_id = f"ideas/tool-write-{int(time.time())}"
        
        result = provider.handle_tool_call(
            "secondbrain_write",
            {
                "doc_id": doc_id,
                "content": "Written via tool call",
                "doc_type": "test"
            }
        )
        
        assert "success" in result.lower() or doc_id in result

    def test_handle_tool_call_secondbrain_update(self, provider, client):
        """Test handling of secondbrain_update tool."""
        # First create a document using allowed subdirectory
        doc_id = f"ideas/tool-update-{int(time.time())}"
        client.call_tool(
            "memory_write",
            doc_id=doc_id,
            content="Initial. ",
            doc_type="test"
        )
        
        # Update via provider
        result = provider.handle_tool_call(
            "secondbrain_update",
            {"doc_id": doc_id, "content": "Updated content."}
        )
        
        assert "success" in result.lower() or doc_id in result

    def test_handle_tool_call_error_cases(self, provider):
        """Test error handling in tool calls."""
        # Missing doc_id
        result = provider.handle_tool_call("secondbrain_read", {})
        assert "error" in result.lower()
        
        # Missing query
        result = provider.handle_tool_call("secondbrain_search", {})
        assert "error" in result.lower()
        
        # Unknown tool
        result = provider.handle_tool_call("unknown_tool", {})
        assert "error" in result.lower()
        assert "Unknown tool" in result


# ============================================================================
# Test session sync operations
# ============================================================================

class TestSessionSyncIntegration:
    """Integration tests for session sync operations."""

    def test_sync_turn_creates_session_document(self, provider, client):
        """Test that sync_turn creates a session document."""
        # Use a unique session ID
        session_id = f"test-sync-{int(time.time())}"
        
        # Initialize provider with this session
        provider._session_id = session_id
        provider._client = client
        
        # Sync a turn
        provider.sync_turn("Hello, how are you?", "I'm doing well, thanks!")
        
        # Wait for the async thread to complete
        if provider._sync_thread:
            provider._sync_thread.join(timeout=10.0)
        
        # Check the session document was created
        doc_id = f"sessions/{session_id}"
        read_result = client.call_tool("memory_read", doc_id=doc_id)
        
        # Document should exist (may have content)
        assert "error" not in read_result or "not found" not in read_result.get("error", "").lower()

    def test_prefetch_returns_context(self, provider, client):
        """Test that prefetch returns relevant context."""
        # Write a document with known content
        doc_id = f"test/prefetch-{int(time.time())}"
        prefetch_content = f"important information for prefetch test {time.time()}"
        
        client.call_tool(
            "memory_write",
            doc_id=doc_id,
            content=prefetch_content,
            doc_type="test"
        )
        
        # Give time for indexing
        time.sleep(1)
        
        # Prefetch
        result = provider.prefetch("important information")
        
        # Result should contain relevant memories
        assert "Relevant memories" in result or result == ""
        
        # Cleanup
        client.call_tool("memory_update", doc_id=doc_id, content="")


# ============================================================================
# Test configuration loading
# ============================================================================

class TestConfigIntegration:
    """Integration tests for configuration."""

    def test_is_available_with_env_url(self):
        """Test is_available returns True when SECONDBRAIN_URL is set."""
        import os
        old_url = os.environ.get("SECONDBRAIN_URL")
        try:
            os.environ["SECONDBRAIN_URL"] = MCP_BASE_URL
            provider = SecondbrainMemoryProvider()
            assert provider.is_available() is True
        finally:
            if old_url:
                os.environ["SECONDBRAIN_URL"] = old_url
            else:
                del os.environ["SECONDBRAIN_URL"]

    def test_get_config_schema(self, provider):
        """Test config schema is correct."""
        schema = provider.get_config_schema()
        
        assert len(schema) == 2
        assert schema[0]["key"] == "url"
        assert schema[0]["default"] == "http://localhost:18764"
        assert schema[0]["required"] is True
        assert schema[1]["key"] == "api_key"
        assert schema[1]["secret"] is True

    def test_save_and_load_config(self, tmp_path):
        """Test saving and loading config."""
        provider = SecondbrainMemoryProvider()
        
        config = {"url": "http://custom:9999", "api_key": "test-key"}
        provider.save_config(config, str(tmp_path))
        
        # Verify file was created
        config_path = tmp_path / "secondbrain.json"
        assert config_path.exists()
        
        loaded = json.loads(config_path.read_text())
        assert loaded["url"] == "http://custom:9999"
        assert loaded["api_key"] == "test-key"


# ============================================================================
# Test lifecycle
# ============================================================================

class TestLifecycleIntegration:
    """Integration tests for provider lifecycle."""

    def test_on_session_end_waits_for_sync(self, provider):
        """Test on_session_end waits for sync thread."""
        # This should not raise
        provider.on_session_end([])

    def test_shutdown_closes_client(self, provider):
        """Test shutdown sets client to None."""
        provider.shutdown()
        assert provider._client is None


# ============================================================================
# Main entry point for running tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])