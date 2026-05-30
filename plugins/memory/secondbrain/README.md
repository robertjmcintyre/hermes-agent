# Secondbrain Memory Provider

A Hermes Agent memory provider plugin that integrates with Secondbrain - a file-based memory system with ChromaDB vector search.

## Overview

Secondbrain provides persistent memory storage with:
- **Semantic search** - Find relevant memories using vector similarity
- **File-based storage** - Memories stored as markdown files
- **Automatic turn sync** - Conversations automatically stored
- **Backlinks and references** - Understand memory relationships

## Requirements

1. **Secondbrain MCP Server** must be running:
   ```bash
   # HTTP mode (for Hermes)
   python -m memory_layer.start_server
   
   # Or with custom port
   MCP_HTTP_PORT=18764 python -m memory_layer.start_server
   ```

2. **ChromaDB** must be running (required by Secondbrain)

## Configuration

### Option 1: Environment Variables

```bash
export SECONDBRAIN_URL=http://localhost:18764
# Optional: export SECONDBRAIN_API_KEY=your-api-key
```

### Option 2: Config File

Create `$HERMES_HOME/secondbrain.json`:

```json
{
  "url": "http://localhost:18764",
  "api_key": "your-api-key (optional)"
}
```

### Option 3: Hermes Setup Command

```bash
hermes memory setup
# Select "secondbrain" as the provider
# Enter the MCP server URL (default: http://localhost:18764)
```

## Activation

In your Hermes `config.yaml`:

```yaml
memory:
  provider: "secondbrain"
  memory_enabled: true
```

## Tools

The provider exposes these tools to the model:

| Tool | Description |
|------|-------------|
| `secondbrain_read` | Read a specific memory by ID |
| `secondbrain_search` | Search memories using semantic similarity |
| `secondbrain_write` | Create a new memory document |
| `secondbrain_update` | Append content to an existing memory |
| `secondbrain_backlinks` | Find documents linking to a given document |
| `secondbrain_references` | Find documents a given document links to |

## Usage Examples

### Search for relevant memories
```
User: What did I work on last week?
Model calls: secondbrain_search({query: "projects work last week"})
```

### Create a new memory
```
Model calls: secondbrain_write({
  doc_id: "projects/architecture",
  content: "Designing a new microservices architecture...",
  doc_type: "project"
})
```

### Update an existing memory
```
Model calls: secondbrain_update({
  doc_id: "sessions/current",
  content: "Added new feature: user authentication"
})
```

## Architecture

```
┌─────────────────────────────────────────┐
│           Secondbrain                   │
│    (ChromaDB + markdown files)          │
└─────────────────────────────────────────┘
                    ▲
                    │ HTTP
                    │
┌──────────────────┴──────────────────┐
│     SecondbrainMemoryProvider        │
│     (Hermes plugin)                  │
└──────────────────────────────────────┘
```

## Troubleshooting

### Provider not available
- Check that the MCP server is running: `curl http://localhost:18764/health_check`
- Verify URL in config: `echo $SECONDBRAIN_URL`
- Check hermes logs for errors

### Health check fails
- Ensure ChromaDB is running
- Check the MCP server logs
- Verify network connectivity

### Tools not working
- Ensure provider is active: `hermes memory status`
- Check that tools are registered: `hermes tools | grep secondbrain`

## See Also

- [Secondbrain Memory Layer](https://github.com/secondbrain/secondbrain)
- [Hermes Memory Provider Documentation](../../docs/developer-guide/memory-provider-plugin.md)