"""
MCP SERVER (Model Context Protocol)
=====================================
MCP is a standard protocol for connecting AI models to external tools/data.
Think of it as a "USB standard" for AI tools — any MCP-compatible client
can connect to any MCP-compatible server.

This simple MCP server exposes one tool: reading a document from your docs folder.

HOW MCP WORKS:
  1. Server defines tools with clear input/output schemas
  2. Client (your app) discovers available tools
  3. LLM decides to call a tool
  4. Client sends tool call to server via standard protocol
  5. Server executes and returns result

Run standalone to test: python src/mcp_server.py
"""

import json
import os


class SimpleMCPServer:
    """
    A minimal MCP-like server implementation.
    In production, you'd use the official 'mcp' Python library.
    This version shows the core concepts clearly.
    """

    def __init__(self, docs_dir: str = "docs"):
        self.docs_dir = docs_dir
        self.server_name = "knowledge-assistant-mcp"
        self.version = "1.0.0"

    def list_tools(self) -> list[dict]:
        """
        MCP servers expose their tools via a list_tools() method.
        Each tool has a name, description, and input schema.
        """
        return [
            {
                "name": "read_document",
                "description": "Read the full contents of a document from the docs folder",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "filename": {
                            "type": "string",
                            "description": "Name of the file to read (e.g., 'ai_basics.txt')"
                        }
                    },
                    "required": ["filename"]
                }
            },
            {
                "name": "list_documents",
                "description": "List all available documents in the docs folder",
                "inputSchema": {
                    "type": "object",
                    "properties": {}
                }
            }
        ]

    def call_tool(self, tool_name: str, arguments: dict) -> dict:
        """
        MCP servers handle tool calls via call_tool().
        Returns a result dict with content.
        """
        if tool_name == "list_documents":
            return self._list_documents()
        elif tool_name == "read_document":
            return self._read_document(arguments.get("filename", ""))
        else:
            return {"error": f"Unknown tool: {tool_name}"}

    def _list_documents(self) -> dict:
        try:
            files = [f for f in os.listdir(self.docs_dir) if f.endswith(".txt")]
            return {
                "content": f"Available documents: {', '.join(files)}",
                "files": files
            }
        except Exception as e:
            return {"error": str(e)}

    def _read_document(self, filename: str) -> dict:
        if not filename:
            return {"error": "No filename provided"}

        # Security: prevent path traversal attacks
        safe_path = os.path.join(self.docs_dir, os.path.basename(filename))

        if not os.path.exists(safe_path):
            available = [f for f in os.listdir(self.docs_dir) if f.endswith(".txt")]
            return {
                "error": f"File '{filename}' not found.",
                "available_files": available
            }

        with open(safe_path, "r", encoding="utf-8") as f:
            content = f.read()

        return {
            "content": content,
            "filename": filename,
            "characters": len(content)
        }

    def get_info(self) -> dict:
        """Returns server info — standard in MCP protocol."""
        tools = self.list_tools()
        return {
            "name": self.server_name,
            "version": self.version,
            "tools": [t["name"] for t in tools],
            "tool_count": len(tools)
        }


# Singleton instance to reuse across the app
mcp_server = SimpleMCPServer(docs_dir="docs")


if __name__ == "__main__":
    # Test the MCP server standalone
    print("🔌 MCP Server Info:")
    print(json.dumps(mcp_server.get_info(), indent=2))

    print("\n📋 Available Tools:")
    for tool in mcp_server.list_tools():
        print(f"  • {tool['name']}: {tool['description']}")

    print("\n📂 Listing documents:")
    result = mcp_server.call_tool("list_documents", {})
    print(f"  {result['content']}")

    print("\n📄 Reading ai_basics.txt (first 200 chars):")
    result = mcp_server.call_tool("read_document", {"filename": "ai_basics.txt"})
    if "content" in result:
        print(f"  {result['content'][:200]}...")
    else:
        print(f"  Error: {result.get('error')}")
