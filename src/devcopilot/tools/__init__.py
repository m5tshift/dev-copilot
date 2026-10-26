"""Tool implementations as plain Python functions.

They know nothing about LLMs, LangChain or MCP: the MCP servers in
`devcopilot/servers/` publish them, and agents receive them through
`devcopilot/mcp_client.py`.
"""
