import { FunctionTool, MCPToolset } from '@google/adk';


export function getAgentTools(): (FunctionTool | MCPToolset)[] {
  const GOOGLE_MAPS_API_KEY = process.env.GOOGLE_MAPS_API_KEY;

  return [
    ...(GOOGLE_MAPS_API_KEY
      ? [
          new MCPToolset({
            type: 'StreamableHTTPConnectionParams',
            url: process.env.MAPS_MCP_URL || 'https://mapstools.googleapis.com/mcp',
            transportOptions: {
              requestInit: {
                headers: {
                  'X-Goog-Api-Key': GOOGLE_MAPS_API_KEY,
                  'Content-Type': 'application/json',
                  Accept: 'application/json, text/event-stream'
                }
              }
            }
          })
        ]
      : [])
  ];
}
