import re
from typing import Any

from azure.core.credentials import AzureKeyCredential
from azure.identity import DefaultAzureCredential

from backend.rtmt import RTMiddleTier, Tool, ToolResult, ToolResultDirection

async def _generate_report_tool(args: Any) -> ToolResult:
    report = {
        "tps_report_id": args["tps_report_id"],
        "customer_name": args["customer_name"],
        "hours_spent": args["hours_spent"],
        "status": args["status"]
    }
    # Return the result to the client
    return ToolResult(report, ToolResultDirection.TO_CLIENT)

# Define the schema for the 'generate_report' tool
_generate_report_tool_schema = {
    "type": "function",
    "name": "generate_report",
    "description": "Generates a JSON report of the TPS report derived from the conversation.",
    "parameters": {
        "type": "object",
        "properties": {
            "tps_report_id": {
                "type": "string",
                "description": "The report id of the TPS report"
            },
            "customer_name": {
                "type": "string",
                "description": "The name of the customer."
            },
            "hours_spent": {
                "type": "string",
                "description": "The amount of hours spent on the TPS report."
            },
            "status": {
                "type": "string",
                "description": "The current status of the TPS report."
            }
        },
        "required": ["tps_report_id", "customer_name", "hours_spent", "status"],
        "additionalProperties": False
    }
}
