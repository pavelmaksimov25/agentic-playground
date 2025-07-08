import logging
import os
from pathlib import Path
from aiohttp import web
from dotenv import load_dotenv
from azure.core.credentials import AzureKeyCredential
from azure.identity import AzureDeveloperCliCredential, DefaultAzureCredential
from backend.tools import _generate_report_tool, _generate_report_tool_schema, Tool
from backend.rtmt import RTMiddleTier

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("voicetps")

async def create_app():
    if not os.environ.get("RUNNING_IN_PRODUCTION"):
        logger.info("Running in development mode, loading from .env file")
        load_dotenv()
    llm_endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
    llm_deployment = os.environ.get("AZURE_VOICE_COMPLETION_DEPLOYMENT_NAME")
    llm_key = os.environ.get("AZURE_OPENAI_API_KEY")

    credential = None
    
    if not llm_key:
        if tenant_id := os.environ.get("AZURE_TENANT_ID"):
            logger.info(
                "Using AzureDeveloperCliCredential with tenant_id %s", tenant_id)
            credential = AzureDeveloperCliCredential(
                tenant_id=tenant_id, process_timeout=60)
        else:
            logger.info("Using DefaultAzureCredential")
            credential = DefaultAzureCredential()
    llm_credential = AzureKeyCredential(llm_key) if llm_key else credential
    
    app = web.Application()

    rtmt = RTMiddleTier(llm_endpoint, llm_deployment, llm_credential)

    rtmt.system_message = (
       "You are a helpful assistant that maintains a conversation with the user, while asking questions according to a specific script.\n"
        "The user is an employee who is driving from a customer meeting and talking to you hands-free in the car. "
        "You MUST start the conversation by asking the user if he or she filed hit TPS resport:\n"
        "If the user says yes, you can proceed you are happy and should wish the user a happy day.\n"
        "If the user says no, you should push the user to provide the correct details and make sure that the report is filed today.\n"
        "You should ask the user to file the TPS report and then must proceed with the conversation to retrieve the following information.\n"
        "1. The number of the TPS report id which should be a three digit number. \n"
        "2. The customer that the TPS report is for.\n"
        "3. The amount of hours spent of the TPS report\n"
        "4. The status of the TPS which can be active, done or postponed?\n"
        "After you have gone through all the questions in the script, output a valid JSON file to the user by calling the 'generate_report' function,\n "
        "with the schema definition being various customer demo and product attributes derived from the conversation.\n "
        "You must engage the user in a conversation and ask the questions in the script. The user will provide the answers to the questions."
    )

    rtmt.tools["generate_report"] = Tool(
        target=_generate_report_tool, schema=_generate_report_tool_schema
    )
        
    rtmt.attach_to_app(app, "/realtime")

    # Serve static files and index.html
    current_directory = Path(__file__).parent  # Points to 'app' directory
    static_directory = current_directory / 'static'

    # Ensure static directory exists
    if not static_directory.exists():
        raise FileNotFoundError("Static directory not found at expected path: {}".format(static_directory))

    # Serve index.html at root
    async def index(request):
        return web.FileResponse(static_directory / 'index.html')

    app.router.add_get('/', index)
    app.router.add_static('/static/', path=str(static_directory), name='static')

    return app

if __name__ == "__main__":
    host = os.environ.get("HOST", "localhost")
    port = int(os.environ.get("PORT", 8765))
    web.run_app(create_app(), host=host, port=port)
