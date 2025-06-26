import os
import asyncio
import logging
import sys
from dotenv import load_dotenv

import vertexai
from vertexai import agent_engines
from google.adk.sessions import VertexAiSessionService

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Load Environment Variables ---
load_dotenv()

# --- Configuration ---
GOOGLE_CLOUD_PROJECT = os.getenv("GOOGLE_CLOUD_PROJECT", "api-project-507154614599")
VERTEX_AI_LOCATION = os.getenv("VERTEX_AI_LOCATION", "us-central1")
# Get AGENT_ENGINE_ID from .env or command line if passed
# AGENT_ENGINE_ID = os.getenv("AGENT_ENGINE_ID")
AGENT_ENGINE_ID = os.getenv("PRICES_COMPARE_AGENT_ENGINE_ID")

async def destroy_agent_engine(agent_engine_id: str):
    """
    Deletes all sessions associated with an Agent Engine and then deletes the Agent Engine itself.
    """
    if not agent_engine_id:
        logger.error("AGENT_ENGINE_ID is not set. Please ensure it's in your .env file or passed as an argument.")
        return

    logger.info(f"Attempting to destroy Agent Engine: {agent_engine_id}")

    try:
        vertexai.init(
            project=GOOGLE_CLOUD_PROJECT,
            location=VERTEX_AI_LOCATION,
        )
        logger.info(f"Vertex AI SDK initialized. Project: {GOOGLE_CLOUD_PROJECT}, Location: {VERTEX_AI_LOCATION}")

        adk_session_service = VertexAiSessionService(
            project=GOOGLE_CLOUD_PROJECT,
            location=VERTEX_AI_LOCATION
        )
        logger.info("Vertex AI ADK Session Service initialized.")

        # --- 1. Attempt to List and Delete Sessions ---
        logger.info(f"Attempting to list and delete sessions for Agent Engine: {agent_engine_id}...")
        try:
            # Call list_sessions which returns a ListSessionsResponse object
            sessions_response = await adk_session_service.list_sessions(
                app_name=agent_engine_id, 
                user_id="*"
            )
            
            # CRITICAL CORRECTION: Access the actual list of sessions from the 'sessions' attribute
            # of the ListSessionsResponse object.
            all_sessions_list = sessions_response.sessions 
            
            # Filter sessions belonging to this specific agent engine (redundant if app_name filter works perfectly, but safer)
            sessions_for_this_engine = [
                s for s in all_sessions_list if s.name.startswith(f"{agent_engine_id}/sessions/")
            ]

            if sessions_for_this_engine:
                logger.info(f"Found {len(sessions_for_this_engine)} active sessions for this engine. Deleting them now...")
                for session in sessions_for_this_engine:
                    try:
                        logger.info(f"Deleting session: {session.id}")
                        await adk_session_service.delete_session(session.id)
                        logger.info(f"Session {session.id} deleted successfully.")
                    except Exception as e:
                        logger.warning(f"Failed to delete session {session.id}: {e}")
                logger.info("Finished attempting to delete all listed sessions.")
            else:
                logger.info("No active sessions found for this Agent Engine.")
        except Exception as e:
            logger.error(f"Failed to list or delete sessions for {agent_engine_id}: {e}", exc_info=True)
            logger.warning("Proceeding to delete the Agent Engine with force=True, as session management failed anyway.")

        # --- 2. Delete the Agent Engine ---
        logger.info(f"Attempting to delete Agent Engine: {agent_engine_id}...")
        
        # Confirm with the user before deleting the engine
        # confirm = input(f"Are you sure you want to delete Agent Engine '{agent_engine_id}'? (yes/no): ").lower()
        confirm = 'yes'
        if confirm != 'yes':
            logger.info("Agent Engine deletion cancelled by user.")
            return

        # Use agent_engines.delete to destroy the engine
        # IMPORTANT: force=True is the most reliable way to delete an engine with sessions.
        agent_engines.delete(agent_engine_id, force=True)
        logger.info(f"Agent Engine '{agent_engine_id}' deleted successfully.")

        # Optional: Remove AGENT_ENGINE_ID from .env after successful deletion
        env_file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".env"))
        if os.path.exists(env_file_path):
            with open(env_file_path, 'r') as f:
                lines = f.readlines()
            with open(env_file_path, 'w') as f:
                for line in lines:
                    if not line.startswith("AGENT_ENGINE_ID="):
                        f.write(line)
            logger.info(f"Removed AGENT_ENGINE_ID from {env_file_path}.")

    except Exception as e:
        logger.error(f"An unexpected error occurred during Agent Engine destruction: {e}", exc_info=True)
        if "NotFound" in str(e) or "not found" in str(e).lower() or "already deleted" in str(e).lower():
            logger.info(f"Agent Engine '{agent_engine_id}' was not found or already deleted.")
        else:
            logger.error(f"Deletion failed with an unhandled error: {e}")

# --- Main execution block ---
if __name__ == "__main__":
    # Ensure GOOGLE_APPLICATION_CREDENTIALS is set in your environment
    if not os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
        logger.error("GOOGLE_APPLICATION_CREDENTIALS environment variable is not set.")
        logger.error("Please set it to the path of your service account key file (e.g., C:\\path\\to\\your\\key.json).")
        exit(1)

    # Use the AGENT_ENGINE_ID from .env or override if provided as command-line arg
    if len(sys.argv) > 1:
        engine_to_delete = sys.argv[1]
    else:
        engine_to_delete = AGENT_ENGINE_ID

    if not engine_to_delete:
        logger.error("No Agent Engine ID provided. Please set AGENT_ENGINE_ID in your .env file or pass it as a command-line argument.")
        exit(1)

    asyncio.run(destroy_agent_engine(engine_to_delete))