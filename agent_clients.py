import os
import vertexai
from vertexai import agent_engines
import logging

# Configure logging for this module
logger = logging.getLogger(__name__)
logger.setLevel(os.environ.get("LOG_LEVEL", "INFO").upper())


def _get_agent_engine_client(agent_engine_id_env_var: str):
    """Retrieves an AgentEngine client from Vertex AI based on an environment variable."""
    agent_engine_id = os.environ.get(agent_engine_id_env_var)
    if not agent_engine_id:
        logger.error(f"Missing required environment variable: {agent_engine_id_env_var}")
        return None # Return None if ID is missing, caller should handle

    try:
        client = agent_engines.get(agent_engine_id)
        logger.info(f"Successfully retrieved Agent Engine client for {agent_engine_id_env_var}: {agent_engine_id}")
        return client
    except Exception as e:
        logger.error(f"Failed to retrieve Agent Engine client for {agent_engine_id_env_var} ({agent_engine_id}): {e}")
        return None

def get_rag_agent_client():
    """Returns the client for the RAG Agent Engine."""
    return _get_agent_engine_client("AGENT_ENGINE_ID")

def get_openapi_agent_client():
    """Returns the client for the OpenAPI Agent Engine."""
    return _get_agent_engine_client("OPENAPI_AGENT_ENGINE_ID")