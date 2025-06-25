# agent_clients.py
import os
import vertexai
from vertexai import agent_engines

# Assume the ADK agent.py is in the same directory or import path allows access
from agent import root_agent as jsonplaceholder_root_agent # Rename to avoid conflict
# We will effectively replace this with a more generic root_agent that includes all
# your API agents. For now, let's assume `agent.py` contains the combined `root_agent`.

# Placeholder for RAG client (as per your app.py)
def get_rag_agent_client():
    # Placeholder for your RAG agent client setup
    # This might involve creating an agent_engines.Agent and returning it
    rag_agent_client_name = os.getenv("RAG_AGENT_CLIENT_NAME", "your-rag-agent-name")
    rag_project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
    rag_location = os.getenv("GOOGLE_CLOUD_LOCATION")
    # Example:
    # return agent_engines.Agent.lookup(
    #     project=rag_project_id,
    #     location=rag_location,
    #     display_name=rag_agent_client_name
    # )
    return None # Replace with actual RAG client instantiation


# Modified to get a generic API agent client that includes all OpenAPI-based agents
def get_api_agent_client():
    """Returns the API agent client (e.g., JSONPlaceholder + Prices Comparison)."""
    api_agent_client_name = os.getenv("API_AGENT_CLIENT_NAME", "your-api-agent-name")
    api_project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
    api_location = os.getenv("GOOGLE_CLOUD_LOCATION")

    # In a real deployment, you would lookup an already deployed agent.
    # For local testing, we are directly creating the agent instance.
    # Ensure root_agent in agent.py includes all sub-agents and their tools.
    # If the root_agent in agent.py is the one orchestrating all APIs,
    # then you can use that directly.
    return jsonplaceholder_root_agent # Assuming the 'root_agent' from agent.py is now the comprehensive one
    # If you deploy it to Vertex AI Agent Builder, you'd do:
    # return agent_engines.Agent.lookup(
    #     project=api_project_id,
    #     location=api_location,
    #     display_name=api_agent_client_name
    # )