import os
import uuid
import logging
import asyncio # For ADK session creation
import sys
from typing import Dict, Any, Union, List

# Flask imports
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

# Dotenv for loading environment variables
from dotenv import load_dotenv

# --- Initial Environment Variable Load (must be at the very top for global access) ---
load_dotenv()

# Vertex AI and ADK imports
import vertexai
from vertexai import agent_engines
from google.adk.sessions import VertexAiSessionService

# LangChain Agent and Tooling imports
from langchain_google_vertexai import ChatVertexAI
# Explicitly import individual prompt message templates for robustness
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    PromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessageTemplate # Corrected from SystemMessagePromptTemplate to match common usage
)
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage, BaseMessage
from langchain.memory import ConversationBufferMemory # Still used for general knowledge and orchestrator history
from langchain_core.output_parsers import StrOutputParser # Explicitly import StrOutputParser

# For PDF document loading (used in upload endpoint) - NOTE: This is no longer directly used in upload_document,
# as document_storage_utils handles it via rag.upload_file, but keeping for completeness if other local loaders are used.
from langchain_community.document_loaders import PyPDFLoader

# Utility for secure filenames
from werkzeug.utils import secure_filename

# NEW: Import agent clients from your new file
from agent_clients import get_rag_agent_client, get_openapi_agent_client

# NEW: Import document storage utilities
import document_storage_utils


# --- Global Flask Application Setup ---
app = Flask(__name__)
CORS(app) # Enable CORS for all routes

# --- Logging Configuration ---
# Get logging level from environment variable, default to INFO
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=LOG_LEVEL, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Global Agent/LLM Clients ---
# These will be initialized once when the application starts
rag_agent_client = None
openapi_agent_client = None
router_llm = None
router_prompt = None
adk_session_service = None # Needed for managing ADK sessions for RAG agent

# In-memory store for AgentExecutors and their memories
# {session_id: {"memory": ConversationBufferMemory_instance, "adk_session_id": Optional[str]}}
agent_sessions: Dict[str, Dict[str, Any]] = {}


# --- Application Configuration from Environment Variables ---
FLASK_APP_NAME = os.environ.get("FLASK_APP_NAME", "AI_Assistant_RAG")
HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", 3010))
FLASK_DEBUG = os.environ.get("FLASK_DEBUG", "False").lower() in ('true', '1', 't')

# Vertex AI specific configurations
GOOGLE_GENAI_USE_VERTEXAI = os.environ.get("GOOGLE_GENAI_USE_VERTEXAI", "0") == "1"
PROJECT = os.environ.get("GOOGLE_CLOUD_PROJECT")
LOCATION = os.environ.get("GOOGLE_CLOUD_LOCATION")
STAGING_BUCKET = os.environ.get("STAGING_BUCKET")
RAG_CORPUS = os.environ.get("RAG_CORPUS") # Used by document_storage_utils

# LLM Configuration for Router (and general fallback if needed)
LLM_MODEL_NAME = os.environ.get("LLM_MODEL_NAME", "gemini-2.0-flash-001")
LLM_TEMPERATURE = float(os.environ.get("LLM_TEMPERATURE", 0.2))

# Document Storage Directory (for temporary file uploads before RAG ingestion)
DOCUMENT_STORAGE_DIRECTORY = os.environ.get("DOCUMENT_STORAGE_DIRECTORY", "/app/data")

# Gunicorn Configuration
GUNICORN_TIMEOUT = int(os.environ.get("GUNICORN_TIMEOUT", 120))
GUNICORN_WORKERS = int(os.environ.get("GUNICORN_WORKERS", 2))


# --- Pre-flight Checks (Crucial for proper operation) ---
def perform_preflight_checks():
    logger.info("Performing pre-flight checks...")

    if not os.getenv("GOOGLE_APPLICATION_CREDENTIALS") and not os.getenv("K_SERVICE"): # K_SERVICE indicates Cloud Run
        logger.warning("Environment variable GOOGLE_APPLICATION_CREDENTIALS is not set. This is recommended for local development to authenticate with Google Cloud services like Vertex AI. For Cloud Run, service account attached to the service will be used.")

    if not PROJECT:
        logger.error("GOOGLE_CLOUD_PROJECT is not set. Please set this environment variable.")
        sys.exit(1)

    if not LOCATION:
        logger.error("GOOGLE_CLOUD_LOCATION is not set. Please set this environment variable.")
        sys.exit(1)

    if not os.environ.get("AGENT_ENGINE_ID") and not os.environ.get("OPENAPI_AGENT_ENGINE_ID"):
        logger.error("Neither AGENT_ENGINE_ID nor OPENAPI_AGENT_ENGINE_ID is set. At least one agent engine ID is required.")
        sys.exit(1)

    if not RAG_CORPUS and os.environ.get("AGENT_ENGINE_ID"):
        # Only warn if RAG_CORPUS is missing AND RAG_AGENT_ENGINE_ID is set
        logger.warning("RAG_CORPUS is not set, but AGENT_ENGINE_ID is. Document upload and RAG queries might fail without a corpus.")

    if not STAGING_BUCKET:
        logger.error("STAGING_BUCKET is not set. This is required for staging files for RAG corpus upload. Please set this environment variable.")
        sys.exit(1)

    logger.info("Essential configuration variables checked.")
# --- End Pre-flight Checks ---

perform_preflight_checks() # Run checks early in the application lifecycle

# --- Agent Initialization Function ---
def initialize_agents():
    """
    Initializes Vertex AI, loads the RAG and OpenAPI Agent Engine clients,
    and sets up the routing LLM and prompt.
    """
    global rag_agent_client, openapi_agent_client, router_llm, router_prompt, adk_session_service

    try:
        # Initialize Vertex AI SDK for this Flask application process
        # This is critical before trying to get agent_engines clients or other Vertex AI services.
        vertexai.init(
            project=PROJECT,
            location=LOCATION,
            staging_bucket=f"gs://{STAGING_BUCKET}", # Staging bucket for general Vertex AI ops, including Agent Engine
        )
        logger.info("Vertex AI SDK initialized for application.")
    except Exception as e:
        logger.critical(f"Failed to initialize Vertex AI SDK: {e}. Exiting.", exc_info=True)
        sys.exit(1) # Critical failure, exit application

    # Initialize Vertex AI ADK Session Service for RAG agent memory management
    try:
        adk_session_service = VertexAiSessionService(
            project=PROJECT,
            location=LOCATION
        )
        logger.info("Vertex AI ADK Session Service initialized.")
    except Exception as e:
        logger.error(f"Failed to initialize Vertex AI ADK Session Service: {e}. RAG agent memory might not persist correctly.", exc_info=True)
        adk_session_service = None # Set to None to indicate failure

    # Initialize RAG Agent Client
    rag_agent_client = get_rag_agent_client()
    if not rag_agent_client:
        logger.warning("RAG Agent Client could not be initialized. RAG functionality will be unavailable.")

    # Initialize OpenAPI Agent Client
    openapi_agent_client = get_openapi_agent_client()
    if not openapi_agent_client:
        logger.warning("OpenAPI Agent Client could not be initialized. OpenAPI functionality will be unavailable.")

    # Initialize Router LLM and Prompt for intelligent routing
    try:
        router_llm = ChatVertexAI(
            model_name=LLM_MODEL_NAME,
            temperature=LLM_TEMPERATURE,
            convert_system_message_to_human=True, # Often needed for Vertex AI models with SystemMessage
        )

        router_prompt = ChatPromptTemplate.from_messages([
            SystemMessageTemplate.from_template(
                """You are an intelligent router that determines which specialized agent should handle a user's query.
                Your output must be one of the following keywords only. Do not add any other text or punctuation, no explanations.

                - 'RAG': Use this if the user's query is about existing documents, policies, stored information, or asking for facts that would be found in a knowledge base (e.g., "What are the rules for residents?", "Tell me about the community events policy").
                - 'OPENAPI': Use this if the user's query is explicitly about JSONPlaceholder API resources: users, posts, or comments, including listing, creating, getting details, updating, or deleting these entities (e.g., "List all users", "Create a new post", "Get comments for post ID 5", "Delete user 10").
                - 'UNKNOWN': Use this for general greetings, chit-chat, or if the user's intent does not clearly fall into RAG or OpenAPI categories.
                """
            ),
            HumanMessagePromptTemplate.from_template("{query}")
        ])
        logger.info("Router LLM and prompt initialized.")
    except Exception as e:
        logger.critical(f"Failed to initialize Router LLM or prompt: {e}. Routing will be basic.", exc_info=True)
        router_llm = None
        router_prompt = None


# --- Flask Routes ---

@app.route("/")
def index():
    """Renders the main index page (your chat UI)."""
    return render_template("index.html")

@app.route("/api/bot", methods=["POST"])
def bot_chat():
    """Handles chat messages, routing them to the appropriate agent."""
    user_message = request.json.get("message")
    if not user_message:
        return jsonify({"response": "No message provided."}), 400

    session_id = request.headers.get('X-Session-ID') or request.json.get('session_id')
    user_id = request.json.get('user_id', 'default_user') # Default user_id if not provided

    if not session_id:
        session_id = str(uuid.uuid4())
        logger.info(f"Generated new session ID: {session_id} for query: '{user_message}'.")
    else:
        logger.info(f"Using existing session ID: {session_id} for query: '{user_message}'.")

    # Initialize session memory if not already present
    if session_id not in agent_sessions:
        agent_sessions[session_id] = {
            "memory": ConversationBufferMemory(memory_key="chat_history", return_messages=True),
            "adk_session_id": None # Initialize ADK session ID as None for RAG
        }
        logger.info(f"Initialized new memory for session: {session_id}")

    current_memory = agent_sessions[session_id]["memory"]

    try:
        # Determine which agent to use
        decision = 'UNKNOWN' # Default decision
        if router_llm and router_prompt:
            try:
                routing_chain = router_prompt | router_llm | StrOutputParser()
                decision_raw = routing_chain.invoke({"query": user_message})
                decision = decision_raw.strip().upper()
                logger.info(f"Router decision for '{user_message}': '{decision_raw.strip()}' -> '{decision}'")
            except Exception as e:
                logger.error(f"Error during routing decision for '{user_message}': {e}", exc_info=True)
                decision = 'UNKNOWN' # Fallback if router fails

        agent_response = ""
        if decision == 'RAG' and rag_agent_client:
            logger.info(f"Routing query to RAG Agent: '{user_message}'")
            adk_session_id = agent_sessions[session_id].get("adk_session_id")

            # Create ADK session if it doesn't exist for RAG agent
            if not adk_session_id and adk_session_service:
                try:
                    logger.info(f"Creating new ADK session for Flask session {session_id}, user {user_id}")
                    adk_session = asyncio.run(adk_session_service.create_session(
                        app_name=os.environ.get("AGENT_ENGINE_ID"), # AGENT_ENGINE_ID is the resource name
                        user_id=user_id
                    ))
                    adk_session_id = adk_session.id
                    agent_sessions[session_id]["adk_session_id"] = adk_session_id
                    logger.info(f"Created ADK session {adk_session_id} for Flask session {session_id}")
                except Exception as e:
                    logger.error(f"Failed to create ADK session for {user_id}/{session_id}: {e}", exc_info=True)
                    agent_response = "Failed to initialize RAG session. Please try again."
                    decision = 'ERROR' # Mark as error to prevent further processing

            if decision != 'ERROR': # Only proceed if session creation was successful
                try:
                    rag_actual_content = ""
                    # Stream query to RAG agent for potentially long responses
                    response_stream = rag_agent_client.stream_query(
                        user_id=user_id,
                        session_id=adk_session_id,
                        message=user_message,
                    )
                    for event in response_stream:
                        if "content" in event and "parts" in event["content"]:
                            for part in event["content"]["parts"]:
                                if "text" in part:
                                    rag_actual_content += part["text"]
                        # Log other event types for debugging if needed
                        if "tool_code" in event: logger.debug(f"RAG tool_code: {event.get('tool_code')}")
                        if "state_delta" in event: logger.debug(f"RAG state_delta: {event.get('state_delta')}")
                        if "actions" in event: logger.debug(f"RAG actions: {event.get('actions')}")

                    agent_response = rag_actual_content.strip()
                    if not agent_response:
                        agent_response = "I couldn't find a direct answer in the documents. Could you rephrase your question?"
                        logger.warning(f"RAG Agent returned empty response for '{user_message}'.")
                except Exception as e:
                    logger.error(f"Error querying RAG Agent for '{user_message}': {e}", exc_info=True)
                    agent_response = "I encountered an error while retrieving information from the documents."

        elif decision == 'OPENAPI' and openapi_agent_client:
            logger.info(f"Routing query to OpenAPI Agent: '{user_message}'")
            try:
                # OpenAPI Agent does not necessarily require ADK session service for simple interactions
                # if its underlying agents are stateless or manage their own state.
                agent_response = openapi_agent_client.predict(user_message)
                if not agent_response:
                    agent_response = "The OpenAPI agent did not provide a specific response. Please try again."
                    logger.warning(f"OpenAPI Agent returned empty response for '{user_message}'.")
            except Exception as e:
                logger.error(f"Error querying OpenAPI Agent for '{user_message}': {e}", exc_info=True)
                agent_response = "I encountered an error while interacting with the OpenAPI service."
        else: # Covers 'UNKNOWN' decision or if clients are not initialized
            logger.warning(f"Query '{user_message}' not routed to RAG or OpenAPI. Decision: '{decision}'. Falling back.")
            agent_response = "I'm currently equipped to answer questions about uploaded documents (RAG) or manage data via the JSONPlaceholder API (users, posts, comments). Please ask me about one of those topics, or provide more context."

        # Add the complete turn (user input + model response) to session history.
        current_memory.save_context({"input": user_message}, {"output": agent_response})

        response_data = {"status": "success", "response": agent_response, "session_id": session_id}
        flask_response = jsonify(response_data)
        flask_response.headers['X-Session-ID'] = session_id
        return flask_response, 200

    except Exception as e:
        logger.exception(f"An unhandled error occurred in bot_chat for query '{user_message}' (Session: {session_id}):")
        error_message = str(e)
        if session_id in agent_sessions:
            current_memory.save_context({"input": user_message}, {"output": f"Error: {error_message}"})
        return jsonify({"status": "error", "message": error_message}), 500

@app.route("/api/documents/upload", methods=["POST"])
def upload_document_endpoint():
    """
    Dedicated endpoint for uploading documents via HTTP POST to the Vertex AI RAG Corpus.
    Delegates the actual upload logic to document_storage_utils.
    """
    if 'file' not in request.files:
        return jsonify({"status": "error", "message": "Missing 'file' in request."}), 400

    uploaded_file = request.files['file']
    if uploaded_file.filename == '':
        return jsonify({"status": "error", "message": "No selected file."}), 400

    # Ensure document storage directory exists
    os.makedirs(DOCUMENT_STORAGE_DIRECTORY, exist_ok=True)

    filename = secure_filename(uploaded_file.filename)
    temp_file_path = os.path.join(DOCUMENT_STORAGE_DIRECTORY, filename)

    try:
        uploaded_file.save(temp_file_path)
        logger.info(f"File '{filename}' temporarily saved to '{temp_file_path}'.")

        # Delegate to document_storage_utils for actual RAG corpus upload
        upload_result = document_storage_utils.upload_document_to_rag_corpus(
            document_path=temp_file_path,
            document_display_name=filename, # Use filename as display name
            document_description=f"Uploaded document: {filename}"
        )

        final_response_message = upload_result.get("output", "Unknown upload result").strip()

        if upload_result.get("status") == "success":
            logger.info(f"Document '{filename}' successfully processed for RAG Corpus. Result: {final_response_message}")
            return jsonify({
                "status": "success",
                "message": f"File '{filename}' processed successfully. {final_response_message}",
                "doc_id": upload_result.get("rag_file_name") # The RAG file resource name
            }), 200
        else:
            logger.error(f"Failed to process document '{filename}'. Result: {final_response_message}")
            return jsonify({"status": "error", "message": f"Failed to process document: {final_response_message}"}), 500

    except Exception as e:
        logger.exception(f"An error occurred during document upload for file '{filename}':")
        return jsonify({"status": "error", "message": str(e)}), 500
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
            logger.info(f"Cleaned up temporary file: '{temp_file_path}'.")

@app.route('/health', methods=['GET'])
def health_check():
    """Provides a simple health check endpoint for the Flask application."""
    logger.debug("Health check requested.")
    return jsonify({"status": "healthy", "message": f"{FLASK_APP_NAME} is running"}), 200

# --- Main Application Entry Point (for Gunicorn or Flask dev server) ---
if __name__ == "__main__":
    # Initialize agents when the application starts
    initialize_agents()

    # If running directly (e.g., via `python app.py`), use Flask's dev server.
    # In a production Docker environment, Gunicorn (or your chosen WSGI server)
    # will call this application.
    try:
        from gunicorn.app.wsgiapp import WSGIApplication
        from gunicorn.config import Config as GunicornConfig
        
        class StandaloneApplication(WSGIApplication):
            def __init__(self, app_name, options=None):
                self.options = options or {}
                self.application = app
                super().__init__(app_name)

            def load_config(self):
                config_items = {key: value for key, value in self.options.items()
                                if key in self.cfg.settings and value is not None}
                for key, value in config_items.items():
                    self.cfg.set(key.lower(), value)

            def load(self):
                return self.application

        logger.info("Starting Gunicorn application.")
        gunicorn_options = {
            'bind': f"{HOST}:{PORT}",
            'workers': GUNICORN_WORKERS,
            'loglevel': LOG_LEVEL.lower(),
            'timeout': GUNICORN_TIMEOUT,
            'reload': FLASK_DEBUG # Reload workers on code changes if debug is true
        }
        StandaloneApplication("app:app", gunicorn_options).run()

    except ImportError:
        logger.warning("Gunicorn not found. Falling back to Flask development server. This is not recommended for production.")
        app.run(host=HOST, port=PORT, debug=FLASK_DEBUG)
    except Exception as e:
        logger.critical(f"Failed to start application: {e}", exc_info=True)
        sys.exit(1)

