# app.py
import os
import uuid
import logging
import asyncio
import sys
import json
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
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langchain.memory import ConversationBufferMemory
from langchain_core.output_parsers import StrOutputParser

# Utility for secure filenames
from werkzeug.utils import secure_filename

# Import agent clients and document storage utilities
from agent_clients import get_rag_agent_client, get_jsonplaceholder_agent_client, get_prices_comparison_agent_client
import document_storage_utils


# --- Global Flask Application Setup ---
app = Flask(__name__)
CORS(app)

# --- Logging Configuration ---
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=LOG_LEVEL, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Global Services and LLM Components ---
adk_session_service = None
orchestrator_llm = None
orchestrator_decision_chain = None
general_knowledge_llm = None
general_knowledge_chain = None

# In-memory store for user sessions
# {session_id: {"memory": ConversationBufferMemory, "rag_adk_session_id": str, ...}}
agent_sessions: Dict[str, Dict[str, Any]] = {}

# --- Centralized Agent Configuration ---
# This dictionary centralizes all agent-specific configurations, making it easier to manage and extend.
AGENT_CONFIG = {
    'RAG': {
        'client_func': get_rag_agent_client,
        'client': None,
        'session_id_key': 'rag_adk_session_id',
        'engine_id_env': 'AGENT_ENGINE_ID', # RAG Agent Engine ID
    },
    'JSONPLACEHOLDER_API': {
        'client_func': get_jsonplaceholder_agent_client,
        'client': None,
        'session_id_key': 'openapi_adk_session_id',
        'engine_id_env': 'OPENAPI_AGENT_ENGINE_ID', # JSONPlaceholder Agent Engine ID
    },
    'PRICES_COMPARISON_API': {
        'client_func': get_prices_comparison_agent_client,
        'client': None,
        'session_id_key': 'prices_comparison_adk_session_id',
        'engine_id_env': 'PRICES_COMPARE_AGENT_ENGINE_ID', # Prices Comparison Agent Engine ID
    }
}


# --- Application Configuration from Environment Variables ---
FLASK_APP_NAME = os.environ.get("FLASK_APP_NAME", "AI_Assistant_RAG")
HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", 3010))
FLASK_DEBUG = os.environ.get("FLASK_DEBUG", "False").lower() in ('true', '1', 't')

PROJECT = os.environ.get("GOOGLE_CLOUD_PROJECT")
LOCATION = os.environ.get("GOOGLE_CLOUD_LOCATION")
STAGING_BUCKET = os.environ.get("STAGING_BUCKET")
RAG_CORPUS = os.environ.get("RAG_CORPUS")

LLM_MODEL_NAME = os.environ.get("LLM_MODEL_NAME", "gemini-1.5-flash-001")
LLM_TEMPERATURE = float(os.environ.get("LLM_TEMPERATURE", 0.2))

DOCUMENT_STORAGE_DIRECTORY = os.environ.get("DOCUMENT_STORAGE_DIRECTORY", "/app/data")

GUNICORN_TIMEOUT = int(os.environ.get("GUNICORN_TIMEOUT", 120))
GUNICORN_WORKERS = int(os.environ.get("GUNICORN_WORKERS", 2))


# --- Pre-flight Checks ---
def perform_preflight_checks():
    logger.info("Performing pre-flight checks...")

    if not os.getenv("GOOGLE_APPLICATION_CREDENTIALS") and not os.getenv("K_SERVICE"):
        logger.warning("GOOGLE_APPLICATION_CREDENTIALS is not set. Recommended for local development.")

    if not PROJECT or not LOCATION or not STAGING_BUCKET:
        logger.error("GOOGLE_CLOUD_PROJECT, GOOGLE_CLOUD_LOCATION, and STAGING_BUCKET must be set.")
        sys.exit(1)

    agent_ids = [os.getenv(conf['engine_id_env']) for conf in AGENT_CONFIG.values()]
    if not any(agent_ids):
        logger.error("At least one agent engine ID environment variable must be set.")
        sys.exit(1)

    if not RAG_CORPUS and os.getenv('AGENT_ENGINE_ID'):
        logger.warning("RAG_CORPUS is not set, but a RAG AGENT_ENGINE_ID is. Document uploads might fail.")

    logger.info("Essential configuration variables checked successfully.")


# --- Initialization Function ---
def initialize_all_components():
    """Initializes Vertex AI, ADK services, agent clients, and LangChain components."""
    global adk_session_service, orchestrator_llm, orchestrator_decision_chain, general_knowledge_llm, general_knowledge_chain

    try:
        vertexai.init(project=PROJECT, location=LOCATION, staging_bucket=f"gs://{STAGING_BUCKET}")
        logger.info("Vertex AI SDK initialized.")

        adk_session_service = VertexAiSessionService(project=PROJECT, location=LOCATION)
        logger.info("Vertex AI ADK Session Service initialized.")

        # Initialize all agent clients defined in AGENT_CONFIG
        for agent_name, config in AGENT_CONFIG.items():
            config['client'] = config['client_func']()
            if not config['client']:
                logger.warning(f"{agent_name} Agent Client could not be initialized. Its functionality will be unavailable.")
            else:
                logger.info(f"{agent_name} Agent Client initialized.")

        orchestrator_llm = ChatVertexAI(model_name=LLM_MODEL_NAME, temperature=0.0)
        general_knowledge_llm = ChatVertexAI(model_name=LLM_MODEL_NAME, temperature=0.7)
        logger.info("Orchestrator and General Knowledge LLMs initialized.")

        # --- Orchestrator Decision Chain (Router) ---
        orchestrator_decision_prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(
                """You are a specialized routing assistant. Your ONLY task is to determine the correct system to handle the user's query based on the rules below.
                You MUST NOT answer the question directly. Your output MUST be SOLELY one of the following exact keywords:
                'RAG', 'JSONPLACEHOLDER_API', 'PRICES_COMPARISON_API', 'GENERAL_KNOWLEDGE'

                Routing Rules:
                1.  'RAG': For queries about documents, policies, or facts in a knowledge base (e.g., "What are the pool hours?", "Tell me about community events").
                2.  'JSONPLACEHOLDER_API': For queries interacting with JSONPlaceholder resources (users, posts, comments), including CRUD and analysis (e.g., "List all users", "Create a new post", "Which user has the most posts?").
                3.  'PRICES_COMPARISON_API': For queries about comparing product prices or finding listings (e.g., "compare prices for iPhone", "find listings for headphones").
                4.  'GENERAL_KNOWLEDGE': For all other queries, including greetings, chit-chat, math, calculations, and general questions (e.g., "Hello", "What is 2+2?", "What is the capital of France?").

                Output only the keyword.
                """
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template("{query}")
        ])
        orchestrator_decision_chain = orchestrator_decision_prompt | orchestrator_llm | StrOutputParser()
        logger.info("Orchestrator decision chain initialized.")

        # --- General Knowledge Chain ---
        general_knowledge_prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(
                """You are a helpful AI assistant. Answer the user's question directly based on the conversation history and your general knowledge.
                If the query is a math problem, provide the answer. If you don't know the answer, politely say so.
                """
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template("{query}")
        ])
        general_knowledge_chain = general_knowledge_prompt | general_knowledge_llm | StrOutputParser()
        logger.info("General knowledge chain initialized.")

    except Exception as e:
        logger.critical(f"Failed to initialize core components: {e}. Exiting.", exc_info=True)
        sys.exit(1)


# --- Refactored Agent Handling Logic ---
async def handle_agent_query(
    decision: str,
    session_id: str,
    user_id: str,
    user_message: str
) -> Union[str, None]:
    """
    Handles the logic for querying a specific agent, including session creation and response streaming.
    Returns the agent's response as a string, or None if the agent fails.
    """
    config = AGENT_CONFIG.get(decision)
    if not config or not config['client']:
        logger.warning(f"Attempted to route to an unconfigured or uninitialized agent: {decision}")
        return None

    logger.info(f"Routing query to {decision} Agent: '{user_message}'")
    session_key = config['session_id_key']
    adk_session_id = agent_sessions[session_id].get(session_key)

    # Create a new ADK session if one doesn't exist for this agent
    if not adk_session_id and adk_session_service:
        try:
            engine_id = os.getenv(config['engine_id_env'])
            if not engine_id:
                raise ValueError(f"Environment variable {config['engine_id_env']} not set for agent {decision}")

            logger.info(f"Creating new ADK session for {decision} agent (Flask session {session_id}, user {user_id})")
            adk_session = await adk_session_service.create_session(app_name=engine_id, user_id=user_id)
            adk_session_id = adk_session.id
            agent_sessions[session_id][session_key] = adk_session_id
            logger.info(f"Created ADK session {adk_session_id} for {decision} agent.")
        except Exception as e:
            logger.error(f"Failed to create ADK session for {decision} agent: {e}", exc_info=True)
            return f"I couldn't start a session for the {decision} service."

    # Query the agent
    try:
        response_stream = config['client'].stream_query(
            user_id=user_id,
            session_id=adk_session_id,
            message=user_message,
        )

        actual_content = ""
        for event in response_stream:
            if isinstance(event, dict):
                # Extract text content
                if "content" in event and "parts" in event["content"]:
                    for part in event["content"]["parts"]:
                        if "text" in part:
                            actual_content += part["text"]
                # Extract and format tool outputs for JSON/API agents
                if decision != 'RAG' and "tool_outputs" in event and isinstance(event["tool_outputs"], list):
                     for output in event["tool_outputs"]:
                        if "data" in output and output["data"] is not None:
                            try:
                                formatted_output = json.dumps(output["data"], indent=2)
                                actual_content += f"\n```json\n{formatted_output}\n```\n"
                            except (json.JSONDecodeError, TypeError):
                                actual_content += f"\nTool Output: {str(output['data'])}\n"
            elif isinstance(event, str):
                actual_content += event

        return actual_content.strip()

    except Exception as e:
        logger.error(f"Error querying {decision} Agent for '{user_message}' (ADK Session: {adk_session_id}): {e}", exc_info=True)
        return f"I encountered an error while interacting with the {decision} service."


# --- Flask Routes ---

@app.route("/")
def index():
    """Renders the main chat UI."""
    return render_template("index.html")


@app.route("/api/bot", methods=["POST"])
async def bot_chat():
    """Handles chat messages, routing them to the appropriate agent or general knowledge fallback."""
    try:
        request_data = request.get_json()
        if not request_data:
            return jsonify({"status": "error", "message": "Invalid JSON payload."}), 400

        user_message = request_data.get("query")
        session_id = request_data.get("session_id")
        user_id = request_data.get('user_id', 'default_user')

        if not user_message:
            return jsonify({"status": "error", "message": "No query provided."}), 400

        if not session_id:
            session_id = str(uuid.uuid4())
            logger.info(f"Generated new session ID: {session_id}")

        if session_id not in agent_sessions:
            agent_sessions[session_id] = {
                "memory": ConversationBufferMemory(memory_key="chat_history", return_messages=True)
            }
            logger.info(f"Initialized new state for session: {session_id}")

        current_memory = agent_sessions[session_id]["memory"]
        chat_history = current_memory.load_memory_variables({})["chat_history"]

        # 1. Orchestration: Decide which system to use
        routing_input = {"query": user_message, "chat_history": chat_history}
        decision_raw = orchestrator_decision_chain.invoke(routing_input)
        decision = decision_raw.strip().upper()
        logger.info(f"Orchestrator decision for '{user_message}': '{decision}'")

        if decision not in AGENT_CONFIG and decision != 'GENERAL_KNOWLEDGE':
            logger.warning(f"Orchestrator returned invalid decision '{decision}'. Defaulting to GENERAL_KNOWLEDGE.")
            decision = 'GENERAL_KNOWLEDGE'

        agent_response = ""

        # 2. Route to the appropriate handler
        if decision in AGENT_CONFIG:
            agent_response = await handle_agent_query(decision, session_id, user_id, user_message)
            # Fallback condition for RAG agent if it doesn't find an answer
            no_answer_indicators = [
                "not available in the provided documents",
                "couldn't find any documents",
                "cannot answer this question based on the provided documents",
            ]
            if decision == 'RAG' and (not agent_response or any(indicator in agent_response.lower() for indicator in no_answer_indicators)):
                logger.info(f"RAG Agent did not provide a substantive answer. Falling back to general knowledge.")
                decision = 'GENERAL_KNOWLEDGE' # Trigger fallback
            else:
                 logger.info(f"{decision} Agent successfully answered.")

        # 3. Handle General Knowledge (or fallback)
        if decision == 'GENERAL_KNOWLEDGE':
            logger.info(f"Routing query to General Knowledge LLM: '{user_message}'")
            try:
                gk_response = general_knowledge_chain.invoke(routing_input)
                agent_response = gk_response or "I couldn't find an answer to your question."
            except Exception as e:
                logger.error(f"Error querying General Knowledge LLM: {e}", exc_info=True)
                agent_response = "I encountered an error using my general knowledge."

        # Final response check
        if not agent_response.strip():
            agent_response = "I apologize, but I couldn't provide a specific answer. Please try rephrasing your question."
            logger.warning(f"All paths failed for '{user_message}'. Returning generic fallback.")

        # Save context and return response
        current_memory.save_context({"input": user_message}, {"output": agent_response})
        return jsonify({"status": "success", "response": agent_response, "session_id": session_id}), 200

    except Exception as e:
        logger.exception(f"An unhandled error occurred in bot_chat endpoint:")
        return jsonify({"status": "error", "message": f"An internal server error occurred: {e}"}), 500


@app.route("/api/documents/upload", methods=["POST"])
def upload_document_endpoint():
    """Endpoint for uploading documents to the Vertex AI RAG Corpus."""
    if 'file' not in request.files or not request.files['file'].filename:
        return jsonify({"status": "error", "message": "No file selected."}), 400

    uploaded_file = request.files['file']
    filename = secure_filename(uploaded_file.filename)
    temp_file_path = os.path.join(DOCUMENT_STORAGE_DIRECTORY, filename)

    try:
        os.makedirs(DOCUMENT_STORAGE_DIRECTORY, exist_ok=True)
        uploaded_file.save(temp_file_path)
        logger.info(f"File '{filename}' temporarily saved to '{temp_file_path}'.")

        upload_result = document_storage_utils.upload_document_to_rag_corpus(
            document_path=temp_file_path,
            document_display_name=filename,
        )

        if upload_result.get("status") == "success":
            logger.info(f"Document '{filename}' successfully processed for RAG Corpus.")
            return jsonify(upload_result), 200
        else:
            logger.error(f"Failed to process document '{filename}': {upload_result.get('output')}")
            return jsonify(upload_result), 500

    except Exception as e:
        logger.exception(f"Error during document upload for '{filename}':")
        return jsonify({"status": "error", "message": str(e)}), 500
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
            logger.info(f"Cleaned up temporary file: '{temp_file_path}'.")


@app.route('/health', methods=['GET'])
def health_check():
    """Provides a simple health check endpoint."""
    return jsonify({"status": "healthy", "message": f"{FLASK_APP_NAME} is running"}), 200

if __name__ == "__main__":
    perform_preflight_checks()
    initialize_all_components()

    logger.info(f"Starting {FLASK_APP_NAME} on {HOST}:{PORT}")
    try:
        # Use Gunicorn for production environments
        from gunicorn.app.base import BaseApplication

        class StandaloneApplication(BaseApplication):
            def __init__(self, app, options=None):
                self.options = options or {}
                self.application = app
                super().__init__()

            def load_config(self):
                for key, value in self.options.items():
                    if key in self.cfg.settings and value is not None:
                        self.cfg.set(key.lower(), value)

            def load(self):
                return self.application

        gunicorn_options = {
            'bind': f"{HOST}:{PORT}",
            'workers': GUNICORN_WORKERS,
            'timeout': GUNICORN_TIMEOUT,
            'loglevel': LOG_LEVEL.lower(),
            'reload': FLASK_DEBUG,
        }
        StandaloneApplication(app, gunicorn_options).run()

    except ImportError:
        # Fallback to Flask's built-in server for development
        logger.warning("Gunicorn not found. Falling back to Flask's development server. Do not use in production.")
        app.run(host=HOST, port=PORT, debug=FLASK_DEBUG)
    except Exception as e:
        logger.critical(f"Failed to start application: {e}", exc_info=True)
        sys.exit(1)