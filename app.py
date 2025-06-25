# app.py
import os
import uuid
import logging
import asyncio
import sys
import json
import pprint # Added for pretty printing debug logs
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
    AIMessagePromptTemplate
)
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage, BaseMessage
from langchain.memory import ConversationBufferMemory
from langchain_core.output_parsers import StrOutputParser

# Utility for secure filenames
from werkzeug.utils import secure_filename

# NEW: Import agent clients (assuming agent_clients.py exists and provides these)
# IMPORTANT: Updated imports to include prices comparison client
from agent_clients import get_rag_agent_client, get_jsonplaceholder_agent_client, get_prices_comparison_agent_client

# NEW: Import document storage utilities (assuming document_storage_utils.py exists)
import document_storage_utils


# --- Global Flask Application Setup ---
app = Flask(__name__)
CORS(app)

# --- Logging Configuration ---
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=LOG_LEVEL, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Global Agent/LLM Clients ---
# IMPORTANT: Added prices_comparison_agent_client
rag_agent_client = None
jsonplaceholder_agent_client = None
prices_comparison_agent_client = None
adk_session_service = None

# Orchestrator and General Knowledge LLM components
orchestrator_llm = None
orchestrator_decision_chain = None
general_knowledge_llm = None
general_knowledge_chain = None

# In-memory store for AgentExecutors and their memories
# {session_id: {"memory": ConversationBufferMemory_instance, "rag_adk_session_id": Optional[str], "openapi_adk_session_id": Optional[str]}}
agent_sessions: Dict[str, Dict[str, Any]] = {}


# --- Application Configuration from Environment Variables ---
FLASK_APP_NAME = os.environ.get("FLASK_APP_NAME", "AI_Assistant_RAG")
HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", 3010))
FLASK_DEBUG = os.environ.get("FLASK_DEBUG", "False").lower() in ('true', '1', 't')

PROJECT = os.environ.get("GOOGLE_CLOUD_PROJECT")
LOCATION = os.environ.get("GOOGLE_CLOUD_LOCATION")
STAGING_BUCKET = os.environ.get("STAGING_BUCKET")
RAG_CORPUS = os.environ.get("RAG_CORPUS")

LLM_MODEL_NAME = os.environ.get("LLM_MODEL_NAME", "gemini-2.0-flash-001")
LLM_TEMPERATURE = float(os.environ.get("LLM_TEMPERATURE", 0.2))

# Agent Engine IDs are full resource names from .env
RAG_AGENT_ENGINE_ID = os.getenv("AGENT_ENGINE_ID") # Assuming this is for RAG
OPENAPI_AGENT_ENGINE_ID = os.getenv("OPENAPI_AGENT_ENGINE_ID") # For JSONPlaceholder
PRICES_COMPARE_AGENT_ENGINE_ID = os.getenv("PRICES_COMPARE_AGENT_ENGINE_ID") # NEW: For Prices Comparison


DOCUMENT_STORAGE_DIRECTORY = os.environ.get("DOCUMENT_STORAGE_DIRECTORY", "/app/data")

GUNICORN_TIMEOUT = int(os.environ.get("GUNICORN_TIMEOUT", 120))
GUNICORN_WORKERS = int(os.environ.get("GUNICORN_WORKERS", 2))


# --- Pre-flight Checks (Crucial for proper operation) ---
def perform_preflight_checks():
    logger.info("Performing pre-flight checks...")

    if not os.getenv("GOOGLE_APPLICATION_CREDENTIALS") and not os.getenv("K_SERVICE"):
        logger.warning("Environment variable GOOGLE_APPLICATION_CREDENTIALS is not set. This is recommended for local development to authenticate with Google Cloud services like Vertex AI. For Cloud Run, service account attached to the service will be used.")

    if not PROJECT:
        logger.error("GOOGLE_CLOUD_PROJECT is not set. Please set this environment variable.")
        sys.exit(1)

    if not LOCATION:
        logger.error("GOOGLE_CLOUD_LOCATION is not set. Please set this environment variable.")
        sys.exit(1)

    # IMPORTANT: Updated check to ensure at least one agent ID is present
    if not RAG_AGENT_ENGINE_ID and not OPENAPI_AGENT_ENGINE_ID and not PRICES_COMPARE_AGENT_ENGINE_ID:
        logger.error("Neither AGENT_ENGINE_ID, OPENAPI_AGENT_ENGINE_ID, nor PRICES_COMPARE_AGENT_ENGINE_ID is set. At least one agent engine ID is required for agent functionality.")
        sys.exit(1)

    if not RAG_CORPUS and RAG_AGENT_ENGINE_ID:
        logger.warning("RAG_CORPUS is not set, but AGENT_ENGINE_ID (for RAG) is. Document upload and RAG queries might fail without a corpus.")

    if not STAGING_BUCKET:
        logger.error("STAGING_BUCKET is not set. This is required for staging files for RAG corpus upload and other Vertex AI operations. Please set this environment variable.")
        sys.exit(1)

    logger.info("Essential configuration variables checked successfully.")


# --- Agent and Orchestrator Initialization Function ---
def initialize_all_components():
    """
    Initializes Vertex AI SDK, ADK Session Service, RAG/OpenAPI Agent Engine clients,
    and the LangChain-based orchestrator and general knowledge chains.
    """
    # IMPORTANT: Updated global statement to include new client
    global rag_agent_client, jsonplaceholder_agent_client, prices_comparison_agent_client, adk_session_service
    global orchestrator_llm, orchestrator_decision_chain, general_knowledge_llm, general_knowledge_chain

    try:
        # Initialize Vertex AI SDK
        vertexai.init(
            project=PROJECT,
            location=LOCATION,
            staging_bucket=f"gs://{STAGING_BUCKET}",
        )
        logger.info("Vertex AI SDK initialized for application.")

        # Initialize ADK Session Service (needed for creating/managing ADK sessions for deployed agents)
        adk_session_service = VertexAiSessionService(
            project=PROJECT,
            location=LOCATION
        )
        logger.info("Vertex AI ADK Session Service initialized.")

        # Initialize RAG Agent Client (if ID provided)
        rag_agent_client = get_rag_agent_client()
        if not rag_agent_client:
            logger.warning("RAG Agent Client could not be initialized. RAG functionality will be unavailable.")

        # Initialize JSONPlaceholder OpenAPI Agent Client (if ID provided)
        jsonplaceholder_agent_client = get_jsonplaceholder_agent_client()
        if not jsonplaceholder_agent_client:
            logger.warning("JSONPlaceholder OpenAPI Agent Client could not be initialized. JSONPlaceholder functionality will be unavailable.")

        # NEW: Initialize Prices Comparison Agent Client (if ID provided)
        prices_comparison_agent_client = get_prices_comparison_agent_client()
        if not prices_comparison_agent_client:
            logger.warning("Prices Comparison Agent Client could not be initialized. Prices Comparison functionality will be unavailable.")


        # Initialize LLM for Orchestrator and General Knowledge
        orchestrator_llm = ChatVertexAI(
            model_name=LLM_MODEL_NAME,
            temperature=0.0,
            convert_system_message_to_human=True,
        )
        general_knowledge_llm = ChatVertexAI(
            model_name=LLM_MODEL_NAME,
            temperature=0.7,
            convert_system_message_to_human=True,
        )
        logger.info("Orchestrator and General Knowledge LLMs initialized.")

        # --- Orchestrator Decision Chain (Context-Aware Router) ---
        orchestrator_decision_prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(
                """You are a specialized routing assistant. Your ONLY task is to determine the correct system to handle the user's query.
                You MUST NOT attempt to answer the user's question directly, perform any calculations, or provide any information beyond the routing keyword.
                Your output MUST be SOLELY one of the following exact keywords, without any additional text, punctuation, or explanation.
                You are a router, not an answer generator. If the query asks for general knowledge, simply output 'GENERAL_KNOWLEDGE'. Do not provide the answer.

                Your output MUST be one of the following:
                - 'RAG'
                - 'JSONPLACEHOLDER_API'
                - 'PRICES_COMPARISON_API' # NEW: Added Prices Comparison API decision
                - 'GENERAL_KNOWLEDGE'

                Here are the rules for choosing the keyword:

                1.  'RAG': Choose this if the user's query is about existing documents, policies, or facts found in a knowledge base (e.g., "What are the rules for residents?", "Tell me about community events", "Where is the nearest post office?").
                2.  'JSONPLACEHOLDER_API': Choose this if the user's query explicitly interacts with JSONPlaceholder API resources: users, posts, or comments, or requires analysis of data from these resources. This includes listing, creating, getting details, updating, deleting, or performing analysis (e.g., "List all users", "Create a new post", "Get comments for post ID 5", "Delete user 10", "Which user has the most posts?", "How many comments does post 3 have?").
                3.  'PRICES_COMPARISON_API': Choose this if the user's query is about comparing prices of products or finding product listings (e.g., "compare prices for iPhone", "find price for Samsung TV", "show me listings for headphones"). # NEW: Added rule for Prices Comparison API
                4.  'GENERAL_KNOWLEDGE': Choose this for any other type of query. This includes:
                    - General questions (e.g., "What is the capital of France?", "Who is the president?").
                    - Greetings, chit-chat, or conversational filler (e.g., "Hello", "How are you?", "Tell me a joke").
                    - ANY arithmetic, calculations, logical reasoning, or math problems, no matter how simple or complex (e.g., "What is 2 + 2?", "Calculate 12345 * 67890", "What is the square root of 81?", "Solve for X: 2X + 5 = 11"). You *must not* solve these; only route them.

                Examples for Routing (Output is ONLY the keyword):
                User: "What are the pool hours?"
                Output: RAG

                User: "How about on Saturdays?"
                Output: RAG

                User: "List all users"
                Output: JSONPLACEHOLDER_API

                User: "Create a new post with title 'My Title' and body 'Hello World'"
                Output: JSONPLACEHOLDER_API

                User: "Which user has the most posts?"
                Output: JSONPLACEHOLDER_API

                User: "compare prices for iPhone" # NEW Example
                Output: PRICES_COMPARISON_API

                User: "find price for Samsung TV" # NEW Example
                Output: PRICES_COMPARISON_API

                User: "What is the capital of France?"
                Output: GENERAL_KNOWLEDGE

                User: "Hello"
                Output: GENERAL_KNOWLEDGE

                User: "What is 2 + 2?"
                Output: GENERAL_KNOWLEDGE

                User: "Calculate 5 times 7, I mean seriously"
                Output: GENERAL_KNOWLEDGE

                User: "Can you tell me how much is 12345 * 67890?"
                Output: GENERAL_KNOWLEDGE

                User: "What is the square root of 81?"
                Output: GENERAL_KNOWLEDGE

                User: "and 4 times 1273017203"
                Output: GENERAL_KNOWLEDGE

                User: "how far is moon from earth"
                Output: GENERAL_KNOWLEDGE

                User: "china population pls"
                Output: GENERAL_KNOWLEDGE
                """
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template("{query}")
        ])
        orchestrator_decision_chain = orchestrator_decision_prompt | orchestrator_llm | StrOutputParser()
        logger.info("Orchestrator decision chain (context-aware) initialized.")

        # --- General Knowledge Chain (with history) ---
        general_knowledge_prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(
                """You are a helpful and knowledgeable AI assistant.
                Answer the user's question directly and concisely based on your general knowledge and the conversation history.
                If the user's query is a math problem or calculation, perform the calculation and provide the numerical answer.
                If you truly do not know the answer or the question is outside your general knowledge,
                politely state "I'm sorry, I don't have enough general knowledge to answer that specific question right now."
                Do not attempt to use any tools or external resources; rely solely on your internal knowledge.
                """
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template("{query}")
        ])
        general_knowledge_chain = general_knowledge_prompt | general_knowledge_llm | StrOutputParser()
        logger.info("General knowledge chain (with history) initialized.")

    except Exception as e:
        logger.critical(f"Failed to initialize one or more core components: {e}. Exiting.", exc_info=True)
        sys.exit(1)


# --- Flask Routes ---

@app.route("/")
def index():
    """Renders the main index page (your chat UI)."""
    return render_template("index.html")


@app.route("/api/bot", methods=["POST"])
def bot_chat():
    """
    Handles chat messages, routing them to the appropriate agent (RAG, JSONPLACEHOLDER_API, PRICES_COMPARISON_API)
    or to a general knowledge fallback based on the user's query intent,
    considering conversation history.
    """
    user_message = None
    session_id = None
    user_id = 'default_user'

    try:
        if not request.is_json:
            logger.error(f"Received non-JSON request to /api/bot. Content-Type: {request.headers.get('Content-Type')}")
            return jsonify({"status": "error", "message": "Invalid request: Expected 'application/json' Content-Type."}), 400

        request_data = request.get_json()
        if not request_data:
            logger.error("Request payload is empty or not valid JSON.")
            return jsonify({"status": "error", "message": "Invalid request: Empty or malformed JSON payload."}), 400

        user_message = request_data.get("query")
        session_id = request_data.get("session_id")
        user_id = request_data.get('user_id', 'default_user')

        if not user_message:
            logger.warning("No 'query' message provided in the request payload.")
            return jsonify({"status": "error", "message": "No message provided. Please type a query."}), 400

        if not session_id:
            session_id = str(uuid.uuid4())
            logger.info(f"Generated new session ID: {session_id} for query: '{user_message}'.")
        else:
            logger.info(f"Using existing session ID: {session_id} for query: '{user_message}'.")

        if session_id not in agent_sessions:
            agent_sessions[session_id] = {
                "memory": ConversationBufferMemory(memory_key="chat_history", return_messages=True),
                "rag_adk_session_id": None,
                "openapi_adk_session_id": None, # For JSONPlaceholder
                "prices_comparison_adk_session_id": None # NEW: For Prices Comparison
            }
            logger.info(f"Initialized new state for session: {session_id}")

        current_memory = agent_sessions[session_id]["memory"]
        chat_history = current_memory.load_memory_variables({})["chat_history"]

        agent_response = ""
        decision = 'UNKNOWN'

        if orchestrator_decision_chain:
            try:
                routing_input = {"query": user_message, "chat_history": chat_history}
                decision_raw = orchestrator_decision_chain.invoke(routing_input)
                decision = decision_raw.strip().upper()
                logger.info(f"Orchestrator decision for '{user_message}' (with history): '{decision_raw.strip()}' -> '{decision}'")

                # IMPORTANT: Updated valid_decisions to include new agent type
                valid_decisions = {'RAG', 'JSONPLACEHOLDER_API', 'PRICES_COMPARISON_API', 'GENERAL_KNOWLEDGE'}
                if decision not in valid_decisions:
                    logger.warning(f"Orchestrator returned an invalid decision '{decision}'. Forcing to GENERAL_KNOWLEDGE.")
                    decision = 'GENERAL_KNOWLEDGE'

            except Exception as e:
                logger.error(f"Error during orchestrator routing decision for '{user_message}': {e}", exc_info=True)
                decision = 'GENERAL_KNOWLEDGE'

        # --- RAG Agent Handling (Existing) ---
        if decision == 'RAG' and rag_agent_client:
            logger.info(f"Routing query to RAG Agent: '{user_message}'")
            rag_adk_session_id = agent_sessions[session_id].get("rag_adk_session_id")

            if not rag_adk_session_id and adk_session_service:
                try:
                    logger.info(f"Creating new ADK session for RAG agent, Flask session {session_id}, user {user_id}")
                    rag_adk_session = asyncio.run(adk_session_service.create_session(
                        app_name=RAG_AGENT_ENGINE_ID,
                        user_id=user_id
                    ))
                    rag_adk_session_id = rag_adk_session.id
                    agent_sessions[session_id]["rag_adk_session_id"] = rag_adk_session_id
                    logger.info(f"Created ADK session {rag_adk_session_id} for RAG agent.")
                except Exception as e:
                    logger.error(f"Failed to create ADK session for RAG agent {user_id}/{session_id}: {e}", exc_info=True)
                    agent_response = "I couldn't start a session for RAG. Falling back to general knowledge."
                    decision = 'GENERAL_KNOWLEDGE'

            if decision == 'RAG': # Re-check decision in case of fallback
                try:
                    rag_actual_content = ""
                    response_stream = rag_agent_client.stream_query(
                        user_id=user_id,
                        session_id=rag_adk_session_id,
                        message=user_message,
                    )
                    for event in response_stream:
                        if isinstance(event, dict):
                            if "content" in event and "parts" in event["content"]:
                                for part in event["content"]["parts"]:
                                    if "text" in part:
                                        rag_actual_content += part["text"]
                            if "tool_code" in event: logger.debug(f"RAG tool_code: {event.get('tool_code')}")
                            if "state_delta" in event: logger.debug(f"RAG state_delta: {event.get('state_delta')}")
                            if "actions" in event: logger.debug(f"RAG actions: {event.get('actions')}")
                        else:
                            logger.warning(f"Received unexpected event type from RAG stream: {type(event)}. Content: {event}")
                            if isinstance(event, str): rag_actual_content += event

                    agent_response = rag_actual_content.strip()

                    no_meaningful_answer_indicators = [
                        "The information is not available in the provided documents.",
                        "I couldn't find any documents",
                        "I'm sorry, I cannot answer this question based on the provided documents.",
                        "I am sorry, but I can't answer that question using the documents I have access to.",
                    ]
                    found_meaningful_answer = True
                    if not agent_response:
                        found_meaningful_answer = False
                        logger.info("RAG Agent returned empty content.")
                    else:
                        for indicator in no_meaningful_answer_indicators:
                            if indicator.lower() in agent_response.lower():
                                found_meaningful_answer = False
                                logger.info(f"RAG Agent response contained no meaningful answer indicator: '{indicator}'.")
                                break

                    if not found_meaningful_answer:
                        logger.info(f"RAG Agent did not provide a substantive answer. Falling back to general knowledge for '{user_message}'.")
                        decision = 'GENERAL_KNOWLEDGE'
                    else:
                        logger.info(f"RAG Agent successfully answered. Response: '{agent_response[:100]}...'")

                except Exception as e:
                    logger.error(f"Error querying RAG Agent for '{user_message}' (ADK Session: {rag_adk_session_id}): {e}", exc_info=True)
                    agent_response = "I encountered an error while retrieving information from the documents. Falling back to general knowledge."
                    decision = 'GENERAL_KNOWLEDGE'


        # --- JSONPlaceholder OpenAPI Agent Handling (Existing) ---
        if decision == 'OPENAPI' and jsonplaceholder_agent_client: # Renamed openapi_agent_client
            logger.info(f"Routing query to JSONPlaceholder OpenAPI Agent: '{user_message}'")
            openapi_adk_session_id = agent_sessions[session_id].get("openapi_adk_session_id")
            if not openapi_adk_session_id and adk_session_service:
                 try:
                    logger.info(f"Creating new ADK session for JSONPlaceholder OpenAPI agent, Flask session {session_id}, user {user_id}")
                    openapi_adk_session = asyncio.run(adk_session_service.create_session(
                        app_name=OPENAPI_AGENT_ENGINE_ID,
                        user_id=user_id
                    ))
                    openapi_adk_session_id = openapi_adk_session.id
                    agent_sessions[session_id]["openapi_adk_session_id"] = openapi_adk_session_id
                    logger.info(f"Created ADK session {openapi_adk_session_id} for JSONPlaceholder OpenAPI agent.")
                 except Exception as e:
                    logger.error(f"Failed to create ADK session for JSONPlaceholder OpenAPI agent {user_id}/{session_id}: {e}", exc_info=True)
                    agent_response = "I couldn't start a session for the JSONPlaceholder OpenAPI service. Falling back to general knowledge."
                    decision = 'GENERAL_KNOWLEDGE'

            if decision == 'OPENAPI': # Re-check decision in case of fallback
                try:
                    openapi_actual_content = ""
                    response_stream = jsonplaceholder_agent_client.stream_query( # Renamed openapi_agent_client
                        user_id=user_id,
                        session_id=openapi_adk_session_id,
                        message=user_message,
                    )
                    for event in response_stream:
                        if isinstance(event, dict):
                            if "content" in event and "parts" in event["content"]:
                                for part in event["content"]["parts"]:
                                    if "text" in part:
                                        openapi_actual_content += part["text"]

                            if "tool_outputs" in event and isinstance(event["tool_outputs"], list):
                                for output in event["tool_outputs"]:
                                    if "data" in output and output["data"] is not None:
                                        try:
                                            formatted_output = json.dumps(output["data"], indent=2)
                                            openapi_actual_content += f"\n```json\n{formatted_output}\n```\n"
                                        except (json.JSONDecodeError, TypeError):
                                            openapi_actual_content += f"\nTool Output: {str(output['data'])}\n"

                            if "tool_code" in event: logger.debug(f"OpenAPI tool_code: {event.get('tool_code')}")
                            if "state_delta" in event: logger.debug(f"OpenAPI state_delta: {event.get('state_delta')}")
                            if "actions" in event: logger.debug(f"OpenAPI actions: {event.get('actions')}")
                        else:
                            logger.warning(f"Received unexpected event type from OpenAPI stream: {type(event)}. Content: {event}")
                            if isinstance(event, str): openapi_actual_content += event

                    agent_response = openapi_actual_content.strip()

                    logger.debug(f"Final JSONPlaceholder OpenAPI actual content before return: \n{agent_response}")

                    if not agent_response:
                        logger.warning(f"JSONPlaceholder OpenAPI Agent returned empty content for '{user_message}'. Falling back to general knowledge.")
                        decision = 'GENERAL_KNOWLEDGE'
                    else:
                        logger.info(f"JSONPlaceholder OpenAPI Agent successfully answered. Response: '{agent_response[:100]}...'")

                except Exception as e:
                    logger.error(f"Error querying JSONPlaceholder OpenAPI Agent for '{user_message}' (ADK Session: {openapi_adk_session_id}): {e}", exc_info=True)
                    agent_response = "I encountered an error while interacting with the JSONPlaceholder OpenAPI service. Falling back to general knowledge."
                    decision = 'GENERAL_KNOWLEDGE'


        # --- NEW: Prices Comparison Agent Handling ---
        if decision == 'PRICES_COMPARISON_API' and prices_comparison_agent_client:
            logger.info(f"Routing query to Prices Comparison Agent: '{user_message}'")
            prices_comparison_adk_session_id = agent_sessions[session_id].get("prices_comparison_adk_session_id")
            if not prices_comparison_adk_session_id and adk_session_service:
                 try:
                    logger.info(f"Creating new ADK session for Prices Comparison agent, Flask session {session_id}, user {user_id}")
                    prices_comparison_adk_session = asyncio.run(adk_session_service.create_session(
                        app_name=PRICES_COMPARE_AGENT_ENGINE_ID, # Use the correct env var for this agent
                        user_id=user_id
                    ))
                    prices_comparison_adk_session_id = prices_comparison_adk_session.id
                    agent_sessions[session_id]["prices_comparison_adk_session_id"] = prices_comparison_adk_session_id
                    logger.info(f"Created ADK session {prices_comparison_adk_session_id} for Prices Comparison agent.")
                 except Exception as e:
                    logger.error(f"Failed to create ADK session for Prices Comparison agent {user_id}/{session_id}: {e}", exc_info=True)
                    agent_response = "I couldn't start a session for the Prices Comparison service. Falling back to general knowledge."
                    decision = 'GENERAL_KNOWLEDGE'

            if decision == 'PRICES_COMPARISON_API': # Re-check decision in case of fallback
                try:
                    prices_comparison_actual_content = ""
                    response_stream = prices_comparison_agent_client.stream_query(
                        user_id=user_id,
                        session_id=prices_comparison_adk_session_id,
                        message=user_message,
                    )
                    for event in response_stream:
                        if isinstance(event, dict):
                            if "content" in event and "parts" in event["content"]:
                                for part in event["content"]["parts"]:
                                    if "text" in part:
                                        prices_comparison_actual_content += part["text"]

                            if "tool_outputs" in event and isinstance(event["tool_outputs"], list):
                                for output in event["tool_outputs"]:
                                    if "data" in output and output["data"] is not None:
                                        try:
                                            formatted_output = json.dumps(output["data"], indent=2)
                                            prices_comparison_actual_content += f"\n```json\n{formatted_output}\n```\n"
                                        except (json.JSONDecodeError, TypeError):
                                            prices_comparison_actual_content += f"\nTool Output: {str(output['data'])}\n"

                            if "tool_code" in event: logger.debug(f"Prices Comparison tool_code: {event.get('tool_code')}")
                            if "state_delta" in event: logger.debug(f"Prices Comparison state_delta: {event.get('state_delta')}")
                            if "actions" in event: logger.debug(f"Prices Comparison actions: {event.get('actions')}")
                        else:
                            logger.warning(f"Received unexpected event type from Prices Comparison stream: {type(event)}. Content: {event}")
                            if isinstance(event, str): prices_comparison_actual_content += event

                    agent_response = prices_comparison_actual_content.strip()

                    logger.debug(f"Final Prices Comparison actual content before return: \n{agent_response}")

                    if not agent_response:
                        logger.warning(f"Prices Comparison Agent returned empty content for '{user_message}'. Falling back to general knowledge.")
                        decision = 'GENERAL_KNOWLEDGE'
                    else:
                        logger.info(f"Prices Comparison Agent successfully answered. Response: '{agent_response[:100]}...'")

                except Exception as e:
                    logger.error(f"Error querying Prices Comparison Agent for '{user_message}' (ADK Session: {prices_comparison_adk_session_id}): {e}", exc_info=True)
                    agent_response = "I encountered an error while interacting with the Prices Comparison service. Falling back to general knowledge."
                    decision = 'GENERAL_KNOWLEDGE'

        # --- General Knowledge Handling (Existing) ---
        if decision == 'GENERAL_KNOWLEDGE':
            logger.info(f"Routing query to General Knowledge LLM: '{user_message}'")
            messages_for_gk = chat_history + [HumanMessage(content=user_message)]

            cleaned_messages_for_gk = []
            for i, msg in enumerate(messages_for_gk):
                if isinstance(msg, (HumanMessage, AIMessage, SystemMessage)) and msg.content is not None and msg.content.strip() != "":
                    cleaned_messages_for_gk.append(msg)
                else:
                    logger.warning(f"Skipping empty or invalid message type in history for GK chain at index {i}: {type(msg)} / Content: {msg.content}")

            try:
                if general_knowledge_chain is None:
                    raise Exception("General knowledge chain is not initialized.")

                gk_response = general_knowledge_chain.invoke({"query": user_message, "chat_history": cleaned_messages_for_gk})

                if gk_response and "I'm sorry, I don't have enough general knowledge" not in gk_response:
                    agent_response = gk_response
                    logger.info(f"General knowledge LLM provided an answer. Response: '{agent_response[:100]}...'")
                else:
                    agent_response = "I'm sorry, I don't have enough general knowledge to answer that specific question right now, and I couldn't find relevant information in specialized systems."
                    logger.info(f"General knowledge LLM provided a non-substantive or fallback response for '{user_message}'.")
            except Exception as e:
                logger.error(f"Error querying General Knowledge LLM for '{user_message}': {e}", exc_info=True)
                agent_response = "I encountered an error while trying to use my general knowledge."

        if not agent_response.strip():
            agent_response = "I apologize, but I couldn't provide a specific answer for that query at this time. It might be outside my current capabilities or require more context."
            logger.warning(f"All agent paths and general knowledge failed for '{user_message}'. Returning generic fallback.")


        current_memory.save_context({"input": user_message}, {"output": agent_response})

        response_data = {"status": "success", "response": agent_response, "session_id": session_id}
        flask_response = jsonify(response_data)
        flask_response.headers['X-Session-ID'] = session_id
        return flask_response, 200

    except Exception as e:
        logger.exception(f"An unhandled critical error occurred in bot_chat endpoint for query '{user_message}' (Session: {session_id}):")
        try:
            req_info = f"Headers: {request.headers}, JSON Data: {request.get_json(silent=True)}"
        except Exception:
            req_info = "Could not parse request data."

        return jsonify({"status": "error", "message": f"An internal error occurred: {e}. Debug info: {req_info}"}), 500


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

    os.makedirs(DOCUMENT_STORAGE_DIRECTORY, exist_ok=True)

    filename = secure_filename(uploaded_file.filename)
    temp_file_path = os.path.join(DOCUMENT_STORAGE_DIRECTORY, filename)

    try:
        uploaded_file.save(temp_file_path)
        logger.info(f"File '{filename}' temporarily saved to '{temp_file_path}'.")

        upload_result = document_storage_utils.upload_document_to_rag_corpus(
            document_path=temp_file_path,
            document_display_name=filename,
            document_description=f"Uploaded document: {filename}"
        )

        final_response_message = upload_result.get("output", "Unknown upload result").strip()

        if upload_result.get("status") == "success":
            logger.info(f"Document '{filename}' successfully processed for RAG Corpus. Result: {final_response_message}")
            return jsonify({
                "status": "success",
                "message": f"File '{filename}' processed successfully. {final_response_message}",
                "doc_id": upload_result.get("rag_file_name")
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

if __name__ == "__main__":
    perform_preflight_checks()
    initialize_all_components()

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
                logger.info(f"Worker {os.getpid()} starting. Components should be ready.")
                return self.application

        logger.info("Starting Gunicorn application.")
        gunicorn_options = {
            'bind': f"{HOST}:{PORT}",
            'workers': GUNICORN_WORKERS,
            'loglevel': LOG_LEVEL.lower(),
            'timeout': GUNICORN_TIMEOUT,
            'reload': FLASK_DEBUG
        }
        StandaloneApplication("app:app", gunicorn_options).run()

    except ImportError:
        logger.warning("Gunicorn not found. Falling back to Flask development server. This is not recommended for production.")
        app.run(host=HOST, port=PORT, debug=FLASK_DEBUG)
    except Exception as e:
        logger.critical(f"Failed to start application: {e}", exc_info=True)
        sys.exit(1)