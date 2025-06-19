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
    SystemMessagePromptTemplate
)
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage, BaseMessage
from langchain.memory import ConversationBufferMemory # Still used for general knowledge and orchestrator history
from langchain_core.output_parsers import StrOutputParser # Explicitly import StrOutputParser

# For PDF document loading (used in upload endpoint)
from langchain_community.document_loaders import PyPDFLoader 

# Utility for secure filenames
from werkzeug.utils import secure_filename

# NEW: Import document storage utilities
import document_storage_utils


# --- Logging Setup ---
# Use LOG_LEVEL from .env or default to DEBUG
LOG_LEVEL = os.getenv("LOG_LEVEL", "DEBUG").upper()
logging.basicConfig(level=LOG_LEVEL, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration (Pulled directly from environment variables) ---
FLASK_APP_NAME = os.getenv("FLASK_APP_NAME", "AI_Assistant_RAG")
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", 3010))
FLASK_DEBUG = os.getenv("FLASK_DEBUG", "True").lower() in ['true', '1']

GOOGLE_CLOUD_PROJECT = os.getenv("GOOGLE_CLOUD_PROJECT", "api-project-507154614599")
VERTEX_AI_LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1") # Using GOOGLE_CLOUD_LOCATION for consistency
GOOGLE_GENAI_USE_VERTEXAI = int(os.getenv("GOOGLE_GENAI_USE_VERTEXAI", 1))

RAG_CORPUS = os.getenv("RAG_CORPUS")
AGENT_ENGINE_ID = os.getenv("AGENT_ENGINE_ID")

# LLM Configuration
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "gemini-2.0-flash-001")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", 0.2))

# Document Storage Directory (for temporary file uploads)
DOCUMENT_STORAGE_DIRECTORY = os.getenv("DOCUMENT_STORAGE_DIRECTORY", "/app/data")

# Gunicorn Configuration
GUNICORN_TIMEOUT = int(os.getenv("GUNICORN_TIMEOUT", 120))
GUNICORN_WORKERS = int(os.getenv("GUNICORN_WORKERS", 2))


# --- Pre-flight Checks (Crucial for proper operation) ---
def perform_preflight_checks():
    logger.info("Performing pre-flight checks...")
    
    # Check for GOOGLE_APPLICATION_CREDENTIALS for local Flask app access to GCP
    if not os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
        logger.warning("Environment variable GOOGLE_APPLICATION_CREDENTIALS is not set. This is required for authenticating with Google Cloud services like Vertex AI. Please set it to the path of your service account key file within the container (e.g., /app/key.json).")
        # sys.exit(1)
    
    if not GOOGLE_CLOUD_PROJECT.strip():
        logger.error("GOOGLE_CLOUD_PROJECT is not set. Please set the GOOGLE_CLOUD_PROJECT environment variable.")
        sys.exit(1)

    if VERTEX_AI_LOCATION == "us-central1" and not os.getenv("GOOGLE_CLOUD_LOCATION"):
        logger.warning("VERTEX_AI_LOCATION is set to default 'us-central1'. Ensure this is your desired Vertex AI location.")
    
    if not AGENT_ENGINE_ID:
        logger.error("AGENT_ENGINE_ID is not set. This is required for Vertex AI RAG Engine. Please set the AGENT_ENGINE_ID environment variable.")
        sys.exit(1)

    if not RAG_CORPUS:
        logger.error("RAG_CORPUS is not set. This is required for uploading documents to the RAG Corpus and for the RAG agent itself. Please set the RAG_CORPUS environment variable.")
        sys.exit(1)
    
    if not os.getenv("STAGING_BUCKET"):
        logger.error("STAGING_BUCKET is not set. This is required for staging files for RAG corpus upload. Please set the STAGING_BUCKET environment variable.")
        sys.exit(1)

    logger.info("All essential configuration variables for RAG Engine integration are set.")
# --- End Pre-flight Checks ---

# --- Global Flask Application Instance and LLM Initializations ---
app = Flask(FLASK_APP_NAME)
CORS(app)
# app.config.from_object(Config) # Removed: No longer using Config object

perform_preflight_checks()

# Global LLM instance (initialized in initialize_rag_components)
llm = None
general_knowledge_chain = None
orchestrator_decision_chain = None
adk_session_service = None
agent_engine_client = None # Will hold the specific agent engine client

# In-memory store for AgentExecutors and their memories
# {session_id: {"memory": ConversationBufferMemory_instance, "adk_session_id": Optional[str]}}
agent_sessions: Dict[str, Dict[str, Any]] = {}


# --- LLM for general knowledge questions (simpler, direct responses) ---
def initialize_general_knowledge_chain():
    global general_knowledge_chain
    general_knowledge_llm = ChatVertexAI(
        project=GOOGLE_CLOUD_PROJECT,
        location=VERTEX_AI_LOCATION,
        model_name=LLM_MODEL_NAME,
        temperature=0.1 # Keep low for factual answers
    )

    general_knowledge_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful and knowledgeable AI assistant.
                   Answer the user's question directly and concisely based on your general knowledge.
                   If you truly do not know the answer or the question is outside your general knowledge,
                   politely state "I'm sorry, I don't have enough general knowledge to answer that specific question right now."
                   Do not attempt to use any tools or external resources; rely solely on your internal knowledge.
                   """),
        MessagesPlaceholder(variable_name="messages")
    ])

    general_knowledge_chain = general_knowledge_prompt | general_knowledge_llm | StrOutputParser()
    logger.info("General knowledge chain initialized.")


# --- LLM for simple orchestration decision (RAG vs. General Knowledge) ---
def initialize_orchestrator_decision_chain():
    global orchestrator_decision_chain
    orchestrator_llm_decision = ChatVertexAI(
        project=GOOGLE_CLOUD_PROJECT,
        location=VERTEX_AI_LOCATION,
        model_name=LLM_MODEL_NAME,
        temperature=0.0 # Keep very low for deterministic decisions
    )

    # NOTE: This prompt is for the orchestrator, guiding it to decide between RAG and GK.
    # It does NOT directly control the deployed RAG Agent Engine's behavior.
    orchestrator_decision_prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="You are a routing assistant. Your task is to decide whether a user's query should be handled "
                              "by a document retrieval system (RAG) or by general knowledge. "
                              "If the query explicitly asks about specific details that would likely be in an uploaded document "
                              "(e.g., 'What are the pool hours?', 'What is HRISHIKESH's education?', 'Rules for children', "
                              "or questions about policies, manuals, specific products, etc.), "
                              "output 'USE_DOCUMENT_QUERY_TOOL'. "
                              "If the query is a general factual question (e.g., 'Who is the president of the USA?', 'What is the capital of France?'), "
                              "output 'USE_GENERAL_KNOWLEDGE'. "
                              "You MUST respond with only one of these two exact strings: 'USE_DOCUMENT_QUERY_TOOL' or 'USE_GENERAL_KNOWLEDGE'. "
                              "DO NOT include any other text, punctuation, or explanations in your response. Just the exact string."),
        HumanMessage(content="{query}"),
    ])

    orchestrator_decision_chain = orchestrator_decision_prompt | orchestrator_llm_decision | StrOutputParser()
    logger.info("Orchestrator decision chain initialized.")


# --- Function to initialize core RAG components (LLM, Reranker) ---
def initialize_rag_components():
    global llm, adk_session_service, agent_engine_client

    try:
        vertexai.init(
            project=GOOGLE_CLOUD_PROJECT,
            location=VERTEX_AI_LOCATION,
        )
        logger.info(f"Vertex AI SDK initialized. Project: {GOOGLE_CLOUD_PROJECT}, Location: {VERTEX_AI_LOCATION}")

        llm = ChatVertexAI( # This LLM is for the orchestrator and general knowledge
            project=GOOGLE_CLOUD_PROJECT,
            location=VERTEX_AI_LOCATION,
            model_name=LLM_MODEL_NAME,
            temperature=LLM_TEMPERATURE
        )
        logger.info(f"Orchestrator/General Knowledge LLM initialized: {LLM_MODEL_NAME}")

        adk_session_service = VertexAiSessionService(
            project=GOOGLE_CLOUD_PROJECT,
            location=VERTEX_AI_LOCATION
        )
        logger.info("Vertex AI ADK Session Service initialized.")

        if AGENT_ENGINE_ID:
            agent_engine_client = agent_engines.get(AGENT_ENGINE_ID)
            logger.info(f"Agent Engine client retrieved for ID: {AGENT_ENGINE_ID}")
        else:
            logger.error("AGENT_ENGINE_ID is not configured. RAG queries will fail.")

    except Exception as e:
        logger.error(f"Error during RAG component initialization: {e}", exc_info=True)
        sys.exit("Failed to initialize core RAG components.")

    initialize_general_knowledge_chain()
    initialize_orchestrator_decision_chain()
    logger.info("All RAG components initialized.")

# --- Main Orchestrator Logic ---
def main_orchestrator(input_data: Dict[str, Any]) -> str:
    user_query = input_data["input"]
    session_id = input_data["session_id"]
    user_id = input_data.get('user_id', 'default_user')

    logger.info(f"Main Orchestrator: Received query '{user_query}' (Session: {session_id})")

    current_memory = agent_sessions[session_id]["memory"]
    history = current_memory.load_memory_variables({})["chat_history"]
    
    logger.debug(f"Main Orchestrator: Determining query type for '{user_query}'.")
    response_content = "I apologize, but I cannot answer that question at this time. Please try rephrasing." # Default fallback

    try:
        if orchestrator_decision_chain is None:
            logger.error("Orchestrator decision chain is not initialized. Cannot make routing decision.")
            return "The AI system is not fully initialized. Please try again later."

        # The orchestrator's decision prompt is more specific now
        decision_raw = orchestrator_decision_chain.invoke({"query": user_query}).strip().upper() 
        decision = ""
        if "USE_DOCUMENT_QUERY_TOOL" in decision_raw:
            decision = "USE_DOCUMENT_QUERY_TOOL"
        elif "USE_GENERAL_KNOWLEDGE" in decision_raw:
            decision = "USE_GENERAL_KNOWLEDGE"

        logger.info(f"Main Orchestrator: Parsed decision for '{user_query}': '{decision}'")

        if decision == "USE_DOCUMENT_QUERY_TOOL":
            logger.info(f"Main Orchestrator: Decision to use RAG Engine for query: '{user_query}'.")
            
            if not agent_engine_client:
                logger.error("Agent Engine client is not initialized. Cannot perform RAG query.")
                return "The RAG system is not available at the moment. Please try again later."

            adk_session_id = agent_sessions[session_id].get("adk_session_id")
            if not adk_session_id:
                try:
                    logger.info(f"Creating new ADK session for Flask session {session_id}, user {user_id}")
                    logger.debug(f"ADK Session Service app_name for creation: {AGENT_ENGINE_ID}")
                    adk_session = asyncio.run(adk_session_service.create_session(
                        app_name=AGENT_ENGINE_ID, 
                        user_id=user_id
                    ))
                    adk_session_id = adk_session.id
                    agent_sessions[session_id]["adk_session_id"] = adk_session_id
                    logger.info(f"Created ADK session {adk_session_id} for Flask session {session_id}")
                except Exception as e:
                    logger.error(f"Failed to create ADK session for {user_id}/{session_id}: {e}", exc_info=True)
                    return "Failed to initialize a session with the RAG system. Please try again."
            
            logger.debug(f"Using ADK session {adk_session_id} for RAG query.")
            logger.debug(f"Calling agent_engine_client.stream_query with user_id='{user_id}', session_id='{adk_session_id}', message='{user_query}'")
            
            rag_actual_content = ""
            stream_events_received = 0 # NEW: Counter for events received from stream
            
            try:
                response_stream = agent_engine_client.stream_query(
                    user_id=user_id,
                    session_id=adk_session_id,
                    message=user_query,
                )
                
                for event in response_stream:
                    stream_events_received += 1
                    # logger.debug(f"Received event from RAG Engine stream (Event {stream_events_received}): {event}")
                    
                    if "content" in event and "parts" in event["content"]:
                        for part in event["content"]["parts"]:
                            if "text" in part:
                                rag_actual_content += part["text"]
                    # NEW: Log specific event types if they contain relevant information
                    if "tool_code" in event:
                        logger.debug(f"RAG Engine sent tool_code event: {event.get('tool_code')}")
                    if "state_delta" in event:
                        logger.debug(f"RAG Engine sent state_delta event: {event.get('state_delta')}")
                    if "actions" in event:
                        logger.debug(f"RAG Engine sent actions event: {event.get('actions')}")

                logger.debug(f"Finished processing RAG Engine stream. Total events received: {stream_events_received}")
                rag_actual_content = rag_actual_content.strip()
                logger.debug(f"RAG Engine Response (FULL RAW): '{rag_actual_content}'")
                logger.info(f"RAG Engine Response (summary): '{rag_actual_content[:100]}'...")
            except Exception as e:
                logger.error(f"Error querying RAG Engine for '{user_query}': {e}", exc_info=True)
                rag_actual_content = "I encountered an issue while trying to retrieve information from the documents."

            # Check if RAG provided a substantial answer
            no_meaningful_answer_indicators = [
                "The information is not available in the provided documents.",
                "I couldn't find any documents",
                "I'm sorry, I cannot answer this question based on the provided documents.",
                "I am sorry, but I can't answer that question using the documents I have access to.",
                # Add any other specific "no answer" phrases your Agent Engine might produce
            ]

            found_meaningful_answer = True
            if not rag_actual_content:
                found_meaningful_answer = False
                logger.info("RAG Engine returned empty content.")
            else:
                for indicator in no_meaningful_answer_indicators:
                    if indicator.lower() in rag_actual_content.lower(): # Case-insensitive check
                        found_meaningful_answer = False
                        logger.info(f"RAG Engine response contained no meaningful answer indicator: '{indicator}'.")
                        break

            if not found_meaningful_answer:
                logger.info(f"Main Orchestrator: RAG Engine did not provide a substantive answer. Falling back to general knowledge for '{user_query}'.")
                messages_for_gk = history + [HumanMessage(content=user_query)]
                
                cleaned_messages_for_gk = []
                for i, msg in enumerate(messages_for_gk):
                    if isinstance(msg, (HumanMessage, AIMessage, SystemMessage)):
                        if msg.content is not None and msg.content.strip() != "":
                            cleaned_messages_for_gk.append(msg)
                        else:
                            logger.warning(f"Skipping empty message content at index {i} in messages_for_gk: {msg}")
                    else:
                        logger.warning(f"Unexpected message type in messages_for_gk at index {i}: {type(msg)}. Content: {msg}")
                        if msg.content is not None and msg.content.strip() != "":
                            cleaned_messages_for_gk.append(msg)

                logger.debug(f"General Knowledge Chain Input: Messages Length: {len(cleaned_messages_for_gk)}")
                gk_response = general_knowledge_chain.invoke({"messages": cleaned_messages_for_gk})
                
                if gk_response and "I'm sorry, I don't have enough general knowledge" not in gk_response:
                    logger.info(f"Main Orchestrator: General knowledge LLM provided an answer. Response: {gk_response[:100]}...") 
                    response_content = gk_response
                else:
                    logger.info(f"Main Orchestrator: Both RAG and general knowledge failed for '{user_query}'.")
                    response_content = "I couldn't find relevant information in your documents, and I don't have general knowledge to answer that."
            else:
                logger.info(f"Main Orchestrator: RAG Engine successfully answered. Final RAG content: '{rag_actual_content[:100]}'...")
                response_content = rag_actual_content
        elif decision == "USE_GENERAL_KNOWLEDGE":
            logger.info(f"Main Orchestrator: Decision to use general knowledge. Executing general knowledge path.")
            messages_for_gk = history + [HumanMessage(content=user_query)]

            cleaned_messages_for_gk = []
            for i, msg in enumerate(messages_for_gk):
                if isinstance(msg, (HumanMessage, AIMessage, SystemMessage)):
                    if msg.content is not None and msg.content.strip() != "":
                        cleaned_messages_for_gk.append(msg)
                    else:
                        logger.warning(f"Skipping empty message content at index {i} in messages_for_gk: {msg}")
                else:
                    logger.warning(f"Unexpected message type in messages_for_gk at index {i}: {type(msg)}. Content: {msg}")
                    if msg.content is not None and msg.content.strip() != "":
                        cleaned_messages_for_gk.append(msg)

            logger.debug(f"General Knowledge Chain Input: Messages Length: {len(cleaned_messages_for_gk)}")
            gk_response = general_knowledge_chain.invoke({"messages": cleaned_messages_for_gk})
            
            if gk_response and "I'm sorry, I don't have enough general knowledge" not in gk_response:
                logger.info(f"Main Orchestrator: General knowledge LLM provided an answer. Response: {gk_response[:100]}...") 
                response_content = gk_response
            else:
                logger.info(f"Main Orchestrator: General knowledge failed for '{user_query}'.")
                response_content = "I'm sorry, I don't have enough general knowledge to answer that specific question right now."
        else:
            logger.warning(f"Main Orchestrator: Unparsed decision from orchestrator LLM: '{decision_raw}'. Falling back to generic response.")
            response_content = "I'm not sure how to handle that request. Could you please rephrase it?"

        logger.info(f"Main Orchestrator: Returning response_content: '{response_content[:100]}'...")
        return response_content

    except Exception as e:
        logger.error(f"Main Orchestrator: Error during query path execution for '{user_query}': {e}", exc_info=True)
        error_response = "I encountered an unexpected error while trying to process your request. Please try again later."
        return error_response


# --- Flask Routes ---

@app.route('/api/bot', methods=['POST'])
def handle_bot_request():
    """
    Main endpoint for the bot to receive and process requests.
    Generates/manages session_id for RAG context.
    """
    if not request.is_json:
        logger.warning("Received non-JSON request to /api/bot (non-JSON content-type).")
        return jsonify({"status": "error", "message": "Request must be JSON"}), 400

    data = request.get_json()
    user_query = data.get('query')
    user_id = data.get('user_id', 'default_user')
    
    if not user_query:
        logger.warning("Invalid request: Missing 'query' in JSON payload.")
        return jsonify({"status": "error", "message": "Missing 'query' in request."}), 400

    session_id = request.headers.get('X-Session-ID') or data.get('session_id')
    if not session_id:
        session_id = str(uuid.uuid4())
        logger.info(f"Generated new session ID: {session_id} for query: '{user_query}'.")
    else:
        logger.info(f"Using existing session ID: {session_id} for query: '{user_query}'.")

    # Initialize session memory if not already present
    if session_id not in agent_sessions:
        agent_sessions[session_id] = {
            "memory": ConversationBufferMemory(memory_key="chat_history", return_messages=True),
            "adk_session_id": None # Initialize ADK session ID as None
        }
        logger.info(f"Initialized new memory for session: {session_id}")

    current_memory = agent_sessions[session_id]["memory"] # Get the memory instance for this session

    try:
        final_response = main_orchestrator({"input": user_query, "session_id": session_id, "user_id": user_id})
        
        if not final_response.strip():
            final_response = "I'm sorry, I couldn't provide a specific answer for that query at this time. It might be outside my current knowledge or require more context."
            logger.warning(f"Final response was empty for query '{user_query}' after main orchestrator. Returning generic fallback.")

        logger.info(f"Overall final response to UI for query '{user_query}' (Session: {session_id}): {final_response[:200]}...") 
        
        # Add the complete turn (user input + model response) to session history.
        current_memory.save_context({"input": user_query}, {"output": final_response}) 
        
        response_data = {"status": "success", "response": final_response, "session_id": session_id}
        flask_response = jsonify(response_data)
        flask_response.headers['X-Session-ID'] = session_id
        return flask_response, 200
        
    except Exception as e:
        logger.exception(f"An error occurred while processing the request for query '{user_query}' (Session: {session_id}):")
        error_message = str(e)
        # Attempt to add error message to history if possible, using save_context
        if session_id in agent_sessions:
            current_memory.save_context({"input": user_query}, {"output": f"Error: {error_message}"})
        return jsonify({"status": "error", "message": error_message}), 500

@app.route('/api/documents/upload', methods=['POST'])
def upload_document_endpoint():
    """
    Dedicated endpoint for uploading documents via HTTP POST to the Vertex AI RAG Corpus.
    """
    session_id = request.headers.get('X-Session-ID') or request.form.get('session_id')
    if not session_id:
        session_id = str(uuid.uuid4())
        logger.info(f"Generated new session ID for upload.")
    else:
        logger.info(f"Using existing session ID for upload: {session_id}.")

    # user_id is not directly used for upload_file to corpus, but good to have if needed for logging/auditing
    # user_id = request.form.get('user_id', 'default_user') 

    if 'file' not in request.files:
        logger.warning("Invalid document upload request: Missing 'file' part.")
        return jsonify({"status": "error", "message": "Missing 'file' in request."}), 400
    
    uploaded_file = request.files['file']
    if uploaded_file.filename == '':
        logger.warning("Invalid document upload request: No selected file.")
        return jsonify({"status": "error", "message": "No selected file."}), 400

    temp_upload_dir = DOCUMENT_STORAGE_DIRECTORY # Use global constant
    os.makedirs(temp_upload_dir, exist_ok=True) # This will create the directory if it doesn't exist
    
    filename = secure_filename(uploaded_file.filename)
    temp_file_path = os.path.join(temp_upload_dir, filename)

    try:
        uploaded_file.save(temp_file_path)
        logger.info(f"File '{filename}' temporarily saved to '{temp_file_path}'.") 

        # The RAG Engine's rag.upload_file handles PDF parsing and chunking.
        # We just need to pass the path.
        # A display name and description are useful for the RAG Corpus.
        document_display_name = filename
        document_description = f"Uploaded document: {filename} for session {session_id}" # Or a more generic description

        upload_result = document_storage_utils.upload_document_to_rag_corpus(
            document_path=temp_file_path,
            document_display_name=document_display_name,
            document_description=document_description
        )
        
        doc_id = upload_result.get("rag_file_name") # The RAG file resource name can serve as a unique ID
        final_response_message = upload_result.get("output", "Unknown upload result").strip()

        if upload_result.get("status") == "success":
            logger.info(f"Document '{doc_id}' processed for session '{session_id}'. Output: {final_response_message}") 
            return jsonify({
                "status": "success",
                "message": f"File '{filename}' processed successfully. Response: {final_response_message}", 
                "doc_id": doc_id, # This will be the RAG file resource name
                "session_id": session_id
            }), 200
        else:
            logger.error(f"Failed to process document '{filename}'. Output: {final_response_message}") 
            return jsonify({"status": "error", "message": f"Failed to process document: {final_response_message}"}), 500

    except Exception as e:
        logger.exception(f"An error occurred during document upload for file '{filename}' (Session: {session_id}):")
        return jsonify({"status": "error", "message": str(e)}), 500
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
            logger.info(f"Cleaned up temporary file: '{temp_file_path}'.") 

@app.route('/')
def index():
    """Renders the main index.html page for the web UI."""
    return render_template('index.html')

@app.route('/health', methods=['GET'])
def health_check():
    """Provides a simple health check endpoint for the Flask application."""
    logger.debug("Health check requested.") 
    return jsonify({"status": "healthy", "message": f"{FLASK_APP_NAME} is running"}), 200

if __name__ == '__main__':
    model_mode = sys.argv[1].lower() if len(sys.argv) > 1 else "rag_engine" # Defaulting to new mode
    logger.info(f"Starting application in '{model_mode}' mode.")

    try:
        from gunicorn.app.wsgiapp import WSGIApplication
        from gunicorn.config import Config as GunicornConfig
    except ImportError:
        logger.error("Gunicorn not found. Please ensure it's in requirements.txt. Falling back to Flask development server.")
        app.run(host=HOST, port=PORT, debug=FLASK_DEBUG)
        sys.exit(0)

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

    # Initialize RAG components before Gunicorn starts workers
    initialize_rag_components() 
    logger.info("RAG components ready for requests (main process)!")

    gunicorn_options = {
        'bind': f"{HOST}:{PORT}",
        'workers': GUNICORN_WORKERS,
        'loglevel': LOG_LEVEL.lower(),
        'timeout': GUNICORN_TIMEOUT,
        'reload': FLASK_DEBUG
    }
    
    StandaloneApplication("app:app", gunicorn_options).run()
