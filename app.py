# app.py
import os
import uuid
import logging
import asyncio
import sys
import json
import pprint
from typing import Dict, Any, Union, List

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from dotenv import load_dotenv

load_dotenv()

import vertexai
from vertexai import agent_engines
from google.adk.sessions import VertexAiSessionService

# LangChain Agent and Tooling imports (keep these if still used by your ADK setup)
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

from werkzeug.utils import secure_filename

# NEW: Import agent clients (assuming agent_clients.py exists and provides these)
# Make sure agent_clients.py is in the same directory or accessible via Python path
from agent_clients import get_rag_agent_client, get_api_agent_client # Updated import

# Setup logging
logging.basicConfig(level=logging.INFO, stream=sys.stdout)
logger = logging.getLogger(__name__)

# Flask app setup
app = Flask(__name__)
CORS(app) # Enable CORS for all routes

# Environment variables for configuration
PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT")
LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION")
VERTEX_AI_LOGGING_ONLY = os.getenv("VERTEX_AI_LOGGING_ONLY", "False").lower() == "true"
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", 3010))
FLASK_DEBUG = os.getenv("FLASK_DEBUG", "False").lower() == "true"
GUNICORN_WORKERS = int(os.getenv("GUNICORN_WORKERS", 4))
GUNICORN_TIMEOUT = int(os.getenv("GUNICORN_TIMEOUT", 120))
LOG_LEVEL = os.getenv("LOG_LEVEL", "info").upper()

# Initialize Vertex AI
if PROJECT_ID and LOCATION:
    vertexai.init(project=PROJECT_ID, location=LOCATION)
    logger.info(f"Vertex AI initialized for project: {PROJECT_ID}, location: {LOCATION}")
else:
    logger.error("GOOGLE_CLOUD_PROJECT or GOOGLE_CLOUD_LOCATION not set. Vertex AI not initialized.")

# Initialize session service for ADK agents
session_service = VertexAiSessionService()

# Cache for agent clients
agent_clients_cache = {}

# Helper to get the correct agent client
def get_agent_client(agent_type: str):
    if agent_type not in agent_clients_cache:
        if agent_type == "RAG":
            agent_clients_cache[agent_type] = get_rag_agent_client()
        elif agent_type == "API":
            agent_clients_cache[agent_type] = get_api_agent_client() # Use the new generic API client
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")
    return agent_clients_cache[agent_type]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/bot', methods=['POST'])
async def handle_bot_query():
    data = request.json
    user_id = data.get('userId', str(uuid.uuid4()))
    query = data.get('query')
    agent_type = data.get('agentType', 'API') # Default to API agent

    if not query:
        return jsonify({"response": "Please provide a query."}), 400

    logger.info(f"Received query: '{query}' for agent type: '{agent_type}' from user: '{user_id}'")

    try:
        current_session_id = request.headers.get('X-Session-ID', str(uuid.uuid4()))
        logger.info(f"Using session ID: {current_session_id} for query: '{query}'.")

        agent_client = get_agent_client(agent_type)
        if not agent_client:
            return jsonify({"response": f"Agent client for type '{agent_type}' not initialized."}), 500

        # Retrieve or create session for the ADK agent
        session = session_service.get_session(
            session_id=current_session_id,
            agent=agent_client,
            orchestrator_options={"include_trace_in_response": True}
        )

        logger.info(f"Sending query to ADK agent. Current session ID: {current_session_id}")
        response = await session.send_message_async(query)

        final_response_text = response.text
        logger.info(f"ADK Agent successfully answered. Response: '{final_response_text}'")

        return jsonify({
            "response": final_response_text,
            "session_id": current_session_id,
            "orchestrator_trace": response.orchestrator_trace.to_json() if response.orchestrator_trace else {}
        })

    except Exception as e:
        logger.error(f"Error processing request for user {user_id}: {e}", exc_info=True)
        return jsonify({"response": "An error occurred while processing your request. Please try again."}), 500

if __name__ == '__main__':
    if FLASK_DEBUG or os.getenv("WERKZEUG_RUN_MAIN") == "true": # Prevents re-running in debug mode
        try:
            # Try to import gunicorn for production deployment
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