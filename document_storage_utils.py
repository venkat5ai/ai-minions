import logging
import os
import tempfile
from typing import Dict, Any

import vertexai
from vertexai.preview import rag
from google.auth import default as VtxDefaultAuth # Renamed to avoid conflict

from dotenv import load_dotenv

# --- Initial Environment Variable Load (must be at the very top for global access) ---
load_dotenv()

# --- Configuration (Pulled directly from environment variables) ---
GOOGLE_CLOUD_PROJECT = os.getenv("GOOGLE_CLOUD_PROJECT", "api-project-507154614599")
VERTEX_AI_LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1") # Using GOOGLE_CLOUD_LOCATION for consistency

RAG_CORPUS = os.getenv("RAG_CORPUS")

# Document Processing Configuration (for RAG upload)
RAG_CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE", 500))
RAG_CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", 50))
STAGING_BUCKET = os.getenv("STAGING_BUCKET", "gs://ai-assistant-staging-507154614599") # From .env


# --- Logging Setup ---
# Use LOG_LEVEL from .env or default to DEBUG
LOG_LEVEL = os.getenv("LOG_LEVEL", "DEBUG").upper()
logging.basicConfig(level=LOG_LEVEL, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

_vertex_initialized = False

def _initialize_vertex_ai_for_rag():
    """Initializes Vertex AI SDK if not already done."""
    global _vertex_initialized
    if not _vertex_initialized:
        try:
            credentials, project_id = VtxDefaultAuth()            

            # --- NEW DEBUG LINE ---
            if hasattr(credentials, 'service_account_email'):
                logger.info(f"Using service account for Vertex AI init: {credentials.service_account_email}")
            elif hasattr(credentials, 'client_id'):
                logger.info(f"Using client ID for Vertex AI init: {credentials.client_id}")
            else:
                logger.info("Credentials type unknown, cannot determine service account email directly.")
            # --- END NEW DEBUG LINE ---
            effective_project_id = GOOGLE_CLOUD_PROJECT or project_id
            if not effective_project_id:
                raise ValueError("Google Cloud Project ID not found. Set GOOGLE_CLOUD_PROJECT.")            

            vertexai.init(
                project=effective_project_id,
                location=VERTEX_AI_LOCATION,
                credentials=credentials,
                staging_bucket=STAGING_BUCKET
            )
            logger.info(f"Vertex AI SDK initialized for RAG utilities. Project: {effective_project_id}, Location: {VERTEX_AI_LOCATION}")
            _vertex_initialized = True
        except Exception as e:
            logger.error(f"Failed to initialize Vertex AI for RAG: {e}", exc_info=True)
            raise

def upload_document_to_rag_corpus(document_path: str, document_display_name: str, document_description: str) -> Dict[str, Any]:
    """
    Uploads a document to the configured Vertex AI RAG Corpus.
    Assumes the corpus (RAG_CORPUS) already exists.
    """
    _initialize_vertex_ai_for_rag()
    logger.info(f"Attempting to upload document '{document_display_name}' to RAG Corpus '{RAG_CORPUS}'.")

    if not RAG_CORPUS:
        logger.error("RAG_CORPUS is not configured. Cannot upload document.")
        return {
            "status": "error",
            "output": "RAG Corpus is not configured in the application."
        }

    try:
        rag_file = rag.upload_file(
            corpus_name=RAG_CORPUS,
            path=document_path,
            display_name=document_display_name,
            description=document_description,
            # Chunking parameters (chunk_overlap, chunk_size) are typically handled by the service or corpus config.
            # Staging bucket is handled by vertexai.init().
        )
        logger.info(f"Successfully uploaded '{document_display_name}' to corpus. RAG File Name: {rag_file.name}")
        return {
            "status": "success",
            "output": f"Document '{document_display_name}' uploaded successfully to RAG Corpus.",
            "rag_file_name": rag_file.name,
            "rag_file_display_name": rag_file.display_name
        }
    except Exception as e:
        logger.error(f"Error uploading document '{document_display_name}' to RAG Corpus: {e}", exc_info=True)
        return {
            "status": "error",
            "output": f"Failed to upload document '{document_display_name}'. Error: {e}"
        }