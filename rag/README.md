# ai-assistant

An advanced AI-powered document assistant leveraging Google Vertex AI (RAG Corpus and Agent Engine), Flask, and the Google Agent Development Kit (ADK) for document ingestion, semantic search, and robust question answering (RAG).

---

## Features
- **Vertex AI RAG Integration**:
    - Utilizes Vertex AI RAG Corpus for efficient document storage and retrieval.
    - Deploys a custom RAG agent (defined with Google ADK) to Vertex AI Agent Engine for powerful, context-aware Q&A.
- **Document Upload & Ingestion**: Upload documents (e.g., PDF) directly to the configured Vertex AI RAG Corpus via the application or helper scripts.
- **Intelligent Q&A**: Ask questions and receive answers synthesized from your uploaded documents, powered by Gemini models via the deployed Agent Engine.
- **Flask Web Interface**: Simple UI for interacting with the assistant and uploading documents.
- **Google Cloud Native**: Built on Vertex AI services, ensuring scalability and integration with the Google Cloud ecosystem.
- **Dockerized Deployment**:
    - Easy to build and run using Docker.
    - Supports multiple operational modes via `entrypoint.sh`:
        - `deploy`: Deploys/re-deploys the RAG agent to Vertex AI Agent Engine and then starts the application.
        - `cloud`: Runs the application using an existing, already deployed Agent Engine.
        - `destroy`: Cleans up the deployed Vertex AI Agent Engine resource.

---

## Architecture Overview

The system consists of several key components:
1.  **Flask Web Application (`app.py`)**: Provides the user interface and API endpoints. It orchestrates requests, manages user sessions, and interacts with the deployed Vertex AI Agent Engine.
2.  **RAG Agent Blueprint (`rag/agent.py`)**: Defines the core logic of the RAG agent using the Google Agent Development Kit (ADK). This includes the LLM (e.g., Gemini), tools (like `VertexAiRagRetrieval`), and instructions.
3.  **Deployment Script (`deployment/deploy.py`)**: Takes the agent blueprint from `rag/agent.py` and deploys it as a service on Vertex AI Agent Engine.
4.  **Vertex AI RAG Corpus**: A managed service where your documents are uploaded, chunked, embedded, and indexed for retrieval.
5.  **Vertex AI Agent Engine**: The deployed RAG agent service that receives queries, retrieves relevant information from the RAG Corpus, and uses an LLM to generate answers.
6.  **Document Storage Utilities (`document_storage_utils.py`)**: Helper functions for uploading documents to the RAG Corpus.
7.  **Docker (`Dockerfile`, `entrypoint.sh`)**: Containerizes the application for consistent deployment and manages different operational modes.

### Architectural Flowcharts
These flowcharts provide a visual representation of the data and control flow within the AI Assistant application.

1. User Query Flow
This diagram illustrates how a user's query is processed, routed, and answered by the AI Assistant.

+------------------+     HTTP Request    +-------------------------------------------------+
|   User (UI)      |-------------------->|             Flask App (app.py)                  |
+------------------+                     |                                                 |
                                         |    +--------------------------+                 |
                                         |    |  /api/bot Endpoint       |<----------------+
                                         |    |  (Handles user queries)  |                 |
                                         |    +--------------------------+                 |
                                         |              |                                  |
                                         |              v                                  |
                                         |    +--------------------------+                 |
                                         |    |   main_orchestrator      |                 |
                                         |    |   (Core Logic)           |                 |
                                         |    +--------------------------+                 |
                                         |              |                                  |
                                         |              v                                  |
                                         |    +--------------------------+                 |
                                         |    | Conversational Memory    |                 |
                                         |    | (ConversationBufferMemory)|<--------------->|
                                         |    | (Per User Session)       |                 |
                                         |    +--------------------------+                 |
                                         |              |                                  |
                                         |              v                                  |
                                         |    +--------------------------+                 |
                                         |    |    Orchestrator LLM      |                 |
                                         |    |    (Routes Query)        |<----------------+
                                         |    +--------------------------+                 |
                                         |              |                                  |
          (Decision: USE_DOCUMENT_QUERY_TOOL)           |                                  |
                 +-------------------------------------------------------------------------+
                 |            |                                                            |
                 |            v                                                            |
                 |    +--------------------------+                                       |
                 |    | ADK Session Service      |                                       |
                 |    | (Manages sessions w/     |                                       |
                 |    |  Agent Engine)           |                                       |
                 |    +--------------------------+                                       |
                 |              |                                                          |
                 |              v                                                          |
                 |    +--------------------------+                                       |
                 |    | Vertex AI Agent Engine   |<--------------------------+           |
                 |    | (Deployed RAG Model)     |                           |           |
                 |    |                          |<--- Retrieval (Search) ---->|           |
                 |    | - Knowledge Base         |                           |           |
                 |    |   (RAG Corpus)           |                           |           |
                 |    | - Internal LLM           |                           |           |
                 |    |   (Generates Response)   |                           |           |
                 |    | - Internal Tools         |                           |           |
                 |    +--------------------------+                           |           |
                 |              | Streamed Response                          |           |
                 |              v                                            |           |
                 |    +--------------------------+                           |           |
                 |    | main_orchestrator        |                           |           |
                 |    | (Processes Stream,       |                           |           |
                 |    |  Handles RAG Content)    |                           |           |
                 |    +--------------------------+                           |           |
                 |              |                                            |           |
                 |              v (If RAG Fails/No Meaningful Answer)        |           |
                 |------------------------------------------------------------------------+
                               |            |
                               |            v
                               |    +--------------------------+
                               |    | General Knowledge LLM    |
                               |    | (e.g., Gemini 2.0 Flash) |
                               |    | (Answers general queries)|
                               |    +--------------------------+
                               |              |
                               |              v
                               |    +--------------------------+
                               |    | Final Response           |
                               |<---| (To User)              |
                               +--------------------------+

2. Document Upload Flow
This diagram outlines the process of how PDF documents are uploaded and ingested into the RAG Corpus.

+------------------+     HTTP Request    +-------------------------------------------------+
|   User (UI)      |-------------------->|             Flask App (app.py)                  |
| (Upload PDF)     |                     |                                                 |
+------------------+                     |    +--------------------------+                 |
                                         |    |  /api/documents/upload   |<----------------+
                                         |    |  (Handles PDF uploads)   |                 |
                                         |    +--------------------------+                 |
                                         |              |                                  |
                                         |              v                                  |
                                         |    +--------------------------+                 |
                                         |    | Save Temp File Locally   |                 |
                                         |    | (e.g., /app/data)        |                 |
                                         |    +--------------------------+                 |
                                         |              |                                  |
                                         |              v                                  |
                                         |    +--------------------------+                 |
                                         |    | document_storage_utils.py|<----------------+
                                         |    | (Upload Logic)           |                 |
                                         |    +--------------------------+                 |
                                         |              |                                  |
                                         |              v                                  |
                                         |    +--------------------------+                 |
                                         |    |  Vertex AI SDK Init      |                 |
                                         |    |  (Uses GOOGLE_APPLICATION |                 |
                                         |    |   _CREDENTIALS / Metadata|                 |
                                         |    |   Server for Auth)       |                 |
                                         |    +--------------------------+                 |
                                         |              |                                  |
                                         |              v                                  |
                                         |    +--------------------------+                 |
                                         |    | vertexai.preview.rag.    |                 |
                                         |    | upload_file() API Call   |                 |
                                         |    +--------------------------+                 |
                                         |              |                                  |
                                         |              v                                  |
+--------------------------+     Staging (Auto-handled)     +--------------------------+
| Google Cloud Storage     |-------------------------------->|    Vertex AI RAG Corpus  |
| (Staging Bucket)         |                                 |                          |
+--------------------------+                                 |  - PDF Parsing           |
                                                             |  - Chunking              |
                                                             |  - Embedding             |
                                                             |  - Indexing              |
                                                             +--------------------------+
                                                                         ^
                                                                         | Ingestion
                                                                         |
                                         +--------------------------+    |
                                         | Upload Status/Result     |<---+
                                         | (Success/Fail, Doc ID)   |
                                         +--------------------------+
                                                     |
                                                     v
                                         +--------------------------+
                                         | Clean Up Temp File       |
                                         +--------------------------+
                                                     |
                                                     v
                                         +--------------------------+
                                         | Return Response to UI    |
                                         +--------------------------+

---
## Detailed Design/Architectural Overview
This section provides a deeper dive into the components and data flow within your AI Assistant application, detailing how it leverages various Google Cloud Vertex AI capabilities for both RAG and general knowledge interactions.

Core Components and Their Roles
Flask Application (app.py):

Web Server: Serves the index.html UI and exposes REST API endpoints (/api/bot, /api/documents/upload, /health).

Orchestrator (LLM - ChatVertexAI with LLM_MODEL_NAME, e.g., gemini-2.0-flash-001):

Brain of the Router: This is a dedicated LLM instance within your Flask app, configured with a temperature=0.0 (for deterministic behavior) and a specific system prompt.

Purpose: Its primary role is to analyze each incoming user query and decide whether it should be answered using the document retrieval system (RAG) or by general knowledge. It outputs one of two exact strings: USE_DOCUMENT_QUERY_TOOL or USE_GENERAL_KNOWLEDGE.

Conversational Memory (agent_sessions dictionary holding langchain.memory.ConversationBufferMemory instances):

Purpose: Maintains the history of the conversation for each unique user session. This is crucial for the orchestrator and the general knowledge LLM to understand context over multiple turns.

Mechanism: Stores HumanMessage (user input) and AIMessage (model response) objects, allowing the LLMs to refer to previous interactions. Each session is identified by a session_id.

Authentication:

Local Development: Relies on the GOOGLE_APPLICATION_CREDENTIALS environment variable, pointing to your user's application_default_credentials.json file (generated by gcloud auth application-default login). This provides the broad scopes needed for both RAG upload and query.

Cloud Run Deployment: Will implicitly use the service account assigned to the Cloud Run service, leveraging the metadata server to obtain credentials with necessary scopes.

Document Upload Flow (/api/documents/upload endpoint):

Client (UI): Sends a multipart/form-data POST request containing the PDF file.

Flask App (app.py):

Receives the uploaded file.

Saves it temporarily to a local DOCUMENT_STORAGE_DIRECTORY (e.g., /app/data) using werkzeug.utils.secure_filename.

Invokes document_storage_utils.upload_document_to_rag_corpus.

document_storage_utils.py:

Initializes vertexai.init() with the determined credentials (primarily from GOOGLE_APPLICATION_CREDENTIALS locally, or metadata server on Cloud Run).

Calls vertexai.preview.rag.upload_file(): This is the core API call.

The file is automatically staged in the STAGING_BUCKET (a Google Cloud Storage bucket configured in your .env).

Vertex AI's RAG service then processes the document:

PDF Parsing: Extracts text content from the PDF.

Chunking: Breaks down the document into smaller, manageable pieces (chunks).

Embedding: Converts these text chunks into numerical vector representations (embeddings) using a model like text-embedding-004.

Indexing: Stores these embeddings in a searchable index within the Vertex AI RAG Corpus.

Returns success/failure status and the rag_file_name (resource ID).

Temporary File Cleanup: The Flask app ensures the temporary local file is deleted after the upload attempt.

Query Flow (/api/bot endpoint):

User Query: The user submits a text query.

Flask App (app.py - main_orchestrator function):

Session Management: Retrieves or creates a ConversationBufferMemory instance for the session_id.

Orchestrator Decision: The user query (along with the chat_history) is sent to the Orchestrator LLM.

Decision Branches:

a) If USE_DOCUMENT_QUERY_TOOL is decided:

ADK Session Management (VertexAiSessionService): Ensures a persistent session is maintained with the deployed Agent Engine. This allows for conversational turns within the Agent Engine's context.

Vertex AI Agent Engine Client (agent_engines.get(AGENT_ENGINE_ID)):

The user's original query is directly sent to this deployed Agent Engine via agent_engine_client.stream_query().

Internal RAG Pipeline (within Agent Engine):

Knowledge Base (RAG Corpus): The Agent Engine's primary "tool" is its association with your deployed Vertex AI RAG Corpus. It searches this corpus using the query to retrieve relevant document chunks.

LLM (Internal to Agent Engine): The Agent Engine itself contains an LLM (Vertex AI's managed models, like Gemini) that synthesizes the final answer by considering the user's query and the retrieved chunks. This LLM's identity and specific configuration are managed by the Agent Engine service, not directly by your app.py.

Tools (within Agent Engine): While your app.py treats the Agent Engine as a single tool, the Agent Engine itself might internally orchestrate multiple steps or sub-tools (like a retrieval tool, a summarization tool, etc.) to generate its response.

Streaming Response: The Agent Engine streams its response back to your Flask app. Your app.py processes these stream events to reconstruct the full answer.

Response Handling: If the Agent Engine provides a meaningful answer, it's returned to the user.

Fallback to General Knowledge: If the RAG Agent Engine returns empty content or an indicator that it couldn't find relevant information, the query is then routed to the General Knowledge LLM.

b) If USE_GENERAL_KNOWLEDGE is decided:

General Knowledge LLM (ChatVertexAI with LLM_MODEL_NAME, e.g., gemini-2.0-flash-001):

This is a separate LLM instance within your Flask app, configured to answer broad factual questions.

It takes the user query and the chat_history (from ConversationBufferMemory) as input.

It does not perform any document retrieval; it relies solely on its pre-trained knowledge.

Response Handling: The LLM's direct response is returned.

Chat History Update: After the final response is generated, both the user's query and the model's response are saved back into the ConversationBufferMemory for the current session.

Technical Flow Diagram
graph TD
    subgraph Frontend (User Browser/UI)
        UI[Web UI]
    end

    subgraph Backend (Flask AI Assistant App - app.py)
        direction LR
        API_BOT_EP[api/bot Endpoint]
        API_UPLOAD_EP[api/documents/upload Endpoint]
        MAIN_ORCH[main_orchestrator Function]
        CONV_MEM[ConversationBufferMemory (in agent_sessions)]
        ORCH_LLM[Orchestrator LLM]
        ADK_SESSION_SVC[VertexAiSessionService]
        GEN_K_LLM[General Knowledge LLM]
    end

    subgraph Google Cloud Vertex AI
        RAG_CORPUS[Vertex AI RAG Corpus]
        AGENT_ENGINE[Vertex AI Agent Engine (Deployed)]
        GCS_STAGING[Google Cloud Storage (Staging Bucket)]
    end

    subgraph Authentication
        AUTH[GOOGLE_APPLICATION_CREDENTIALS / Metadata Server]
    end

    UI -- User Query --> API_BOT_EP
    API_BOT_EP -- Load/Update --> CONV_MEM
    API_BOT_EP -- Input Query & History --> MAIN_ORCH
    MAIN_ORCH -- Query & History --> ORCH_LLM
    ORCH_LLM -- Decision (USE_DOCUMENT_QUERY_TOOL / USE_GENERAL_KNOWLEDGE) --> MAIN_ORCH

    MAIN_ORCH -- If USE_DOCUMENT_QUERY_TOOL --> ADK_SESSION_SVC
    ADK_SESSION_SVC -- Manage Session --> AGENT_ENGINE
    MAIN_ORCH -- User Query --> AGENT_ENGINE
    AGENT_ENGINE -- Retrieval --> RAG_CORPUS
    AGENT_ENGINE -- Response Generation (Internal LLM) --> AGENT_ENGINE
    AGENT_ENGINE -- Streamed Response --> MAIN_ORCH
    MAIN_ORCH -- Parse Streamed Content --> MAIN_ORCH
    MAIN_ORCH -- If RAG Fails/No Answer --> GEN_K_LLM

    MAIN_ORCH -- If USE_GENERAL_KNOWLEDGE --> GEN_K_LLM
    GEN_K_LLM -- Generate Response --> MAIN_ORCH

    MAIN_ORCH -- Final Response --> API_BOT_EP
    API_BOT_EP -- Save Context --> CONV_MEM
    API_BOT_EP -- Return Response --> UI

    UI -- Upload PDF --> API_UPLOAD_EP
    API_UPLOAD_EP -- Save Temp File --> LOCAL_FS[Local Flask App File System]
    LOCAL_FS -- Upload Path & Doc Metadata --> DOCUMENT_STORAGE_UTILS_PY[document_storage_utils.py]
    DOCUMENT_STORAGE_UTILS_PY -- Credentials --> AUTH
    DOCUMENT_STORAGE_UTILS_PY -- Upload File API Call --> GCS_STAGING
    GCS_STAGING -- Ingest & Index --> RAG_CORPUS
    DOCUMENT_STORAGE_UTILS_PY -- Cleanup Temp File --> LOCAL_FS
    DOCUMENT_STORAGE_UTILS_PY -- Upload Status --> API_UPLOAD_EP
    API_UPLOAD_EP -- Return Status --> UI


Key Takeaways:

Orchestration: Your app.py acts as a smart router, deciding the best path for each user query based on its nature.

Separation of Concerns: General knowledge and RAG are handled by distinct LLM chains/components, allowing for optimized prompts and models for each task.

RAG Agent Engine as a "Black Box": From app.py's perspective, the Vertex AI Agent Engine is a powerful pre-configured tool that performs both retrieval and generation internally. You send it a query and get an answer back.

Conversational Continuity: ConversationBufferMemory is vital for maintaining the conversational flow and enabling the LLMs to understand the context of follow-up questions.

This detailed architecture should provide a comprehensive understanding of your application's current workings and its interactions with Google Cloud Vertex AI.

---
## Prerequisites

- **Google Cloud Project**:
    - Billing enabled.
    - APIs Enabled:
        - Vertex AI API (`aiplatform.googleapis.com`)
        - Cloud Storage API (`storage.googleapis.com`)
        - Identity and Access Management (IAM) API (`iam.googleapis.com`)
        - Cloud Resource Manager API (`cloudresourcemanager.googleapis.com`) (often enabled by default)
- **Service Account**:
    - Create a service account in your GCP project.
    - Download its JSON key file.
    - Grant necessary IAM roles to this service account:
        - `Vertex AI User` (for general Vertex AI access)
        - `Vertex AI RAG Admin` (or more granular `aiplatform.ragCorpora.*`, `aiplatform.ragFiles.*` permissions for managing RAG corpora and files)
        - `Vertex AI Reasoning Engines Admin` (or `aiplatform.reasoningEngines.*` for deploying and managing Agent Engines)
        - `Storage Admin` or `Storage Object Admin` (for the `STAGING_BUCKET` used by ADK and RAG uploads)
        - `Service Account User` (if the Agent Engine needs to impersonate this or another service account, though typically not required for this setup if the engine runs with its own identity).
- **`gcloud` CLI**: Installed and configured (e.g., `gcloud auth application-default login` for local testing if not using the service account directly for helper scripts).
- **Docker**: Installed and running.

---

## Setup Instructions

### 1. Clone the Repository
```powershell
git clone https://github.com/venkat5ai/ai-assistant.git
cd ai-assistant
```

### 2. Set Up Environment Variables
- Copy your Google Cloud service account key to the project directory (e.g., as `key.json`).
- Create a `.env` file (see `config.py` for required variables):

```env
GOOGLE_APPLICATION_CREDENTIALS=/app/key.json
GOOGLE_CLOUD_PROJECT=your_project_id
VERTEX_AI_LOCATION=us-central1
STAGING_BUCKET=Staging bucket name for ADK agent deployment to Vertex AI Agent Engine gs://bucket-name
ME_INDEX_ID=your_matching_engine_index_id
ME_ENDPOINT_ID=your_matching_engine_endpoint_id
```

---

## Docker Commands

### Build the Docker Image
```powershell
docker build -t venkat5ai/ai-assistant:1.0 -t venkat5ai/ai-assistant:latest .
```

### Run the Docker Container
```powershell
docker run --name ai-assistant --rm -p 3010:3010 -v "%GOOGLE_APPLICATION_CREDENTIALS%":/tmp/keys.json -e GOOGLE_APPLICATION_CREDENTIALS="/tmp/keys.json" venkat5ai/ai-assistant:1.0 cloud

docker run --name ai-assistant --rm -p 3010:3010 -v "%GOOGLE_APPLICATION_CREDENTIALS%":/tmp/keys.json -e GOOGLE_APPLICATION_CREDENTIALS="/tmp/keys.json" venkat5ai/ai-assistant:1.0 deploy

docker run --name ai-assistant --rm -p 3010:3010 -v "%GOOGLE_APPLICATION_CREDENTIALS%":/tmp/keys.json -e GOOGLE_APPLICATION_CREDENTIALS="/tmp/keys.json" venkat5ai/ai-assistant:1.0 destroy
```

### Run with Interactive Shell (Bash)
```powershell
docker run --name ai-assistant --rm -it -p 3010:3010 -v ".:/app" -v "%GOOGLE_APPLICATION_CREDENTIALS%":/tmp/keys.json -e GOOGLE_APPLICATION_CREDENTIALS="/tmp/keys.json" --entrypoint /bin/bash  venkat5ai/ai-assistant:1.0
```

### Access the Container Shell (if already running)
```powershell
docker exec -it ai-assistant sh
```

---
## gcloud commands

### GCP Artifact Registry image push
```
	gcloud services enable artifactregistry.googleapis.com
	gcloud artifacts repositories create ai-assistant-repo --repository-format=docker --location=us-central1 --description="Docker images for AI Assistant"	
	gcloud auth configure-docker us-central1-docker.pkg.dev
	docker build --no-cache -t venkat5ai/ai-assistant:latest .
  docker push  venkat5ai/ai-assistant:latest
	docker tag venkat5ai/ai-assistant:latest us-central1-docker.pkg.dev/api-project-507154614599/ai-assistant-repo/ai-assistant:latest
	docker push us-central1-docker.pkg.dev/api-project-507154614599/ai-assistant-repo/ai-assistant:latest
```
### Cloud run Deploy
```
	gcloud services enable run.googleapis.com
	gcloud run deploy ai-assistant-service --image us-central1-docker.pkg.dev/api-project-507154614599/ai-assistant-repo/ai-assistant:latest     --platform managed --region us-central1 --allow-unauthenticated --service-account ai-assistant-sa@api-project-507154614599.iam.gserviceaccount.com --cpu 2 --memory 2Gi --port 3010 --concurrency 80 --timeout 300s --min-instances 0 --max-instances 1  # --ingress=internal-and-cloud-load-balancing  # Restricts to LB/VPC only	
			Deploying container to Cloud Run service [ai-assistant-service] in project [api-project-507154614599] region [us-central1]
			OK Deploying... Done.
			  OK Creating Revision...
			  OK Routing traffic...
			  OK Setting IAM Policy...
			Done.
			Service [ai-assistant-service] revision [ai-assistant-service-00002-qz6] has been deployed and is serving 100 percent of traffic.
			Service URL: https://ai-assistant-service-507154614599.us-central1.run.app
			
		  gcloud run deploy ai-assistant-service --region us-central1 --min-instances 1 --platform managed      
		  
      gcloud run services delete ai-assistant-service --region us-central1 --platform managed
```

### Cloud run Re-Deploy with new image tags
```
  docker build --no-cache -t venkat5ai/ai-assistant:1.1 .
  docker push venkat5ai/ai-assistant:1.1
  docker tag venkat5ai/ai-assistant:1.1 us-central1-docker.pkg.dev/api-project-507154614599/ai-assistant-repo/ai-assistant:1.1
	docker push us-central1-docker.pkg.dev/api-project-507154614599/ai-assistant-repo/ai-assistant:1.1
	gcloud run deploy ai-assistant-service --image us-central1-docker.pkg.dev/api-project-507154614599/ai-assistant-repo/ai-assistant:1.1 --region us-central1 --platform managed
```
---
## API Endpoints

- `POST /api/bot` — Ask questions (JSON: `{ "query": "your question", "session_id": "optional" }`)
- `POST /api/documents/upload` — Upload a document (multipart/form-data, with `file` and optional `session_id`)
- `GET /health` — Health check

---

## Requirements
- Docker
- Google Cloud Project with Vertex AI, Matching Engine, and GCS enabled
- Service account with appropriate permissions

---

## Development
- Python 3.10+
- See `requirements.txt` for dependencies
- To run locally (not recommended for production):
  ```powershell
  pip install -r requirements.txt
  python app.py cloud
  ```

---

## Deployment steps in order
- Prepare RAG CORPUS and upload data docs to RAG Engine (RAG_CORPUS)
```
  python utils/prepare_corpus_and_data.py
```
- Deploy the AGENT on to Agent Engine (AGENT_ENGINE_ID)
```
  python deployment/deploy.py
```
- Run the app
```
  python app.py
```
---

## Cloud Run URLs
User  - https://waterlynn-community-service-507154614599.us-central1.run.app

Admin - https://waterlynn-community-admin-507154614599.us-central1.run.app

---

## Acknowledgements
- [LangChain](https://github.com/langchain-ai/langchain)
- [Google Vertex AI](https://cloud.google.com/vertex-ai)
- [Unstructured](https://github.com/Unstructured-IO/unstructured)


## References
- Google ADK Sample examples repo - https://github.com/google/adk-samples/tree/main/python
- OpenAPI tools - https://google.github.io/adk-docs/tools/openapi-tools/#example

- Chat: https://gemini.google.com/app/f898d6caf5431a07
        https://gemini.google.com/app/315936c357c69470
        https://gemini.google.com/app/baba4e2f7b8ffb27

- 

## License
MIT License © 2025 venkat5ai