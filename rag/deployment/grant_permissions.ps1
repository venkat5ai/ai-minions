# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     [http://www.apache.org/licenses/LICENSE-2.0](http://www.apache.org/licenses/LICENSE-2.0)
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Script to grant RAG Corpus access permissions to AI Platform Reasoning Engine Service Agent (PowerShell version)

# --- Configuration Variables (Hardcoded for your project) ---
$PROJECT_ID = "api-project-507154614599"
$RAG_CORPUS_FULL_PATH = "projects/507154614599/locations/us-central1/ragCorpora/4611686018427387904"

# --- Retrieve Project Number ---
Write-Host "Retrieving project number for project $($PROJECT_ID)..."
try {
    $PROJECT_NUMBER = (gcloud projects describe $PROJECT_ID --format="value(projectNumber)")
    if ([string]::IsNullOrWhiteSpace($PROJECT_NUMBER)) {
        throw "Failed to retrieve project number for project $($PROJECT_ID)."
    }
    Write-Host "Project number: $($PROJECT_NUMBER)"
} catch {
    Write-Error "Error: $($_.Exception.Message)"
    exit 1
}

# --- Define the Service Account for the Deployed Agent Engine ---
# This is the default service account for the AI Platform Reasoning Engine in your project.
$SERVICE_ACCOUNT = "service-$($PROJECT_NUMBER)@gcp-sa-aiplatform-re.iam.gserviceaccount.com"
Write-Host "Target Service Account: $($SERVICE_ACCOUNT)"

# --- Extract RAG Corpus ID ---
# Example: "projects/123/locations/us-central1/ragCorpora/456" -> "456"
$RAG_CORPUS_ID = $RAG_CORPUS_FULL_PATH.Split('/')[-1]
$RAG_CORPUS_RESOURCE = "projects/$($PROJECT_NUMBER)/locations/us-central1/ragCorpora/$($RAG_CORPUS_ID)"

Write-Host "Granting permissions to $($SERVICE_ACCOUNT)..."

# --- Ensure the AI Platform service identity exists ---
Write-Host "Ensuring AI Platform service identity exists..."
try {
    gcloud alpha services identity create --service=aiplatform.googleapis.com --project=$PROJECT_ID
    Write-Host "AI Platform service identity ensured."
} catch {
    Write-Error "Error ensuring AI Platform service identity: $($_.Exception.Message)"
    Write-Error "This step might fail if the identity already exists, which is often okay. Continuing..."
    # Do not exit, as this command might fail if the identity already exists.
}

# --- Create a custom role with only the RAG Corpus query permission ---
$ROLE_ID = "ragCorpusQueryRole"
$ROLE_TITLE = "RAG Corpus Query Role"
$ROLE_DESCRIPTION = "Custom role with permission to query RAG Corpus"

Write-Host "Checking if custom role $($ROLE_ID) exists..."
try {
    gcloud iam roles describe $ROLE_ID --project=$PROJECT_ID -ErrorAction Stop | Out-Null
    Write-Host "Custom role $($ROLE_ID) already exists."
} catch {
    Write-Host "Custom role $($ROLE_ID) does not exist. Creating it..."
    try {
        gcloud iam roles create $ROLE_ID `
            --project=$PROJECT_ID `
            --title="$ROLE_TITLE" `
            --description="$ROLE_DESCRIPTION" `
            --permissions="aiplatform.ragCorpora.query" -ErrorAction Stop
        Write-Host "Custom role $($ROLE_ID) created successfully."
    } catch {
        Write-Error "Error creating custom role $($ROLE_ID): $($_.Exception.Message)"
        exit 1
    }
}

# --- Grant the custom role to the service account ---
Write-Host "Granting custom role for RAG Corpus query permissions for $($RAG_CORPUS_RESOURCE)..."
try {
    gcloud projects add-iam-policy-binding $PROJECT_ID `
        --member="serviceAccount:$($SERVICE_ACCOUNT)" `
        --role="projects/$($PROJECT_ID)/roles/$($ROLE_ID)" -ErrorAction Stop
    Write-Host "Permissions granted successfully."
    Write-Host "Service account $($SERVICE_ACCOUNT) can now query the specific RAG Corpus: $($RAG_CORPUS_RESOURCE)"
} catch {
    Write-Error "Error granting permissions: $($_.Exception.Message)"
    exit 1
}

Write-Host "Script finished."
