# tools.py
import os
import json
import yaml # Import yaml

from google.adk.tools.openapi_tool.openapi_spec_parser.openapi_toolset import OpenAPIToolset
from google.adk.tools.openapi_tool.auth.auth_helpers import token_to_scheme_credential


# Get the directory of the current file
current_dir = os.path.dirname(__file__)
# Construct the path to the OpenAPI spec file
# Ensure this matches your file name (e.g., jsonplaceholder_spec.yaml)
OPENAPI_SPEC_FILE = os.path.join(current_dir, "jsonplaceholder_spec.yaml")

# Load the OpenAPI specification from the file
# Use yaml.safe_load for .yaml files
with open(OPENAPI_SPEC_FILE, 'r') as f:
    jsonplaceholder_spec = yaml.safe_load(f)

# Common authentication (none for JSONPlaceholder)
auth_scheme = None
auth_credential = None

# --- JSONPlaceholder OpenAPI API Toolset ---
# This toolset exposes the API operations defined in jsonplaceholder_spec.yaml
jsonplaceholder_apis = OpenAPIToolset(
    spec_str=json.dumps(jsonplaceholder_spec), # Convert to JSON string for OpenAPIToolset
    spec_str_type="json", # The toolset expects JSON string, even if loaded from YAML
    auth_scheme=auth_scheme,
    auth_credential=auth_credential
)