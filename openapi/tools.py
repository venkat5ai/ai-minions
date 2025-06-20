# tools.py
import os
import json
from google.adk.tools.openapi_tool.openapi_spec_parser.openapi_toolset import OpenAPIToolset
# from google.adk.tools.openapi_tool.auth.auth_helpers import NoAuthScheme, NoAuthCredential
from google.adk.tools.openapi_tool.auth.auth_helpers import token_to_scheme_credential


# Get the directory of the current file
current_dir = os.path.dirname(__file__)
# Construct the path to the OpenAPI spec file
OPENAPI_SPEC_FILE = os.path.join(current_dir, "jsonplaceholder_spec.json")

# Load the OpenAPI specification from the file
with open(OPENAPI_SPEC_FILE, 'r') as f:
    jsonplaceholder_spec = json.load(f)

# Common authentication (none for JSONPlaceholder)
auth_scheme = None # NoAuthScheme()
auth_credential = None # NoAuthCredential()

# --- User API Toolset ---
# Filter operations that start with 'listUsers', 'getUser', 'createUser', 'updateUser', 'deleteUser'
jsonplaceholder_apis = OpenAPIToolset(
    spec_str=json.dumps(jsonplaceholder_spec),
    spec_str_type="json",
    # tool_name="jsonplaceholder-users-api",
    auth_scheme=auth_scheme,
    auth_credential=auth_credential
)
