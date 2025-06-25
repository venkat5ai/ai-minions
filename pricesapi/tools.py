# prices_comparison_agent_folder/tools.py
import os
import json
import yaml
from dotenv import load_dotenv # Ensure this is at the top if not already

from google.adk.tools.openapi_tool.openapi_spec_parser.openapi_toolset import OpenAPIToolset
from google.adk.tools.openapi_tool.auth.auth_helpers import token_to_scheme_credential

load_dotenv() # Load environment variables from .env file

# Get the directory of the current file
current_dir = os.path.dirname(__file__)

# --- Prices Comparison API Tools ---
PRICES_COMPARISON_SPEC_FILE = os.path.join(current_dir, "prices_comparison_spec.yaml")

with open(PRICES_COMPARISON_SPEC_FILE, 'r') as f:
    prices_comparison_spec = yaml.safe_load(f)

# Get the API key from environment variables for security
# It's highly recommended to use a Secret Manager in production.
PRICES_COMPARISON_API_KEY = os.getenv("PRICES_COMPARISON_API_KEY", "YOUR_API_KEY_HERE")
if PRICES_COMPARISON_API_KEY == "YOUR_API_KEY_HERE":
    print("WARNING: PRICES_COMPARISON_API_KEY not set in environment. Please set it or replace 'YOUR_API_KEY_HERE'.")

# Define authentication scheme for the Prices Comparison API
# The API key needs to be in the 'Authorization' header with 'Bearer' scheme
auth_scheme_prices = "bearer"
auth_credential_prices = token_to_scheme_credential("bearer", "header", "Authorization", PRICES_COMPARISON_API_KEY)


prices_comparison_apis = OpenAPIToolset(
    spec_str=json.dumps(prices_comparison_spec),
    spec_str_type="json",
    auth_scheme=auth_scheme_prices,
    auth_credential=auth_credential_prices
)