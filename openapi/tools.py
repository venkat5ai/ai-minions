# tools.py
import os
import json
from google.adk.tools.openapi_tool.openapi_toolset import OpenApiToolset
from google.adk.tools.openapi_tool.auth.auth_helpers import NoAuthScheme, NoAuthCredential

# Get the directory of the current file
current_dir = os.path.dirname(__file__)
# Construct the path to the OpenAPI spec file
OPENAPI_SPEC_FILE = os.path.join(current_dir, "jsonplaceholder_spec.json")

# Load the OpenAPI specification from the file
with open(OPENAPI_SPEC_FILE, 'r') as f:
    jsonplaceholder_spec = json.load(f)

# Common authentication (none for JSONPlaceholder)
auth_scheme = NoAuthScheme()
auth_credential = NoAuthCredential()

# --- User API Toolset ---
# Filter operations that start with 'listUsers', 'getUser', 'createUser', 'updateUser', 'deleteUser'
users_api = OpenApiToolset(
    name="jsonplaceholder-users-api",
    description="API for managing JSONPlaceholder users, including listing, retrieving, creating, updating, and deleting users.",
    openapi_spec=jsonplaceholder_spec,
    auth_scheme=auth_scheme,
    auth_credential=auth_credential,
    include_operations=[
        "listUsers",
        "getUserById",
        "createUser",
        "updateUser",
        "deleteUser"
    ]
)

# --- Posts API Toolset ---
# Filter operations related to posts, including listing comments for a post
posts_api = OpenApiToolset(
    name="jsonplaceholder-posts-api",
    description="API for managing JSONPlaceholder posts, including listing, retrieving, creating, and listing comments for specific posts.",
    openapi_spec=jsonplaceholder_spec,
    auth_scheme=auth_scheme,
    auth_credential=auth_credential,
    include_operations=[
        "listPosts",
        "getPostById",
        "createPost",
        "listCommentsForPost" # This operation is associated with posts/{id}/comments
    ]
)

# --- Comments API Toolset ---
# Filter operations related to top-level comments
comments_api = OpenApiToolset(
    name="jsonplaceholder-comments-api",
    description="API for managing JSONPlaceholder comments, including listing, retrieving, and creating comments.",
    openapi_spec=jsonplaceholder_spec,
    auth_scheme=auth_scheme,
    auth_credential=auth_credential,
    include_operations=[
        "listComments",
        "getCommentById",
        "createComment"
    ]
)

# NOTE: We no longer need separate get_tools() functions, as each OpenApiToolset
# instance now directly represents a logical API grouping, and its .get_tools()
# method will return only the operations specified in 'include_operations'.