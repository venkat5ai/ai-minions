# agent.py
import os

from google.adk.agents import Agent
from google.adk.tools.agent_tool import AgentTool

# Import only the OpenAPI API toolset, as the DataAnalysisAgent will use these directly
from .tools import jsonplaceholder_apis

LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "gemini-2.0-flash-001")

# --- User Management Sub-agent ---
user_management_agent = Agent(
    name="user_management_agent",
    model=LLM_MODEL_NAME,
    description="Specialized agent for managing JSONPlaceholder users.",
    instruction="""You are a helpful assistant specifically for managing users on JSONPlaceholder.
    You can list all users, get a user by their ID, create new users, update existing users, and delete users.

    When retrieving data:
    - If retrieving details for a *single user* (e.g., 'get user by ID'), extract the most relevant information (like name, email, address, phone, website, company name) and present it in a clear, conversational summary. Do NOT return the raw JSON unless the user explicitly asks for 'raw JSON' or 'full details'.
    - If listing *multiple users* (e.g., 'list all users'), provide a brief summary of the users found (e.g., 'Here are the users I found: [User 1 Name], [User 2 Name], ...' or 'I found X users.'). Do NOT return the full JSON list unless the user explicitly asks for 'raw JSON' or 'full details.
    - For creating, updating, or deleting, confirm the action and any changes made.
    """
)

# --- Post Management Sub-agent ---
post_management_agent = Agent(
    name="post_management_agent",
    model=LLM_MODEL_NAME,
    description="Specialized agent for managing JSONPlaceholder posts.",
    instruction="""You are a helpful assistant specifically for managing posts on JSONPlaceholder.
    You can list all posts, get a post by its ID, create new posts, update existing posts, and delete posts.
    You can also get comments for a specific post by its ID.

    When retrieving data:
    - If retrieving details for a *single post* (e.g., 'get post by ID'), extract the title and body and present it. If retrieving comments for a post, list the comments briefly.
    - If listing *multiple posts* (e.g., 'list all posts'), provide a brief summary of the posts found.
    - For creating, updating, or deleting, confirm the action and any changes made.
    """
)

# --- Comment Management Sub-agent ---
comment_management_agent = Agent(
    name="comment_management_agent",
    model=LLM_MODEL_NAME,
    description="Specialized agent for managing JSONPlaceholder comments.",
    instruction="""You are a helpful assistant specifically for managing comments on JSONPlaceholder.
    You can list all comments, get a comment by its ID, create new comments, update existing comments, and delete comments.

    When retrieving data:
    - If retrieving details for a *single comment* (e.g., 'get comment by ID'), extract the relevant details and present it.
    - If listing *multiple comments* (e.g., 'list all comments'), provide a brief summary of the comments found.
    - For creating, updating, or deleting, confirm the action and any changes made.
    """
)


# The main agent
root_agent = Agent(
    name="root_agent",
    model=LLM_MODEL_NAME,
    instruction="""You are a helpful assistant for interacting with the JSONPlaceholder API.
    Your main goal is to route user requests to the most appropriate sub-agent based on their intent using the `transfer_to_sub_agent` tool.

    **Key Delegation Rules:**
    - For any request related to JSONPlaceholder users (e.g., list users, create a user, get user details, update user, delete user), you should use the `transfer_to_sub_agent` tool and specify `user_management_agent`.
    - For any request related to JSONPlaceholder posts (e.g., list posts, create a post, get post details, get comments *for a specific post* by post ID), use the `transfer_to_sub_agent` tool and specify `post_management_agent`.
    - For any request related to JSONPlaceholder comments (e.g., list all comments, create a comment, get a comment by its ID, update, delete; but *NOT* comments linked to users or posts indirectly), use the `transfer_to_sub_agent` tool and specify `comment_management_agent`.

    **IMPORTANT:** When delegating, do NOT explicitly mention transferring control to a sub-agent. The `transfer_to_sub_agent` tool itself handles the transfer silently. The delegated sub-agent will then generate the final user-facing response. If a sub-agent cannot fulfill the request or needs more information, it will indicate that, and you should try to re-route or ask clarifying questions to the user in a helpful manner.
    """,
    sub_agents=[user_management_agent, post_management_agent, comment_management_agent],
    tools=[
        jsonplaceholder_apis, # Tools for direct API calls (e.g., root could theoretically call them)
    ],
)