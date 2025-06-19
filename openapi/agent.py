# agent.py
from google.adk.agents import Agent
# from google.adk.tools import AgentTool
from vertexai.preview.reasoning_engines.tool import AgentTool # Corrected import for AgentTool

# Import the API toolsets
from .tools import users_api, posts_api, comments_api

# --- User Management Sub-agent ---
user_management_agent = Agent(
    name="user_management_agent",
    model="gemini-1.5-flash",
    description="Specialized agent for managing JSONPlaceholder users.",
    instruction="""You are a helpful assistant specifically for managing users on JSONPlaceholder.
    You can list all users, get a user by their ID, create new users, update existing users, and delete users.
    Do not greet the user. Directly assist with user-related queries.
    If the user asks for actions not related to users, inform them that you can only help with user management and transfer them back to the main agent.
    Always provide clear and concise responses based on the tool outputs.""",
    tools=[*users_api.get_tools()] # Use the get_tools() from the specific toolset
)

# --- Post Management Sub-agent ---
post_management_agent = Agent(
    name="post_management_agent",
    model="gemini-1.5-flash",
    description="Specialized agent for managing JSONPlaceholder posts and comments related to posts.",
    instruction="""You are a helpful assistant specifically for managing posts and their comments on JSONPlaceholder.
    You can list all posts, get a post by its ID, create new posts, and list comments for a specific post.
    Do not greet the user. Directly assist with post-related queries.
    If the user asks for actions not related to posts or getting comments for posts, inform them that you can only help with post management and transfer them back to the main agent.
    Always provide clear and concise responses based on the tool outputs.""",
    tools=[*posts_api.get_tools()] # Use the get_tools() from the specific toolset
)

# --- Comment Management Sub-agent (for top-level comment operations) ---
comment_management_agent = Agent(
    name="comment_management_agent",
    model="gemini-1.5-flash",
    description="Specialized agent for managing JSONPlaceholder comments directly.",
    instruction="""You are a helpful assistant specifically for managing comments on JSONPlaceholder.
    You can list all comments, create new comments, and get a comment by its ID.
    Do not greet the user. Directly assist with comment-related queries.
    If the user asks for actions not related to comments, inform them that you can only help with comment management and transfer them back to the main agent.
    Always provide clear and concise responses based on the tool outputs.""",
    tools=[*comments_api.get_tools()] # Use the get_tools() from the specific toolset
)

# --- The main orchestrator agent ---
root_agent = Agent(
    name="JsonPlaceholderOrchestrator",
    global_instruction="""You are a helpful virtual assistant that can manage data on JSONPlaceholder.
    Your goal is to understand the user's request and delegate to the most appropriate specialized sub-agent (users, posts, or comments).""",
    instruction="""You are the main assistant for JSONPlaceholder data.
    Welcome the user and ask how you can help them with user data, post data, or comment data.
    - If the user asks about 'users' (e.g., list users, create a user, get user details, update user, delete user), transfer to the `user_management_agent`.
    - If the user asks about 'posts' (e.g., list posts, create a post, get post details, get comments for a post), transfer to the `post_management_agent`.
    - If the user asks about 'comments' (e.g., list comments, create a comment, get comment details, NOT comments for a specific post), transfer to the `comment_management_agent`.
    - After a sub-agent completes its task, ask if there's anything else you can help with.
    - If the user doesn't need anything else, politely thank them for using the JSONPlaceholder assistant.""",
    sub_agents=[user_management_agent, post_management_agent, comment_management_agent],
    # The root agent uses AgentTool to call the sub-agents.
    tools=[
        AgentTool(
            name="manageUsers",
            agent=user_management_agent,
            description="Tool to manage user-related operations on JSONPlaceholder (list, get by ID, create, update, delete users)."
        ),
        AgentTool(
            name="managePosts",
            agent=post_management_agent,
            description="Tool to manage post-related operations on JSONPlaceholder (list, get by ID, create posts, list comments for a specific post)."
        ),
        AgentTool(
            name="manageComments",
            agent=comment_management_agent,
            description="Tool to manage top-level comment-related operations on JSONPlaceholder (list, get by ID, create comments)."
        ),
    ],
    model="gemini-1.5-pro", # A more capable model for the orchestrator
)