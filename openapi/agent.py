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
    When retrieving data (e.g., listing users, getting a specific user), **include the full and unedited raw JSON output from the tool calls directly in your response.** You may provide a brief introductory or concluding sentence in natural language.
    For creating, updating, or deleting users, confirm the action and any relevant details in natural language, without dumping full JSON.
    Do not greet the user. Directly assist with user-related queries.
    If the user asks for actions not related to users, inform them that you can only help with user management and transfer them back to the main agent.
    """,
    tools=[jsonplaceholder_apis] 
)

# --- Post Management Sub-agent ---
post_management_agent = Agent(
    name="post_management_agent",
    model=LLM_MODEL_NAME,
    description="Specialized agent for managing JSONPlaceholder posts and comments related to posts.",
    instruction="""You are a helpful assistant specifically for managing posts and their comments on JSONPlaceholder.
    You can list all posts, get a post by its ID, create new posts, update existing posts, and delete posts.
    You can also get comments for a specific post.
    When retrieving data (e.g., listing posts, getting a specific post or its comments), **include the full and unedited raw JSON output from the tool calls directly in your response.** You may provide a brief introductory or concluding sentence in natural language.
    For creating, updating, or deleting posts, confirm the action and any relevant details in natural language, without dumping full JSON.
    Do not greet the user. Directly assist with post-related queries.
    If the user asks for actions not related to posts or their comments, inform them that you can only help with post management and transfer them back to the main agent.
    """,
    tools=[jsonplaceholder_apis]
)

# --- Comment Management Sub-agent ---
comment_management_agent = Agent(
    name="comment_management_agent",
    model=LLM_MODEL_NAME,
    description="Specialized agent for managing JSONPlaceholder comments.",
    instruction="""You are a helpful assistant specifically for managing comments on JSONPlaceholder.
    You can list all comments, get a comment by its ID, create new comments, update existing comments, and delete comments.
    When retrieving data (e.g., listing comments, getting a specific comment), **include the full and unedited raw JSON output from the tool calls directly in your response.** You may provide a brief introductory or concluding sentence in natural language.
    For creating, updating, or deleting comments, confirm the action and any relevant details in natural language, without dumping full JSON.
    Do not greet the user. Directly assist with comment-related queries.
    If the user asks for actions not related to comments, inform them that you can only help with comment management and transfer them back to the main agent.
    """,
    tools=[jsonplaceholder_apis] 
)

# --- Data Analysis Agent ---
data_analysis_agent = Agent(
    name="data_analysis_agent",
    model=LLM_MODEL_NAME,
    description="Specialized agent for analyzing JSONPlaceholder data from users, posts, and comments.",
    instruction="""You are a highly capable assistant specialized in analyzing JSONPlaceholder data. Your primary goal is to answer analytical questions accurately.

    **To answer a complex analytical question (e.g., 'which user has the most comments', 'average posts per user'), you MUST follow these precise steps:**

    1.  **Identify all necessary data sources:** Determine which data is required (e.g., users, posts, comments).
    2.  **Retrieve ALL relevant raw data using your `jsonplaceholder_apis` tools.** For example:
        * To analyze posts, **immediately call `jsonplaceholder_apis.listPosts()`**.
        * To analyze comments, **immediately call `jsonplaceholder_apis.listComments()`**.
        * To link posts to users, **immediately call `jsonplaceholder_apis.listUsers()`**.
        * **CRITICAL: Do NOT explain that you *will* retrieve data; just retrieve it by executing the tool call.**
    3.  **Perform In-Memory Analysis:** Once you have the raw JSON output from all necessary tool calls, meticulously process and analyze this data **in your mind**. This involves tasks like:
        * Counting items (e.g., posts per user, comments per post).
        * Grouping data (e.g., comments by post ID, posts by user ID).
        * Joining or correlating data across different datasets (e.g., linking comments to posts, and posts to users to count comments per user).
        * Calculating averages, sums, min/max values.
        * Filtering data based on criteria.
    4.  **Formulate Concise Answer:** After completing your rigorous analysis, **provide your final answer in clear, concise natural language, summarizing your findings directly.**
        * **NEVER return raw JSON data as your final response for an analytical query.**
        * **NEVER just state that you *can* retrieve data or *have code* for it; you MUST actually retrieve and analyze it.**
        * **If, after attempting to retrieve and analyze, you genuinely cannot answer due to insurmountable limitations of your reasoning or the available data, explicitly state what you tried, why you failed, and offer to provide the raw data for the user to analyze themselves.**

    Directly assist with data analysis queries. If the user asks for actions not related to data analysis, inform them that you can only help with data analysis and transfer them back to the main agent.
    """,
    tools=[jsonplaceholder_apis] # This agent uses the existing OpenAPI tools to get raw data
)


# --- The main orchestrator agent ---
root_agent = Agent(
    name="JsonPlaceholderOrchestrator",
    global_instruction="""You are a helpful virtual assistant that can manage and analyze data on JSONPlaceholder.
    Your goal is to understand the user's request and delegate to the most appropriate specialized sub-agent (users, posts, comments, or data analysis).""", 
    instruction="""You are the main assistant for JSONPlaceholder data.
    Welcome the user and ask how you can help them with user data, post data, or comment data.
    - If the user asks about 'users' (e.g., list users, create a user, get user details, update user, delete user), transfer to the `user_management_agent`.
    - If the user asks about 'posts' (e.g., list posts, create a post, get post details, get comments for a post), transfer to the `post_management_agent`.
    - If the user asks about 'comments' (e.g., list comments, create a comment, get comment details, NOT comments for a specific post), transfer to the `comment_management_agent`.
    - **If the user asks a question requiring data analysis, aggregation, comparison, filtering across multiple data points, or summarization of data (e.g., "which user has the most posts", "how many comments does post X have", "average posts per user", "show posts by user X", "find users from city Y"), transfer to the `data_analysis_agent`.**
    - After a sub-agent completes its task, ask if there's anything else you can help with.
    - If the user doesn't need anything else, politely thank them for using the JSONPlaceholder assistant.""",
    sub_agents=[user_management_agent, post_management_agent, comment_management_agent, data_analysis_agent], 
    tools=[jsonplaceholder_apis], # Root only needs OpenAPI tools as the analysis agent will use them
    model=LLM_MODEL_NAME,
)