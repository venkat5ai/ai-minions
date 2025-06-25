# prices_comparison_agent_folder/agent.py
import os

from google.adk.agents import Agent
from google.adk.tools.agent_tool import AgentTool # Often needed for sub-agent definitions, though not directly for tools here

# Import OpenAPI API toolset specifically for prices comparison
from .tools import prices_comparison_apis

# Import types for better LLM understanding of data structure
from .prices_comparison_types import ProductListing, PriceComparisonResponse

LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "gemini-2.0-flash-001")

# --- Prices Comparison Agent ---
# This agent handles all price comparison queries.
# It uses the prices_comparison_apis toolset.
prices_comparison_agent = Agent(
    name="prices_comparison_agent",
    model=LLM_MODEL_NAME,
    description="Specialized agent for comparing product prices from various online shops.",
    instruction="""You are a helpful assistant for comparing prices of products.
    You can search for a product (e.g., "iphone", "Samsung TV") and I will provide you with a list of available listings, their prices, and shops.
    When presenting the results:
    - Always state how many listings were found for the product.
    - For each listing, clearly state the 'title', 'price', and 'shop'.
    - If available, also mention 'shipping' information, 'rating', and 'reviews'.
    - Provide the 'link' to the product if the user asks for more details or to see the product.
    - If no results are found, inform the user that you couldn't find any listings for the given product.
    """,
    tools=[prices_comparison_apis], # Only add tools relevant to this agent
    sub_agents=[], # No sub-agents for this specific agent unless you decide to break down price comparison further
)

# This will be the entry point for the Prices Comparison functionalities.
# If you deploy this as a standalone agent, this `prices_comparison_agent` would be
# the "root" for its domain.
# For local testing and integration with app.py, you'd typically reference this agent.