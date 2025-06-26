# pricesapi/agent.py
import os

from google.adk.agents import Agent

# Import OpenAPI API toolset specifically for prices comparison
from .tools import prices_comparison_apis

LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "gemini-2.0-flash-001")

# --- Prices Comparison Agent ---
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
    
    - ***IMPORTANT FORMATTING RULE***: When you provide the link to the product, you MUST format it as a Markdown hyperlink. For example, instead of writing out the full URL, write it as: [link](https://www.example.com/product-url)

    - If no results are found, inform the user that you couldn't find any listings for the given product.
    """,
    tools=[prices_comparison_apis],
    sub_agents=[],
)