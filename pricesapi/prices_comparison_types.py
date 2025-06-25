# prices_comparison_agent_folder/prices_comparison_types.py
from typing import List, Optional
from pydantic import BaseModel, Field

class ProductListing(BaseModel):
    """Represents a product listing from the Prices Comparison API."""
    title: str = Field(description="The title of the product listing.")
    price: str = Field(description="The price of the product, including currency and any qualifiers (e.g., 'refurbished').")
    shop: str = Field(description="The name of the shop selling the product.")
    shipping: Optional[str] = Field(None, description="Shipping information (e.g., 'Free shipping').")
    rating: Optional[str] = Field(None, description="Product rating (e.g., '4.5 out of 5 stars').")
    reviews: Optional[str] = Field(None, description="Number of reviews (e.g., '(76,009)').")
    link: str = Field(description="URL to the product page.")
    img: Optional[str] = Field(None, description="URL to the product image.")

class PriceComparisonResponse(BaseModel):
    """Represents the full response from the Prices Comparison API, which is an array of ProductListing objects."""
    __root__: List[ProductListing]