# jsonplaceholder_types.py
from typing import List, Optional
from pydantic import BaseModel, Field

# Nested models for User's address and company
class Geo(BaseModel):
    lat: str = Field(description="Latitude coordinate.")
    lng: str = Field(description="Longitude coordinate.")

class Address(BaseModel):
    street: str = Field(description="Street name.")
    suite: str = Field(description="Suite number.")
    city: str = Field(description="City name.")
    zipcode: str = Field(description="Zip code.")
    geo: Geo = Field(description="Geographical coordinates.")

class Company(BaseModel):
    name: str = Field(description="Company name.")
    catchPhrase: str = Field(description="Company's catchphrase.")
    bs: str = Field(description="Company's business strategy.")

class User(BaseModel):
    """Represents a JSONPlaceholder user."""
    id: int = Field(description="Unique identifier for the user.")
    name: str = Field(description="Full name of the user.")
    username: str = Field(description="Unique username.")
    email: str = Field(description="User's email address.")
    address: Address = Field(description="User's address details.")
    phone: str = Field(description="User's phone number.")
    website: str = Field(description="User's website URL.")
    company: Company = Field(description="User's company details.")

class Post(BaseModel):
    """Represents a JSONPlaceholder post."""
    userId: int = Field(description="The ID of the user who created the post.")
    id: int = Field(description="Unique identifier for the post.")
    title: str = Field(description="Title of the post.")
    body: str = Field(description="Content of the post.")

class Comment(BaseModel):
    """Represents a JSONPlaceholder comment."""
    postId: int = Field(description="The ID of the post the comment belongs to.")
    id: int = Field(description="Unique identifier for the comment.")
    name: str = Field(description="Name of the comment author.")
    email: str = Field(description="Email of the comment author.")
    body: str = Field(description="Content of the comment.")