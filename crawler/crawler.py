import asyncio
import os
import json
import re
from typing import Optional, List
from pydantic import BaseModel, Field

from crawl4ai import (
    AsyncWebCrawler,
    BrowserConfig,
    CrawlerRunConfig,
    CacheMode,
    LLMConfig,
    RateLimiter,
)
from crawl4ai.deep_crawling import (
    BFSDeepCrawlStrategy,
    URLPatternFilter,
    FilterChain,
)
from crawl4ai.extraction_strategy import LLMExtractionStrategy
from crawl4ai.content_scraping_strategy import LXMLWebScrapingStrategy


# Define the Pydantic Model for a Car Listing
class CarListing(BaseModel):
    year: Optional[int] = Field(None, description="The manufacturing year of the vehicle.")
    make: Optional[str] = Field(None, description="The make of the vehicle (e.g., 'Toyota', 'Ford').")
    model: Optional[str] = Field(None, description="The model of the vehicle (e.g., 'Camry', 'F-150').")
    trim: Optional[str] = Field(None, description="The specific trim level of the vehicle (e.g., 'Limited', 'XLT').")
    price: Optional[float] = Field(None, description="The selling price of the vehicle, as a numerical value.")
    mileage: Optional[int] = Field(None, description="The mileage of the vehicle, as an integer.")
    exterior_color: Optional[str] = Field(None, description="The exterior color of the vehicle.")
    interior_color: Optional[str] = Field(None, description="The interior color of the vehicle.")
    vin: Optional[str] = Field(None, description="The Vehicle Identification Number (VIN).")
    stock_number: Optional[str] = Field(None, description="The stock number of the vehicle at the dealership.")
    body_style: Optional[str] = Field(None, description="The body style of the vehicle (e.g., 'SUV', 'Sedan', 'Truck').")
    engine: Optional[str] = Field(None, description="The engine specifications of the vehicle.")
    transmission: Optional[str] = Field(None, description="The transmission type (e.g., 'Automatic', 'Manual').")
    drive_type: Optional[str] = Field(None, description="The drive type (e.g., 'FWD', 'RWD', 'AWD', '4x4').")
    fuel_type: Optional[str] = Field(None, description="The fuel type (e.g., 'Gasoline', 'Electric', 'Hybrid').")
    city_mpg: Optional[int] = Field(None, description="Estimated city miles per gallon.")
    highway_mpg: Optional[int] = Field(None, description="Estimated highway miles per gallon.")
    features: Optional[List[str]] = Field(None, description="A list of key features or options of the vehicle.")
    description: Optional[str] = Field(None, description="A brief description of the vehicle from the listing.")
    url: Optional[str] = Field(None, description="The URL of the car listing page.")


async def main():
    # --- Configuration ---
    start_url = "https://www.randymarion.com/searchall.aspx"
    
    # Updated Regex to identify individual vehicle details pages, now accounting for .com and .net subdomains
    vehicle_details_pattern = r"https://www\.randymarion\w*\.(com|net)/(new|used)-.*"
    vehicle_details_compiled_regex = re.compile(vehicle_details_pattern)
    
    max_pages_to_crawl = 200 # Limit the number of pages for this example
    output_filename = "extracted_cars.json"

    # Ensure GOOGLE_API_KEY environment variable is set for LLM access
    google_api_key = os.environ.get("GOOGLE_API_KEY")
    if not google_api_key:
        print("Error: GOOGLE_API_KEY environment variable not set. LLM features may not work.")
        return

    # LLM Configuration for content extraction
    llm_config = LLMConfig(
        provider="gemini/gemini-2.0-flash-001", 
        api_token=google_api_key,
    )

    # Define the LLM extraction strategy using the Pydantic model
    extraction_strategy = LLMExtractionStrategy(
        schema=CarListing.model_json_schema(),
        llm_config=llm_config,
        instruction=f"""
        You are an expert at extracting vehicle information from car dealership websites.
        Extract all available details for a single car listing based on the provided schema.
        Focus only on the details of the specific car on the current page.
        If a field is not explicitly present on the page, return null for that field.
        Do not make up any information.
        The price should be a numerical value without currency symbols or commas.
        Mileage should be an integer.
        The URL should be the canonical URL of the listing page.
        """,
        extra_args={"temperature": 0.1, "max_tokens": 1000},
    )

    # Define URL filters for deep crawling
    # Updated to allow crawling on all randymarion*.com and randymarion*.net subdomains
    allowed_patterns = [
        r"https://www\.randymarion\w*\.(com|net)/.*" # Allows randymarion.com, randymarionford.com, randymarionhonda.com, etc., and their .net counterparts.
    ]
    # Denies common non-HTML file types and specific content pages
    denied_file_types_pattern = r".*\.(pdf|jpg|png|gif|zip|css|js|xml|ico|txt|mp4|webp|svg|woff|woff2|ttf|eot|json|csv|rtf|xls|xlsx|doc|docx|ppt|pptx|gz|rar|7z)$"
    denied_content_pages_patterns = [
        r".*sitemap\.aspx.*", # Deny sitemap and related paths
        r".*/join-the-team.*", # Deny career/join the team pages
        r".*/about-randy-marion\.html", # Deny about us page
        r".*/privacy-policy\.html" # Deny privacy policy page
    ]

    # Create URLPatternFilter instances
    allowed_filter = URLPatternFilter(patterns=allowed_patterns, reverse=False)
    denied_file_types_filter = URLPatternFilter(patterns=[denied_file_types_pattern], reverse=True) # reverse=True means deny if pattern matches
    denied_content_filter = URLPatternFilter(patterns=denied_content_pages_patterns, reverse=True)

    # Chain the filters - re-introduced for filtering control
    filter_chain = FilterChain(filters=[allowed_filter, denied_file_types_filter, denied_content_filter])

    # Deep Crawl Strategy (Breadth-First Search)
    deep_crawl_config = BFSDeepCrawlStrategy(
        max_depth=5, # How deep to crawl from the start URL
        max_pages=max_pages_to_crawl, # Maximum total pages to crawl
        filter_chain=filter_chain, # Re-instated filter_chain
    )

    # Browser Configuration
    browser_config = BrowserConfig(
        headless=True, # Run browser in headless mode (no visible UI)
        user_agent_mode="random", # Use a random user agent for each request
        verbose=True, # Enable verbose logging from the browser
    )

    # Rate Limiter to be polite to the website
    rate_limiter = RateLimiter(base_delay=(3.0, 7.0), max_delay=60.0, max_retries=3)

    # Crawler Run Configuration
    run_config = CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS, # Bypass cache for fresh content during development/testing
        extraction_strategy=extraction_strategy, # Apply the LLM extraction strategy
        deep_crawl_strategy=deep_crawl_config, # Apply the deep crawling strategy
        scraping_strategy=LXMLWebScrapingStrategy(), # Use LXML for robust HTML cleaning before LLM
    )

    results_container = []
    successful_extractions = 0

    print(f"Starting crawl of {start_url} for car listings...")

    # Initialize the AsyncWebCrawler and run the crawl
    async with AsyncWebCrawler(config=browser_config, rate_limiter=rate_limiter) as crawler:
        results = await crawler.arun_many(urls=[start_url], config=run_config)
        for result in results:
            print(f"\n--- Processing {result.url} ---")
            if result.success:
                if result.extracted_content and vehicle_details_compiled_regex.match(result.url):
                    try:
                        car_listing_data = json.loads(result.extracted_content)
                        car_listing = CarListing(**car_listing_data)
                        results_container.append(car_listing.model_dump(exclude_unset=True))
                        successful_extractions += 1
                        print(f"✅ Successfully extracted car details from: {result.url}")
                    except Exception as e:
                        print(f"❌ Pydantic processing error for {result.url}: {e}")
                        print(f"   Content from LLM: {result.extracted_content}")
                elif not result.extracted_content and vehicle_details_compiled_regex.match(result.url):
                    print(f"⚠️ No content extracted from vehicle details page: {result.url}")
                    print(f"   LLM might not have found relevant data or page was empty or format was unexpected.")
                else:
                    print(f"ℹ️ Successfully crawled non-detail page: {result.url}")
            else:
                # This is where the 'list' object has no attribute 'status_code' error appears to be caught and reported by crawl4ai.
                print(f"❌ Crawl failed for {result.url}: {result.error_message}")

    # Write all extracted data to a JSON file
    with open(output_filename, 'w', encoding='utf-8') as outfile:
        json.dump(results_container, outfile, indent=4, ensure_ascii=False)

    print(f"\n--- Crawl Summary ---")
    print(f"Total pages attempted (including start page): {len(results_container) + (0 if successful_extractions == len(results_container) else (max_pages_to_crawl - successful_extractions))}") # Approximation
    print(f"Total successful car extractions: {successful_extractions}")
    print(f"All extracted details saved to '{output_filename}'")

# Run the main asynchronous function
if __name__ == "__main__":
    asyncio.run(main())