import asyncio
import os
import json
from typing import Optional, List
from pydantic import BaseModel, Field

from crawl4ai import (
    AsyncWebCrawler,
    BrowserConfig,
    CrawlerRunConfig,
    CacheMode,
    LLMConfig,
)
from crawl4ai.deep_crawling import (
    BFSDeePCrawlStrategy,
    DomainFilter,
    URLPatternFilter,
    FilterChain,
    RateLimiter # Import RateLimiter
)
from crawl4ai.extraction_strategy import LLMExtractionStrategy

# Define the Pydantic Model for a Car Listing
# The LLM will try to map the extracted information to these fields.
# We make most fields Optional, as not all information might be present for every listing.
class CarListing(BaseModel):
    year: Optional[int] = Field(None, description="The manufacturing year of the vehicle.")
    make: Optional[str] = Field(None, description="The make of the vehicle (e.g., 'Toyota', 'Ford').")
    model: Optional[str] = Field(None, description="The model of the vehicle (e.g., 'Camry', 'F-150').")
    trim: Optional[str] = Field(None, description="The specific trim level of the vehicle (e.g., 'Limited', 'XLT').")
    price: Optional[float] = Field(None, description="The selling price of the vehicle, as a numerical value.")
    mileage: Optional[int] = Field(None, description="The mileage of the vehicle, as an integer.")
    exterior_color: Optional[str] = Field(None, description="The exterior color of the vehicle.")
    vin: Optional[str] = Field(None, description="The Vehicle Identification Number (VIN).")
    engine: Optional[str] = Field(None, description="Engine specifications (e.g., 'V6 3.5L').")
    transmission: Optional[str] = Field(None, description="Transmission type (e.g., 'Automatic', 'Manual').")
    body_style: Optional[str] = Field(None, description="The body style of the vehicle (e.g., 'Sedan', 'SUV', 'Truck').")
    stock_number: Optional[str] = Field(None, description="The dealership's internal stock number for the vehicle.")
    features: Optional[List[str]] = Field(None, description="A list of key features or options of the vehicle.")
    dealer_comments: Optional[str] = Field(None, description="Any specific comments or description provided by the dealership.")
    eligible_benefits: Optional[List[str]] = Field(None, description="A list of eligible benefits or programs associated with the vehicle.")

# Main asynchronous function to perform the crawl
async def crawl_car_dealership():
    # --- 1. Configure LLM for Extraction (Using Google Gemini) ---
    llm_config = LLMConfig(
        provider="google/gemini-2.0-flash-001",  # Specify the Google Gemini model
        api_token="",                            # No explicit API token needed for Canvas environment
        base_url="https://generativelanguage.googleapis.com/v1beta/models/", # Base URL for Google Gemini API
        temperature=0.1,                         # Lower temperature for more deterministic, factual extraction
        max_tokens=1000                          # Limit response length to control cost and focus
    )

    # Define the LLM extraction strategy
    llm_extraction_strategy = LLMExtractionStrategy(
        llm_config=llm_config,
        schema=CarListing.model_json_schema(), # Use the Pydantic model's JSON schema
        instruction="""
        Extract all available details for a single vehicle from the provided webpage content.
        Map the information to the specified schema, inferring types where necessary.
        Pay close attention to the year, make, model, trim, price, mileage, exterior color,
        VIN, engine, transmission, body style, stock number, features, dealer comments, and eligible benefits.
        If a piece of information is not explicitly found on the page, return null for that field.
        """,
        input_format="markdown", # Instruct LLM to process the page's Markdown content
        chunk_token_threshold=1500 # Adjust if pages are very long
    )

    # --- 2. Configure Deep Crawling ---
    base_domain = "www.randymarion.com"
    start_url = f"https://{base_domain}/searchall.aspx" # Corrected start URL
    
    # Filter to only allow crawling within randymarion.com
    domain_filter = DomainFilter(allowed_domains=[base_domain])

    # URL pattern filter to target only individual vehicle detail pages
    vehicle_details_pattern_filter = URLPatternFilter(
        patterns=[r"https://www\.randymarion\.com/VehicleDetails\.aspx\?ID=\d+"]
    )

    # Combine filters into a FilterChain
    filter_chain = FilterChain(filters=[
        domain_filter,
        vehicle_details_pattern_filter
    ])

    # Configure the BFS (Breadth-First Search) deep crawl strategy
    bfs_strategy = BFSDeePCrawlStrategy(
        max_depth=1,         # Crawl initial page, then one level deep (to detail pages)
        max_pages=20,        # Limit to 20 total pages (for initial testing to save API calls)
        filter_chain=filter_chain, # Apply our defined filters
        include_external=False, # Do not follow links outside the allowed domain
        # --- ADDED: RateLimiter for responsible crawling ---
        rate_limiter=RateLimiter(
            mean_delay=2.0,  # Average delay of 2 seconds between requests
            max_range=1.0    # Randomize delay between 1.0 and 3.0 seconds
        )
    )

    # --- 3. Configure Crawler Run ---
    run_config = CrawlerRunConfig(
        deep_crawl_strategy=bfs_strategy,       # Enable deep crawling with BFS
        extraction_strategy=llm_extraction_strategy, # Apply LLM extraction to each crawled page
        cache_mode=CacheMode.BYPASS,            # Always fetch fresh content for development
        word_count_threshold=50,                # Ignore pages with very little content
        remove_overlay_elements=True,           # Attempt to remove pop-ups/overlays
        # verbose=True # Uncomment for more detailed logging from crawl4ai
    )

    # --- 4. Initialize and Run AsyncWebCrawler ---
    browser_config = BrowserConfig(
        headless=True,
        user_agent_mode="random" # Use a random user agent for stealth
    )

    print(f"Starting deep crawl of {start_url} for car listings using Google Gemini...")
    print(f"Will extract up to {bfs_strategy.max_pages} vehicle detail pages.")
    print(f"Extracted car details will be saved to 'car-inventory.txt'.")

    output_filename = "car-inventory.txt"
    successful_extractions = 0

    # Use AsyncWebCrawler as a context manager for automatic resource cleanup
    async with AsyncWebCrawler(config=browser_config) as crawler:
        # Open the output file in write mode
        with open(output_filename, "w", encoding="utf-8") as outfile:
            outfile.write("--- Randy Marion Car Inventory ---\n\n")

            results_container = await crawler.arun(url=start_url, config=run_config)

            for i, result in enumerate(results_container):
                if result.success and result.extracted_content:
                    try:
                        car_data = json.loads(result.extracted_content)
                        
                        if isinstance(car_data, list) and car_data:
                            car_data_dict = car_data[0]
                        elif isinstance(car_data, dict):
                            car_data_dict = car_data
                        else:
                            outfile.write(f"\n--- Warning: Unexpected LLM output format for {result.url} ---\n")
                            outfile.write(f"Raw extracted_content: {result.extracted_content}\n")
                            continue

                        car_listing_obj = CarListing(**car_data_dict)
                        successful_extractions += 1

                        # Construct the banner for the car
                        banner_make = car_listing_obj.make or "N/A"
                        banner_model = car_listing_obj.model or "N/A"
                        banner_year = car_listing_obj.year or "N/A"
                        banner = f"--- Car {successful_extractions}: {banner_year} {banner_make} {banner_model} ---\n"
                        outfile.write(banner)
                        outfile.write(f"Source URL: {result.url}\n")
                        
                        # Write details to the file
                        outfile.write(f"  Year: {car_listing_obj.year or 'N/A'}\n")
                        outfile.write(f"  Make: {car_listing_obj.make or 'N/A'}\n")
                        outfile.write(f"  Model: {car_listing_obj.model or 'N/A'}\n")
                        outfile.write(f"  Trim: {car_listing_obj.trim or 'N/A'}\n")
                        outfile.write(f"  Price: ${car_listing_obj.price:,.2f}\n" if car_listing_obj.price is not None else "  Price: N/A\n")
                        outfile.write(f"  Mileage: {car_listing_obj.mileage:,} miles\n" if car_listing_obj.mileage is not None else "  Mileage: N/A\n")
                        outfile.write(f"  Exterior Color: {car_listing_obj.exterior_color or 'N/A'}\n")
                        outfile.write(f"  VIN: {car_listing_obj.vin or 'N/A'}\n")
                        outfile.write(f"  Engine: {car_listing_obj.engine or 'N/A'}\n")
                        outfile.write(f"  Transmission: {car_listing_obj.transmission or 'N/A'}\n")
                        outfile.write(f"  Body Style: {car_listing_obj.body_style or 'N/A'}\n")
                        outfile.write(f"  Stock Number: {car_listing_obj.stock_number or 'N/A'}\n")
                        
                        if car_listing_obj.features:
                            outfile.write(f"  Features: {', '.join(car_listing_obj.features)}\n")
                        else:
                            outfile.write("  Features: N/A\n")

                        if car_listing_obj.eligible_benefits:
                            outfile.write(f"  Eligible Benefits: {', '.join(car_listing_obj.eligible_benefits)}\n")
                        else:
                            outfile.write("  Eligible Benefits: N/A\n")

                        if car_listing_obj.dealer_comments:
                            outfile.write(f"  Dealer Comments: {car_listing_obj.dealer_comments}\n")
                        else:
                            outfile.write("  Dealer Comments: N/A\n")
                        outfile.write("\n") # Add a blank line for separation

                    except json.JSONDecodeError as e:
                        outfile.write(f"\n--- Error decoding JSON from LLM for {result.url} ---\n")
                        outfile.write(f"Error: {e}\n")
                        outfile.write(f"Raw extracted_content: {result.extracted_content}\n\n")
                    except Exception as e:
                        outfile.write(f"\n--- Error processing Pydantic model for {result.url} ---\n")
                        outfile.write(f"Error: {e}\n")
                        outfile.write(f"Content from LLM: {result.extracted_content}\n\n")
                else:
                    if not result.success:
                        outfile.write(f"\n--- Crawl Failed for {result.url} ---\n")
                        outfile.write(f"  Error: {result.error_message}\n\n")
                    elif not result.extracted_content:
                        outfile.write(f"\n--- No Content Extracted for {result.url} ---\n")
                        outfile.write(f"  Page may not be a car detail page or LLM couldn't extract.\n\n")

        print(f"\n--- Crawl Summary ---")
        print(f"Total pages attempted (including start page): {len(results_container)}")
        print(f"Total successful car extractions: {successful_extractions}")
        print(f"All extracted details saved to '{output_filename}'")

# Run the asynchronous main function
if __name__ == "__main__":
    asyncio.run(crawl_car_dealership())
