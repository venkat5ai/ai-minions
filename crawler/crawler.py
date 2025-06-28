import asyncio
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig, BrowserConfig, AsyncPlaywrightCrawlerStrategy

async def test_costco_crawl():
    # Configure the browser behavior.
    # Using a common user-agent to appear as a regular browser.
    browser_config = BrowserConfig(
        user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )

    # Configure the crawling run.
    # We use AsyncPlaywrightCrawlerStrategy because costco.com is a dynamic website.
    # check_robots_txt=True ensures we respect Costco's robots.txt directives.
    # mean_delay adds a pause between requests to be polite to the server.
    # wait_for can be used to ensure specific elements load, but for a simple test, we might skip it or use a general one.
    run_config = CrawlerRunConfig(
        crawler_strategy=AsyncPlaywrightCrawlerStrategy(),
        check_robots_txt=True,
        mean_delay=2, # Wait 2 seconds between requests
        max_range=1,  # Add a random variance of up to 1 second to the delay
        page_timeout=30000 # Wait up to 30 seconds for a page to load
        # For deeper crawls and specific data extraction, you would add:
        # deep_crawl_strategy=BFSDeePCrawlStrategy(max_depth=1, max_pages=10),
        # extraction_strategy=LLMExtractionStrategy(...)
    )

    # Initialize the crawler
    crawler = AsyncWebCrawler(browser_config=browser_config)

    # The URL we want to test.
    # For a real deep crawl, you'd have a list of starting category URLs.
    test_url = "https://www.costco.com/"

    print(f"Attempting to crawl: {test_url}")

    # Run the crawl for a single URL
    results = await crawler.arun(url=test_url, run_config=run_config)

    # Process the results
    if results.success:
        print(f"\nSuccessfully crawled: {results.url}")
        print(f"Page Title: {results.title}")
        # You can access the raw HTML or Markdown content if needed
        # print(f"First 500 chars of Markdown content:\n{results.markdown.raw_markdown[:500]}...")
        print(f"Number of links found: {len(results.links)}")
        if results.links:
            print("First 5 links found:")
            for i, link in enumerate(results.links[:5]):
                print(f"  - {link.text}: {link.url}")
    else:
        print(f"\nFailed to crawl {results.url}: {results.error_message}")
        print("Please check the URL, network connection, or if the site is blocking the request.")

# To run the asynchronous function
if __name__ == "__main__":
    asyncio.run(test_costco_crawl())