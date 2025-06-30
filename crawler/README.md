# crawl4ai

## docker commands
```
    cd crawler
	docker build --no-cache -t venkat5ai/crawl4ai-utils:latest -f Dockerfile-crawl4ai .
	docker run --name crawl4ai-utils --rm -it -v ".:/app" -v "%GOOGLE_APPLICATION_CREDENTIALS%":/tmp/keys.json -e GOOGLE_APPLICATION_CREDENTIALS="/tmp/keys.json" --entrypoint /bin/bash  venkat5ai/crawl4ai-utils
```

## References
    - https://docs.crawl4ai.com/core/installation/
    - https://github.com/unclecode/crawl4ai
    - 
