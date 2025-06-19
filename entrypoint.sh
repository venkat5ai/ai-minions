#!/bin/sh

# This script is the ENTRYPOINT for the Docker container. 
# It executes different Python applications based on the first argument.

# Ensure unbuffered output for Python, which is good for Docker logs.
# Setting PYTHONUNBUFFERED=1 in Dockerfile ENV is sufficient, but this doesn't hurt.
export PYTHONUNBUFFERED=1

# The first argument to the entrypoint script dictates which Python script to run
COMMAND=$1
shift # Remove the first argument (command) from $@, leaving only its arguments

case "$COMMAND" in
    cloud)
        echo "Starting main application (app.py) in cloud mode..."
        # Use exec as it's the final, long-running process that stays alive
        exec python app.py "cloud" "$@"
        ;;
    deploy)
        echo "Running deployment script (deployment/deploy.py)..."
        
        # Run the deployment script and capture its output (both stdout and stderr)
        # This allows us to check for success and extract the AGENT_ENGINE_ID.
        DEPLOY_OUTPUT=$(python deployment/deploy.py "$@" 2>&1)
        DEPLOY_STATUS=$? # Capture the exit status of deploy.py

        # Print the output of the deploy script to the container logs
        echo "$DEPLOY_OUTPUT"

        if [ $DEPLOY_STATUS -eq 0 ]; then
            echo "Deployment script completed successfully."
            
            # Extract AGENT_ENGINE_ID from the deployment output.
            # This regex looks for the pattern "projects/<number>/locations/<location>/reasoningEngines/<number>"
            AGENT_ENGINE_ID=$(echo "$DEPLOY_OUTPUT" | grep -oP 'projects/\d+/locations/[a-z0-9-]+/reasoningEngines/\d+' | tail -n 1) # tail -n 1 to get the last match

            if [ -z "$AGENT_ENGINE_ID" ]; then
                echo "WARNING: Could not extract AGENT_ENGINE_ID from deployment output."
                echo "         Please ensure deployment/deploy.py prints the full Agent Engine ID upon successful creation."
                echo "         Attempting to start app.py anyway, but it might fail if AGENT_ENGINE_ID is not properly set in .env."
                # If ID extraction fails, app.py will use what's in .env (if set) or its default.
            else
                echo "Extracted AGENT_ENGINE_ID: $AGENT_ENGINE_ID"
                # Export the extracted ID as an environment variable for app.py
                # This overrides any AGENT_ENGINE_ID set in the .env file for this specific run.
                export AGENT_ENGINE_ID
                echo "AGENT_ENGINE_ID environment variable set for subsequent application start."
            fi

            echo "Starting main application (app.py) in cloud mode after deployment..."
            # Use exec for app.py as it's now the final, long-running process
            exec python app.py "cloud" "$@" # Pass "cloud" and any other original arguments to app.py
        else
            echo "Deployment script failed with exit code $DEPLOY_STATUS. Not starting application."
            exit $DEPLOY_STATUS # Exit with the failure code of the deployment script
        fi
        ;;
    destroy)
        echo "Running destruction script (destroy.py)..."
        # Do not use exec here, as we want the shell script to exit after destroy.py finishes
        python destroy.py "$@"
        DESTROY_STATUS=$?
        if [ $DESTROY_STATUS -eq 0 ]; then
            echo "Destruction script completed successfully."
        else
            echo "Destruction script failed with exit code $DESTROY_STATUS."
        fi
        exit $DESTROY_STATUS # Exit with the failure code of the destruction script
        ;;
    *)
        echo "Unknown command: $COMMAND"
        echo "Usage: docker run ... <image> [cloud|deploy|destroy]"
        exit 1
        ;;
esac