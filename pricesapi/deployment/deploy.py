# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from absl import app
import os
from dotenv import load_dotenv
from dotenv import set_key
import vertexai
from vertexai import agent_engines
from vertexai.preview.reasoning_engines import AdkApp
from pricesapi.agent import prices_comparison_agent


def main(argv: list[str]) -> None:

    ENV_FILE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))
    # Load environment variables from the specified .env file
    load_dotenv(dotenv_path=ENV_FILE_PATH)

    PROJECT = os.environ["GOOGLE_CLOUD_PROJECT"]
    LOCATION = os.environ["GOOGLE_CLOUD_LOCATION"]
    STAGING_BUCKET = os.environ["STAGING_BUCKET"]

    if not PROJECT:
        print("Missing required environment variable: GOOGLE_CLOUD_PROJECT")
        return
    elif not LOCATION:
        print("Missing required environment variable: GOOGLE_CLOUD_LOCATION")
        return
    elif not STAGING_BUCKET:
        print("Missing required environment variable: STAGING_BUCKET")
        return

    print(f"PROJECT: {PROJECT}")
    print(f"LOCATION: {LOCATION}")
    print(f"STAGING_BUCKET: {STAGING_BUCKET}")

    vertexai.init(
        project=PROJECT,
        location=LOCATION,
        staging_bucket=STAGING_BUCKET,
    )

    app = AdkApp(agent=prices_comparison_agent, enable_tracing=False)

    remote_agent = agent_engines.create(
        app,
        requirements="/app/requirements.txt",
        display_name="PricesComparision Agent",
        description="PricesComparision Agent that makes API calls and presents prices and compare them.",
    )

    print(f"Created remote agent: {remote_agent.resource_name}")
    print(f"Updating .env file with PRICES_COMPARE_AGENT_ENGINE_ID={remote_agent.resource_name}")
    
    try:
        set_key(ENV_FILE_PATH, "PRICES_COMPARE_AGENT_ENGINE_ID", remote_agent.resource_name)
        print(f"Updated PRICES_COMPARE_AGENT_ENGINE_ID in {ENV_FILE_PATH} to {remote_agent.resource_name}")
    except Exception as e:
        print(f"Error updating PRICES_COMPARE_AGENT_ENGINE_ID in {ENV_FILE_PATH} file: {e}")


if __name__ == "__main__":
    app.run(main)
