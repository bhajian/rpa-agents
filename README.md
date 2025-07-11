RPA Agent
This project implements an agentic AI that can control a web browser to perform tasks based on user instructions. It uses LangChain for the agent logic, Playwright for browser automation, and AWS Bedrock as the AI model provider.

Setup
Clone the repository:

git clone <your-repo-url>
cd rpa-agent

Create a virtual environment:

python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

Install the dependencies:

pip install -r requirements.txt

Install Playwright browsers:

playwright install

Configure AWS Credentials:
Create a file named .env in the main rpa-agent directory. You will need to add your AWS bearer token and region to this file. See the .env file generation for the exact format.

Running the Agent
To start the agent, run the main module from the project's root directory:

python -m src.main

You will then be prompted in your terminal to enter a task for the agent to perform.
