import asyncio
import nest_asyncio
from dotenv import load_dotenv
from playwright.async_api import async_playwright
from .agent.core import call_agent

# Allows asyncio to be used in environments that have an existing event loop (like Jupyter)
nest_asyncio.apply()
load_dotenv()

async def main():
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=False)
            page = await browser.new_page()
            await page.goto("https://www.google.com")
            
            # Get user input for the task
            task = input("Please enter the task for the agent: ")

            final_response = await call_agent(task, page)
            print(f"Final response: {final_response}")
            
            await browser.close()

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    # In some environments, you might need to get or create an event loop
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    # Ensure the main coroutine is run in the loop
    if loop.is_running():
        loop.create_task(main())
    else:
        loop.run_until_complete(main())
        