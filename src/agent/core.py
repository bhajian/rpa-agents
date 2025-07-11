import os
import base64
import re
import asyncio
from pathlib import Path
from langchain_core.messages import SystemMessage
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, chain
from langchain_aws import ChatBedrock
from langgraph.graph import StateGraph, END
from IPython.display import display, Image
from .state import AgentState
from ..tools.browser import click, type_text, scroll, wait, go_back, to_google

# Load JavaScript to mark page elements
js_path = Path(__file__).parent.parent / "js/mark_page.js"
with open(js_path) as f:
    mark_page_script = f.read()

@chain
async def mark_page(page):
    await page.evaluate(mark_page_script)
    for _ in range(10):
        try:
            bboxes = await page.evaluate("markPage()")
            break
        except Exception:
            await asyncio.sleep(0.5)
    else:
        bboxes = []
    screenshot = await page.screenshot()
    await page.evaluate("unmarkPage()")
    return {
        "img": base64.b64encode(screenshot).decode(),
        "bboxes": bboxes,
    }

async def annotate(state):
    marked_page = await mark_page.with_retry().ainvoke(state["page"])
    return {**state, **marked_page}

def format_descriptions(state):
    labels = []
    for i, bbox in enumerate(state["bboxes"]):
        text = bbox.get("ariaLabel") or bbox.get("text", "")
        el_type = bbox.get("type", "unknown")
        labels.append(f'{i} (<{el_type}/>): "{text}"')
    bbox_descriptions = "\nValid Bounding Boxes:\n" + "\n".join(labels)
    return {**state, "bbox_descriptions": bbox_descriptions}

def parse(text: str) -> dict:
    action_prefix = "Action: "
    if not text.strip().split("\n")[-1].startswith(action_prefix):
        return {"action": "retry", "args": f"Could not parse LLM Output: {text}"}
    action_block = text.strip().split("\n")[-1]
    action_str = action_block[len(action_prefix):]
    split_output = action_str.split(" ", 1)
    action = split_output[0].strip()
    action_input = None
    if len(split_output) > 1:
        action_input = [inp.strip().strip("[]") for inp in split_output[1].strip().split(";")]
    return {"action": action, "args": action_input}

# LLM and agent setup
prompt = hub.pull("wfh/web-voyager")

# Using Claude 3.5 Sonnet, a powerful multi-modal model on Bedrock.
# The ChatBedrock class will automatically use the credentials configured
# in the environment, including the new AWS_BEARER_TOKEN_BEDROCK.
llm = ChatBedrock(
    model_id="anthropic.claude-3-5-sonnet-20240620-v1:0",
    model_kwargs={"temperature": 0.1},
)

agent_runnable = annotate | RunnablePassthrough.assign(
    prediction=format_descriptions | prompt | llm | StrOutputParser() | parse
)

def update_scratchpad(state: AgentState):
    old_scratchpad = state.get("scratchpad", [])
    if old_scratchpad:
        content = old_scratchpad[0].content
        last_line = content.rsplit("\n", 1)[-1]
        match = re.match(r"(\d+)\.", last_line)
        step = int(match.group(1)) + 1 if match else len(old_scratchpad) + 1
    else:
        content = "Previous action observations:"
        step = 1
    
    updated_content = content + f"\n{step}. {state['observation']}"
    return {**state, "scratchpad": [SystemMessage(content=updated_content)]}

# Graph setup
graph_builder = StateGraph(AgentState)
graph_builder.add_node("agent", agent_runnable)
graph_builder.set_entry_point("agent")
graph_builder.add_node("update_scratchpad", update_scratchpad)
graph_builder.add_edge("update_scratchpad", "agent")

tools = {
    "Click": click,
    "Type": type_text,
    "Scroll": scroll,
    "Wait": wait,
    "GoBack": go_back,
    "Google": to_google,
}

for node_name, tool in tools.items():
    graph_builder.add_node(
        node_name,
        RunnableLambda(tool) | (lambda observation: {"observation": observation}),
    )
    graph_builder.add_edge(node_name, "update_scratchpad")

def select_tool(state: AgentState):
    action = state["prediction"]["action"]
    if action == "ANSWER":
        return END
    if action == "retry":
        return "agent"
    return action

graph_builder.add_conditional_edges("agent", select_tool)
graph = graph_builder.compile()

async def call_agent(question: str, page, max_steps: int = 150):
    event_stream = graph.astream(
        {
            "page": page,
            "input": question,
            "scratchpad": [],
        },
        {
            "recursion_limit": max_steps,
        },
    )
    final_answer = None
    steps = []
    async for event in event_stream:
        if "agent" not in event:
            continue
        pred = event["agent"].get("prediction") or {}
        action = pred.get("action")
        action_input = pred.get("args")
        
        try:
            display.clear_output(wait=False)
            steps.append(f"{len(steps) + 1}. {action}: {action_input}")
            print("\n".join(steps))
            display(Image(base64.b64decode(event["agent"]["img"])))
        except (NameError, ImportError):
            print(f"Step {len(steps) + 1}: {action} - {action_input}")

        if action and "ANSWER" in action:
            final_answer = action_input[0] if action_input else ""
            break
    return final_answer
