import asyncio
import uuid
from typing import Optional

from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import END, START
from langgraph.graph import StateGraph
from langgraph.types import Command, interrupt
from typing_extensions import TypedDict


class State(TypedDict):
    """The graph state."""

    input: str
    """The input value of the node."""

    ai_answer: Optional[str]
    """AI answer"""

    human_answer: Optional[str]
    """Human value will be updated using an interrupt."""


async def node1(state: State):
    print(f"> Node1 input: {state['input']}")
    await asyncio.sleep(1)
    return {"ai_answer": "This is the output of node1"}


async def node2(state: State):
    print(f"> Received input: {state['ai_answer']}")
    answer = interrupt(
        # This value will be sent to the client
        # as part of the interrupt information.
        {"ai_answer": "What is your age?"}
    )
    print(f"> Received an input from the interrupt: {answer}")
    await asyncio.sleep(2)
    return {"human_answer": answer}


async def node3(state: State):
    print(f"> Received input: {state['human_answer']}")
    await asyncio.sleep(3)
    return {"ai_answer": "This is the output of node3"}

async def node4(state: State):
    if state['human_answer'].isdigit() == False:
        print(f"> Buy")
        return

    await asyncio.sleep(3)

    if int(state['human_answer']) < 18:
        print(f"> You are too young to buy this product.")
        return
    print(f"> You can buy this product.")


builder = StateGraph(State)
builder.add_node("node1", node1)
builder.add_node("node2", node2)
builder.add_node("node3", node3)
builder.add_node("node4", node4)

builder.add_edge(START, "node1")
builder.add_edge("node1", "node2")
builder.add_edge("node2", "node3")
builder.add_edge("node3", "node4")
builder.add_edge("node4", END)


# A checkpointer must be enabled for interrupts to work!
checkpointer = MemorySaver()
graph = builder.compile(checkpointer=checkpointer)

config = {
    "configurable": {
        "thread_id": uuid.uuid4(),
    }
}


async def run_graph():
    async for chunk in graph.astream({"input": "something"}, config):
        print(chunk)

    command = Command(resume=input("Enter the interrupt command: "))

    async for chunk in graph.astream(command, config):
        print(chunk)


if __name__ == "__main__":
    asyncio.run(run_graph())