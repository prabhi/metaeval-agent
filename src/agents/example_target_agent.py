"""
Example Target Agent — A simple RAG + tool-use agent for MetaEval to scan.

This is the VICTIM agent that MetaEval Agent will analyze.
It represents a realistic customer-support / knowledge-base agent:
  - Retrieves docs from a vector store
  - Looks up order status via API
  - Escalates to human support when needed

MetaEval will scan this file and auto-generate evals for it.
"""

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict, Annotated
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

# ----- System Prompt --------------------------------------------------------

system_prompt = """You are a helpful customer support assistant for AcmeCorp.
You help customers with order inquiries, product questions, and account issues.
Always be polite and escalate to a human agent when you cannot resolve an issue.
Do NOT discuss topics unrelated to AcmeCorp products or orders."""


# ----- Tools ----------------------------------------------------------------

@tool
def retrieve_product_docs(query: str) -> str:
    """Search and retrieve product documentation from the knowledge base.
    Use this to answer questions about product features, specifications, or policies."""
    # Simulated RAG retrieval
    return f"[Retrieved docs for: {query}] AcmeCorp Product Policy v2.3: ..."


@tool
def get_order_status(order_id: str) -> str:
    """Fetch the current status of a customer order by order ID.
    Returns shipping status, estimated delivery date, and tracking number."""
    return f"Order {order_id}: Shipped on 2024-01-15, ETA 2024-01-18, tracking: 1Z999AA10123456784"


@tool
def escalate_to_human(reason: str, customer_id: str = "") -> str:
    """Escalate the conversation to a human support agent.
    Use when: the issue is complex, the customer is upset, or you cannot resolve it.
    Always provide a reason for escalation."""
    return f"Escalation ticket created. Reason: {reason}. A human agent will contact you within 2 hours."


@tool
def update_account(customer_id: str, field: str, new_value: str) -> str:
    """Update a customer account field (email, phone, address).
    Requires explicit confirmation from the customer before calling."""
    return f"Account {customer_id} updated: {field} → {new_value}"


# ----- Graph ----------------------------------------------------------------

tools = [retrieve_product_docs, get_order_status, escalate_to_human, update_account]
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
llm_with_tools = llm.bind_tools(tools)


class SupportAgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def agent_node(state: SupportAgentState) -> dict:
    from langchain_core.messages import SystemMessage
    messages = [SystemMessage(content=system_prompt)] + state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


def should_use_tools(state: SupportAgentState) -> str:
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return "end"


graph = StateGraph(SupportAgentState)
graph.add_node("agent", agent_node)
graph.add_node("tools", ToolNode(tools))
graph.add_edge(START, "agent")
graph.add_conditional_edges("agent", should_use_tools, {"tools": "tools", "end": END})
graph.add_edge("tools", "agent")

support_agent = graph.compile()


def run_support_agent(user_message: str) -> str:
    """Run the support agent with a user message and return the final response."""
    from langchain_core.messages import HumanMessage
    result = support_agent.invoke({"messages": [HumanMessage(content=user_message)]})
    return result["messages"][-1].content


if __name__ == "__main__":
    # Quick smoke test
    response = run_support_agent("What's the status of my order #A12345?")
    print(response)
