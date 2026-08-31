from langgraph.graph import StateGraph, END, START, add_messages
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import Literal, TypedDict, Annotated
import os

load_dotenv()

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


class OutputSchema(BaseModel):
    response: str
    model_config = {"extra": "ignore"}


llm = ChatGoogleGenerativeAI(model="gemini-3.7-flash", api_key=os.getenv("GOOGLE_API_KEY"))
structured_llm = llm.with_structured_output(OutputSchema)

def chatty_bot(state: ChatState):
    response = structured_llm.invoke(state["messages"])
    state["messages"].append(response.response)
    return state["messages"]

graph = StateGraph(ChatState)

graph.add_node("chatbot", chatty_bot)

graph.add_edge(START, "chatbot")
graph.add_edge("chatbot", END)

graph = graph.compile()

result = graph.invoke({"messages": [HumanMessage(content="How are you today?")]})
print(result)
