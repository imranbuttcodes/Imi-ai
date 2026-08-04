from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class FileSystemState(TypedDict):
    """
    State for the FileSystem Agent loop.
    Must track the full conversational history (messages) so the LLM
    can see the output of the tool calls it makes.
    """
    query: str
    messages: Annotated[list[BaseMessage], add_messages]
