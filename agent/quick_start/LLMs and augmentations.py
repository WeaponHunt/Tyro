import os
import getpass
from langchain.chat_models import init_chat_model

llm = init_chat_model(
    "qwen-turbo",
    model_provider="openai",
    openai_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
    openai_api_key="sk-9d42be35fbba4ef8ab8d217c2a613869",
    temperature=0
)

# Schema for structured output
from pydantic import BaseModel, Field


#强制规定模型输出的格式，方便后续解析和使用
class SearchQuery(BaseModel):
    search_query: str = Field(None, description="Query that is optimized web search.")
    justification: str = Field(
        None, description="Why this query is relevant to the user's request."
    )


# Augment the LLM with schema for structured output
structured_llm = llm.with_structured_output(SearchQuery)

# Invoke the augmented LLM
# output = structured_llm.invoke("How does Calcium CT score relate to high cholesterol?")
# print(output)

# Define a tool
def multiply(a: int, b: int) -> int:
    return a * b

# Augment the LLM with tools
llm_with_tools = llm.bind_tools([multiply])

# Invoke the LLM with input that triggers the tool call
msg = llm_with_tools.invoke("103乘以101等于多少?")

# Get the tool call
print(msg.tool_calls)