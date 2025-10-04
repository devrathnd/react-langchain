from typing import List
from dotenv import load_dotenv
from langchain.agents import tool
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_groq import ChatGroq

from callbacks import AgentCallbackHandler

load_dotenv()

@tool
def get_text_length(text: str) -> int:
    """Returns the length of the text by characters"""
    print(f"get_text_length enter with {text=}")
    text = text.strip("'\n").strip("'")
    return len(text)


if __name__ == "__main__":
    print("Hello, World!")

    tools = [get_text_length]

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0,
        callbacks=[AgentCallbackHandler()]
    )
    
    # Bind tools to the LLM using native tool calling
    llm_with_tools = llm.bind_tools(tools)
    
    messages = [
        HumanMessage(content="What is the text length of 'Hello, World' in characters?")
    ]
    
    # Agent loop
    while True:
        # Invoke the LLM with tools
        response = llm_with_tools.invoke(messages)
        print(f"Response: {response}")
        
        # Add AI response to messages
        messages.append(response)
        
        # Check if there are tool calls
        if not response.tool_calls:
            # No more tool calls, we have the final answer
            print(f"\nFinal Answer: {response.content}")
            break
        
        # Execute tool calls
        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            
            print(f"\nExecuting tool: {tool_name}")
            print(f"Tool arguments: {tool_args}")
            
            # Find and execute the tool
            selected_tool = None
            for tool in tools:
                if tool.name == tool_name:
                    selected_tool = tool
                    break
            
            if selected_tool:
                observation = selected_tool.invoke(tool_args)
                print(f"Observation: {observation}")
                
                # Add tool result to messages using ToolMessage
                messages.append(
                    ToolMessage(
                        content=str(observation),
                        tool_call_id=tool_call["id"]
                    )
                )
            else:
                print(f"Tool {tool_name} not found!")
