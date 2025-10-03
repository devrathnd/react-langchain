from dotenv import load_dotenv
from langchain.agents import tool
load_dotenv()

@tool
def get_text_length(text:str) -> int:
    """Returns the length of the text by characters"""
    print(f"get_text_length enter with {text=}")
    text = text.strip("'\n").strip("'")
    return len(text)

if __name__ == "__main__":
    print("Hello, World!")
    print(get_text_length.invoke(input={"text": "Hello, World!"}))