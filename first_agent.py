import tavily
from dotenv import load_dotenv

load_dotenv()

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from tavily import TavilyClient

tavily = TavilyClient()

@tool
def search(query: str)-> str:
    """
    tool that searches the query over internet
    Args:
        query (str): the search query
    Returns:
        The search result
    """
    print(f"Searching for {query}")
    return tavily.search(query=query)

llm = ChatOpenAI()
tools = [search]
agent = create_agent(model=llm, tools=tools)


def main():
    print("Basic agent workflow")
    result = agent.invoke({"messages":HumanMessage(content="what is the weather in Tokyo")})
    print(result["messages"][-1].content)

if __name__ == "__main__":
    main()