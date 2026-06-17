from dotenv import load_dotenv

load_dotenv(verbose=True)

from langchain_core.messages import HumanMessage
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch

llm = ChatOpenAI()
tools = [TavilySearch()]
agent = create_agent(model=llm, tools=tools)

def main():
    print(f"Tavily Search Starting")
    result = agent.invoke({"messages":[HumanMessage(content="Search for 3 job postings for an AI Engineer and get me the verified contact details of the recruiters who are currently handling them")]})
    print(result["messages"][-1].content)

if __name__ == "__main__":
    main()