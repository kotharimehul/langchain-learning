from langchain.tools import tool
from tavily import TavilyClient
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

load_dotenv()
tavily_client=TavilyClient()

@tool
def search(query: str) -> str:
    
    """Search the web for information"""

    

    print(f"Searching the web for: {query}")
    return tavily_client.search(query)

llm=ChatOpenAI(model="gpt-4o-mini")
tools=[search]
agent=create_agent(model=llm, tools=tools)

def main(): 
    query="Search for 5 job openings for platform engineer in india"
    response=agent.invoke({"messages": HumanMessage(content=query)})
    print(response)

if __name__ == "__main__":
    main()


    