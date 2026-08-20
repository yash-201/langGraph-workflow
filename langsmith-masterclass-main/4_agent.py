import os
import warnings
import requests
import urllib3
from dotenv import load_dotenv

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*create_react_agent has been moved.*")
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()
os.environ['LANGCHAIN_PROJECT'] = 'Agent App'

# Initialize Gemini Model
api_key = os.getenv('GEMINI_API_KEY') or os.getenv('GOOGLE_API_KEY')

search_tool = DuckDuckGoSearchRun()

@tool
def get_weather_data(city: str) -> str:
  """
  This function fetches the current weather data for a given city
  """
  url = f'https://api.weatherstack.com/current?access_key=f07d9636974c4120025fadf60678771b&query={city}'

  try:
      response = requests.get(url, verify=False)
      return response.json()
  except Exception as e:
      return f"Error fetching weather: {e}"

llm = ChatGoogleGenerativeAI(
    model='gemini-2.5-flash',
    max_retries=6,
    google_api_key=api_key
)

# Step 2: Create the agent using Gemini and tools
agent_executor = create_agent(
    model=llm,
    tools=[search_tool, get_weather_data]
)

# Step 3: Invoke
response = agent_executor.invoke({"messages": [("user", "What is the lastest update of AI in India and the weather in Banglore")]})
print("\n=== Final Answer ===")
print(response["messages"][-1].content)