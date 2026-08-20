from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
import os 

load_dotenv(override=True)

# Initialize Gemini Model
api_key = os.getenv('GEMINI_API_KEY') or os.getenv('GOOGLE_API_KEY')
# Generator: Production model for complex creative/technical output
model = ChatGoogleGenerativeAI(
    model='gemini-3.5-flash',  # gemini-3.6-flash # gemini-3.5-flash
    max_retries=6,
    google_api_key=api_key
)

# Simple one-line prompt
prompt = PromptTemplate.from_template("{question}")

# model = ChatOpenAI()
parser = StrOutputParser()

# Chain: prompt → model → parser
chain = prompt | model | parser

# Run it
result = chain.invoke({"question": "What is the capital of india?"})
print(result)
