from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
import os 


load_dotenv()
os.environ['LANGCHAIN_PROJECT'] = 'Sequential LLM App'

prompt1 = PromptTemplate(
    template='Generate a detailed report on {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='Generate a 5 pointer summary from the following text \n {text}',
    input_variables=['text']
)

# Initialize Gemini Model
api_key = os.getenv('GEMINI_API_KEY') or os.getenv('GOOGLE_API_KEY')
# Generator: Production model for complex creative/technical output
model1 = ChatGoogleGenerativeAI(
    model='gemini-3.6-flash',  # gemini-3.6-flash # gemini-3.5-flash
    max_retries=6,
    google_api_key=api_key
)

model2 = ChatGoogleGenerativeAI(
    model='gemini-3.6-flash',  # gemini-3.6-flash # gemini-3.5-flash
    max_retries=6,
    google_api_key=api_key
)


parser = StrOutputParser()

chain = prompt1 | model1 | parser | prompt2 | model2 | parser

config = {
    'run_name':'Sequential LLM App-3',
    'tags' : ['Sequential LLM App-2'],
    'metadata': {
        'topic': 'Unemployment in India',
        'user': 'Siddharth'
    }
}

result = chain.invoke({'topic': 'Unemployment in India'}, config=config)

print(result)
