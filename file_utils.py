import logging
import base64
import os
import json
from google import genai
from google.genai import types
from openai import AzureOpenAI, OpenAI
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv(dotenv_path=".env")

# embedding = AzureOpenAI(
#     api_key=os.getenv("AZURE_OPENAI_KEY"),  
#     api_version="2023-05-15",
#     azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
# )

# llm = AzureOpenAI(
#     azure_endpoint="", 
#     api_key="",  
#     api_version=""
# )

embedding = OpenAI(api_key=os.getenv("OPENAI_KEY"))

llm = OpenAI(api_key=os.getenv("DS_OPENAI_KEY"), base_url="https://api.deepseek.com/'")

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY"),
)