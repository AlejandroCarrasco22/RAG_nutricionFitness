from langchain_openai import AzureChatOpenAI
from langchain_openai import AzureOpenAIEmbeddings

ENDPOINT = "***" # Azure OpenAI endpoint Sustituir por la URL de tu cuenta de Azure OpenAI
MODEL_NAME = "gpt-4o"
DEPLOYMENT = "gpt-4o"
SUBSCRIPTION_KEY = '***' # Sustituir por la clave de tu cuenta de Azure OpenAI
API_VERSION = "2024-12-01-preview"


DEPLOYMENT_EMBBEDING = "text-embedding-3-small"
API_VERSION_EMBEDDING = "2024-02-01"

CHROMA_DB_DIR = "./chroma_db_dir"
CHROMA_ABSTRACTS_DB_DIR = "./chroma_db_abstracts_dir"

LLM = AzureChatOpenAI(
    openai_api_key=SUBSCRIPTION_KEY,
    azure_endpoint=ENDPOINT,
    api_version=API_VERSION,
    azure_deployment=DEPLOYMENT,
    temperature=0
)

# Configurar embeddings para Azure
EMBBEDING = AzureOpenAIEmbeddings(
    azure_deployment=DEPLOYMENT_EMBBEDING,
    api_key=SUBSCRIPTION_KEY,
    azure_endpoint=ENDPOINT,
    openai_api_version=API_VERSION_EMBEDDING
)