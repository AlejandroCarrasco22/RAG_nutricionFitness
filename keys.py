from langchain_openai import AzureChatOpenAI
from langchain_openai import AzureOpenAIEmbeddings

ENDPOINT = "https://aleja-mavdk5nu-eastus2.cognitiveservices.azure.com/"
MODEL_NAME = "gpt-4o-mini"
DEPLOYMENT = "gpt-4o-mini"
SUBSCRIPTION_KEY = 'D55vuDUYoqo7Bzj9R0SwUhQFA8MZ9l2Eh5yWCOfKSlMXLmKw7JBpJQQJ99BEACHYHv6XJ3w3AAAAACOGRJuO'
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