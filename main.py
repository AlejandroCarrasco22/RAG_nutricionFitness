from langchain_chroma import Chroma
import chromadb
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from keys import CHROMA_DB_DIR, CHROMA_ABSTRACTS_DB_DIR
from keys import LLM, EMBBEDING
from create_new_files import create_new_files
from routing import RouterRagSql

new_files = create_new_files()
router = RouterRagSql()

# Configurar el modelo ChatOpenAI para Azure
llm = LLM

# Configurar embeddings para Azure
embedding = EMBBEDING
# Conectarse a la base Chroma local con embeddings correctos
chroma_local = Chroma(persist_directory=CHROMA_DB_DIR, embedding_function=embedding)
chorma_abstracts_db = Chroma(persist_directory=CHROMA_ABSTRACTS_DB_DIR, embedding_function=embedding)



def add_files():
    persistent_client = chromadb.PersistentClient(path = CHROMA_DB_DIR)
    llm = LLM
    embedding = EMBBEDING
    new_files.add_files_to_vectordb(persistent_client=persistent_client,
                                    embedding=embedding,
                                    llm=llm,
                                    full_chroma_db_dir=CHROMA_DB_DIR,
                                    abtracts_chorma_db_dir=CHROMA_ABSTRACTS_DB_DIR)
    
if __name__ == '__main__':
    # print(respuesta(input('Haz tu pregunta: '), llm, chroma_local, prompt(texto))['answer'])
    add_files()
    
    
