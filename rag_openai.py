from langchain_chroma import Chroma
import chromadb
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from keys import ENDPOINT, DEPLOYMENT, SUBSCRIPTION_KEY, API_VERSION, DEPLOYMENT_EMBBEDING, API_VERSION_EMBEDDING
from keys import CHROMA_DB_DIR, CHROMA_ABSTRACTS_DB_DIR
from keys import LLM, EMBBEDING
from create_new_files import create_new_files

new_files = create_new_files()


# Configurar el modelo ChatOpenAI para Azure
llm = LLM

# Configurar embeddings para Azure
embedding = EMBBEDING
# Conectarse a la base Chroma local con embeddings correctos
chroma_local = Chroma(persist_directory=CHROMA_DB_DIR, embedding_function=embedding)
chorma_abstracts_db = Chroma(persist_directory=CHROMA_ABSTRACTS_DB_DIR, embedding_function=embedding)

# Prompt y funciones RAG
def prompt(texto):
    system_prompt = texto + "\n\n{context}"
    return ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

def respuesta(pregunta, llm, chroma_db, prompt):
    retriever = chroma_db.as_retriever()
    chain = create_stuff_documents_chain(llm, prompt)
    rag = create_retrieval_chain(retriever, chain)
    results = rag.invoke({"input": pregunta})
    return results

texto = """Tú eres un asistente para tareas de respuesta a preguntas.
Usa los siguientes fragmentos de contexto recuperado para responder 
la pregunta. Si no sabes la respuesta, di que no sabes. 
Usa un máximo de tres oraciones y mantén la respuesta concisa."""


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
    
    
