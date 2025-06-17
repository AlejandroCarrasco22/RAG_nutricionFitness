from langchain_chroma import Chroma
import pandas as pd
import chromadb
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from keys import CHROMA_DB_DIR, CHROMA_ABSTRACTS_DB_DIR
from keys import LLM, EMBBEDING
from create_new_files import create_new_files
from routing import RouterRagSql
from resolvers.rag_resolver import RAGResolver
from resolvers.sql_resolver import SQLResolver
from resolvers.end_to_end_resolver import EndToEndResolver

new_files = create_new_files()
router = RouterRagSql()
sql_resolver = SQLResolver()
rag_resolver = RAGResolver()

end_to_end_resolver = EndToEndResolver(
    router=router,
    sql_resolver=sql_resolver,
    rag_resolver=rag_resolver
)
# Conectarse a la base Chroma local con embeddings correctos
chroma_local = Chroma(persist_directory=CHROMA_DB_DIR, embedding_function=EMBBEDING)
chorma_abstracts_db = Chroma(persist_directory=CHROMA_ABSTRACTS_DB_DIR, embedding_function=EMBBEDING)



def add_files():
    persistent_client = chromadb.PersistentClient(path = CHROMA_DB_DIR)
    new_files.add_files_to_vectordb(persistent_client=persistent_client,
                                    embedding=EMBBEDING,
                                    llm=LLM,
                                    full_chroma_db_dir=CHROMA_DB_DIR,
                                    abtracts_chorma_db_dir=CHROMA_ABSTRACTS_DB_DIR)
    
if __name__ == '__main__':
    # add_files()
    
    print("\n=== Sistema de Consultas de Nutrición y Deporte ===")
    print("Puedes preguntar sobre:")
    print("1. Composición nutricional de alimentos")
    print("2. Información sobre suplementación deportiva")
    print("3. Recomendaciones de nutrición deportiva")
    print("(Escribe 'salir', 'exit' o 'quit' para terminar)\n")
    
    while True:
        pregunta = input("\nHaz tu pregunta: ")
        if pregunta.lower() in ["salir", "exit", "quit"]:
            print("\n¡Hasta luego!")
            break

        try:
            respuesta = end_to_end_resolver.resolver(pregunta)
            print("\n--- Respuesta ---")
            print(respuesta)
            
        except Exception as e:
            print(f"\nError inesperado: {str(e)}")
            continue
    
