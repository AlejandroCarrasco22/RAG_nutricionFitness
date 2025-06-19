from langchain_chroma import Chroma
import pandas as pd
import time
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
from langchain.memory import ConversationBufferMemory

new_files = create_new_files()
router = RouterRagSql()

shared_memory = ConversationBufferMemory()

# Inicializar resolvers con la memoria compartida
sql_resolver = SQLResolver(memory=shared_memory)
rag_resolver = RAGResolver(memory=shared_memory)
end_to_end_resolver = EndToEndResolver(
    router=router,
    sql_resolver=sql_resolver,
    rag_resolver=rag_resolver,
    memory=shared_memory
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
            if isinstance(respuesta, str):
                print(respuesta)
            elif isinstance(respuesta, dict):
                print(respuesta.get("resultado", "No disponible"))
            else:
                print("Formato de respuesta no reconocido.")
        except Exception as e:
            print(f"\nError inesperado: {str(e)}")
            continue
    # preguntas_creatina = [
    #     "Respondeme las siguientes preguntas con respuestas breves, con sí o no vale en la mayoría de los cosas. Vamos a preguntar todo el rato respecto a la creatina. ¿Puede la creatina aumentar la fuerza muscular?", # Si
    #     "¿Mejora la memoria?", # Si
    #     "¿Es mala para las mujeres?", # No
    #     "¿Fortalece los huesos?", # Si
    #     "¿Daña los riñones?", # No
    #     "¿Ayuda a ganar músculo?", # Si
    #     "¿Se asocia con cancer de riñón?", # NO
    #     "¿Reduce el azúcar en sangre?", # Si
    #     "¿Baja el colesterol?", # Sí
    #     "¿Produce caída del cabello?", # No
    #     "¿Sobrecarga el hígado?", # No
    #     "¿Ayuda con la depresión?", # Si
    #     "¿Es antiinflamatoria?", # Si
    #     "¿Disminuye la grasa corporal?", # Si
    #     "¿Deshidrata el cuerpo?", # No
    #     "¿Reduce el daño muscular por ejercicio?", # Si
    #     "¿Reduce la pérdida muscular el envejecimiento?", # Si
    #     "¿Mejora la resistencia?", # Si
    #     "¿Reduce los sintomas como la lentitud mental?", # Si
    #     "¿Es la creatina mejor en hombres que en mujeres?", # No
    #     "¿Ayuda a recuperar más rápido del ejercicio?", # Si
    #     "¿Son algunas formas de cratina mejores que otras?", # No o no de manera significativa
    #     "¿Cúal es la mejor creatina que puedo tomar?", 
    #     "¿Cuánta creatina tienes que tomar al día para conseguir todos estos beneficios?" # Entre 5 a 10 gr  
    # ]
    # for pregunta in preguntas_creatina:
        
    #     print(f"\n--------------Pregunta: {pregunta}------------------")
    #     try:
    #         respuesta = end_to_end_resolver.resolver(pregunta)
    #         print("\n--- Respuesta ---")
    #         if isinstance(respuesta, str):
    #             print(respuesta)
    #         elif isinstance(respuesta, dict):
    #             print(respuesta.get("resultado", "No disponible"))
    #         else:
    #             print("Formato de respuesta no reconocido.")
    #     except Exception as e:
    #         print(f"\nError inesperado: {str(e)}")
    #         continue
    #     time.sleep(1)