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
import webbrowser
import threading
import os

# Importar Flask app
from app import app

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

def open_browser():
    """Abre el navegador después de un pequeño delay para que Flask se inicie"""
    time.sleep(2)
    webbrowser.open('http://localhost:5000')

if __name__ == '__main__':
    # add_files()
    
    print("\n=== Sistema de Consultas de Nutrición y Deporte ===")
    print("🚀 Iniciando NutriBot Web Interface...")
    print("📱 Abriendo navegador automáticamente...")
    print("🌐 Si el navegador no se abre, visita: http://localhost:5000")
    print("⏹️  Presiona Ctrl+C para detener el servidor\n")
    
    # Iniciar el navegador en un hilo separado
    browser_thread = threading.Thread(target=open_browser)
    browser_thread.daemon = True
    browser_thread.start()
    
    # Iniciar la aplicación Flask
    try:
        app.run(debug=False, host='0.0.0.0', port=5000)
    except KeyboardInterrupt:
        print("\n👋 ¡Hasta luego! Servidor detenido.")
    except Exception as e:
        print(f"\n❌ Error al iniciar el servidor: {str(e)}")
        print("💡 Asegúrate de que el puerto 5000 esté disponible.")

    # Código comentado de la interfaz de consola original
    # while True:
    #     pregunta = input("\nHaz tu pregunta: ")
    #     if pregunta.lower() in ["salir", "exit", "quit"]:
    #         print("\n¡Hasta luego!")
    #         break

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
    # preguntas_creatina = [
    #     "Respondeme las siguientes preguntas con respuestas breves, con sí o no vale en la mayoría de las cosas. Vamos a preguntar todo el rato respecto a la creatina. ¿Puede la creatina aumentar la fuerza muscular?", # Si
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