from flask import Flask, render_template, request, jsonify, url_for
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

app = Flask(__name__, static_folder='static')

# Inicializar componentes del chatbot
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

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        pregunta = data.get('message', '')
        
        if not pregunta.strip():
            return jsonify({'error': 'Por favor, ingresa una pregunta válida'})
        
        # Procesar la pregunta con el resolver
        respuesta = end_to_end_resolver.resolver(pregunta)
        
        # Formatear la respuesta
        if isinstance(respuesta, str):
            resultado = respuesta
        elif isinstance(respuesta, dict):
            resultado = respuesta.get("resultado", "No disponible")
        else:
            resultado = "Formato de respuesta no reconocido."
        
        return jsonify({
            'response': resultado,
            'timestamp': time.strftime('%H:%M')
        })
        
    except Exception as e:
        return jsonify({'error': f'Error inesperado: {str(e)}'})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000) 