from langchain_chroma import Chroma 
from langchain.docstore.document import Document
from langchain_community.document_loaders import PyMuPDFLoader
from pathlib import Path
import re
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb


class create_new_files:
    """
        Extrae o genera resúmenes (abstracts) de PDFs con dos lógicas:
        1. Si no hay abstract, lo genera desde cero de forma exhaustiva.
        2. Si hay un abstract, lo enriquece con información clave del texto completo.
    """

    def generate_from_azure_openai(self, prompt: str, llm_instance) -> str:
        """
        Genera una respuesta a partir de un prompt usando una instancia de AzureChatOpenAI.
        """
        try:
            # El método invoke es más simple para una única llamada no-streaming
            response = llm_instance.invoke(prompt)
            return response.content.strip()
        except Exception as e:
            # En caso de error, devolvemos un string de error para poder manejarlo
            print(f"ERROR durante la generación: {e}")
            return f"ERROR_GENERACION: {e}"


    def create_or_extract_abstracts(self, pdf_paths, llm_instance):
        """
        Extrae el resumen (abstract) de un PDF. Si no existe, lo genera usando un LLM.
        """
        abstract_docs = []
        
        # --- PROMPT 1: Para generar un abstract desde cero ---
        generation_prompt_template = """
        Eres un asistente de investigación experto en la síntesis de documentos técnicos y académicos.
        Tu tarea es generar un resumen exhaustivo y bien estructurado (un 'abstract') del siguiente documento.

        Para ello, sigue estas instrucciones rigurosamente:
        1.  **Objetivo Principal:** Identifica y expón claramente el propósito central del estudio o documento.
        2.  **Puntos Clave:** Extrae y lista TODOS los temas o argumentos importantes tratados.
        3.  **Entidades Específicas (¡Muy Importante!):** Enumera de forma explícita y completa todos los siguientes elementos mencionados en el texto:
            - **Suplementos:** Todos los suplementos mencionados (e.g., Creatina, Proteína Whey, Omega-3, HMB, etc.).
            - **Tipos de Ejercicio:** Todas las modalidades de entrenamiento (e.g., entrenamiento de fuerza, HIIT, carrera de resistencia).
            - **Alimentos o Dietas:** Cualquier dieta o tipo de alimento relevante (e.g., dieta mediterránea, ayuno intermitente, dieta vegetariana).
            - **Población Objetivo:** El grupo de personas al que se dirige el estudio (e.g., atletas de élite, adultos mayores, adolescentes).
        4.  **Conclusiones:** Resume los hallazgos o conclusiones principales del documento.

        El resumen final debe ser denso en información, objetivo y escrito en un estilo formal.

        Texto del documento:
        ---
        {document_text}
        ---
        """
        
        # --- PROMPT 2: Para enriquecer un abstract existente ---
        enhancement_prompt_template = """
        Eres un asistente de investigación experto que mejora y enriquece resúmenes existentes.
        A continuación, te proporciono un 'Resumen Original' y el 'Texto Completo' de un documento.

        Tu tarea es la siguiente:
        1.  Lee el 'Texto Completo' y compáralo con el 'Resumen Original'.
        2.  Identifica la información clave que NO está presente en el 'Resumen Original'. Enfócate especialmente en encontrar:
            - **Suplementos:** (e.g., Creatina, Proteína Whey).
            - **Tipos de Ejercicio o Entrenamiento.**
            - **Población Objetivo:** (e.g., atletas, personas mayores).
        3.  Genera un 'Resumen Mejorado' que combine la información del 'Resumen Original' con los puntos clave que has encontrado. El resultado final debe ser un único párrafo coherente y fluido, que mantenga el estilo del original pero que sea más completo. No añadas encabezados como "Resumen Mejorado".

        Resumen Original:
        ---
        {original_abstract}
        ---

        Texto Completo:
        ---
        {document_text}
        ---
        """

        for path in pdf_paths:
            print(f"--- Procesando: {path} ---")
            try:
                loader = PyMuPDFLoader(str(path))
                pages = loader.load()
                full_text = " ".join([page.page_content for page in pages])
                
                # 1. INTENTAR EXTRAER EL ABSTRACT EXISTENTE
                start_match = re.search(r'(Abstract|Resumen)\b', full_text, re.IGNORECASE)
                abstract_text = ""
                
                # Preparamos el texto truncado para pasarlo a los prompts
                max_chars = 100000 
                truncated_text = full_text[:max_chars]

                if start_match:
                    start_index = start_match.end()
                    end_match = re.search(r'\b(Introduction|Introducción|Keywords|Palabras clave)\b', full_text[start_index:], re.IGNORECASE)
                    original_abstract = full_text[start_index:end_match.start() if end_match else start_index+2500].strip()
                    # Preparamos y ejecutamos el prompt de mejora
                    prompt_para_mejorar = enhancement_prompt_template.format(
                        original_abstract=original_abstract,
                        document_text=truncated_text
                    )
                    final_abstract_text = self.generate_from_azure_openai(prompt_para_mejorar, llm_instance)
                    
                    if "ERROR_GENERACION" not in final_abstract_text:
                        print("Abstract enriquecido con éxito.")
                    else:
                        print(f"Fallo al enriquecer el abstract para {path.name}. Usando el original.")
                        final_abstract_text = original_abstract # Usamos el original como fallback

                else:
                    print(f"No se encontró abstract en '{path}'. Generando uno nuevo con Azure OpenAI...")

                    # gpt-4o-mini tiene 128k tokens
                    # 4 caracteres ~= 1 token. 100,000 caracteres ~= 25,000 tokens, más que seguro.
                    max_chars = 100000
                    truncated_text = full_text[:max_chars]

                    prompt_para_generar = generation_prompt_template.format(document_text=truncated_text)
                    final_abstract_text = self.generate_from_azure_openai(prompt_para_generar, llm_instance)
                    
                    if "ERROR_GENERACION" not in final_abstract_text:
                        print("Abstract generado con éxito.")
                    else:
                        print(f"Fallo al generar abstract para {path}. Saltando documento.")
                        continue

                # 3. CREAR EL DOCUMENTO DE LANGCHAIN
                doc = Document(
                    page_content=abstract_text,
                    metadata={
                        'title': path,
                        'source': str(path)
                    }
                )
                abstract_docs.append(doc)

            except Exception as e:
                print(f"Error fatal procesando {path}: {e}")
                
        return abstract_docs

    

    def get_unique_sources_list(self, persistent_client):
        # Obtén los datos de la colección
        collection_data = persistent_client.get_collection('langchain').get(include=['embeddings', 'documents', 'metadatas'])
        
        # Extrae los metadatos
        metadatas = collection_data['metadatas']
        # Obtén los valores únicos de 'source'
        sources = set()
        for metadata in metadatas:
            source = metadata.get('source', None)
            if source:
                sources.add(source)
        
        # Obtener solo el nombre de archivo de cada ruta
        # file_names = list(set(source.split('/')[-1] for source in sources))
        file_names = list(set(Path(source).name for source in sources))
        files = []
        for file_name in file_names:
            files.append("data/pdf/" + file_name)
            
        return files

    def update_abstract_db(self, abstract_documents, embedding, abstract_db_dir="./chroma_db_abstracts_dir"):
        """
        Actualiza la base de datos de abstracts en Chroma.
        Si la base de datos ya existe, se actualiza con los nuevos documentos.
        """
        # Verifica si el directorio ya existe
        if Path(abstract_db_dir).exists():
            db_abstracts = Chroma(
                embedding_function=embedding,
                persist_directory=abstract_db_dir
            )
            db_abstracts.add_documents(abstract_documents)
        else:
            print(f"No existe la base de datos: '{abstract_db_dir}'")
            # Crea la base de datos
            db_abstracts = Chroma.from_documents(
                documents=abstract_documents,
                embedding=embedding,
                persist_directory=abstract_db_dir
            )
        
        return db_abstracts

    def add_file_to_vectordb(self, filepath, embedding, llm,  full_chroma_db_dir, abtracts_chorma_db_dir):
        abstrabs_doc = self.create_or_extract_abstracts([filepath], llm)
        self.update_abstract_db(abstrabs_doc, embedding, abtracts_chorma_db_dir)
        
        loader = PyMuPDFLoader(filepath)
        print(f"Cargando {filepath}...")
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=500)
        splits = text_splitter.split_documents(docs)
        for i in range(0, len(docs), 10):  # Procesa de 10 en 10
            chunk = splits[i:i + 10]
            vectorstore = Chroma.from_documents(
                documents=chunk,
                embedding=embedding,
                persist_directory=full_chroma_db_dir
            )
        print(f"Procesados {len(chunk)} documentos y guardados en Chroma.") 

    def add_files_to_vectordb(self, persistent_client, embedding, llm, full_chroma_db_dir="./chroma_db_dir", abtracts_chorma_db_dir="./chroma_db_abstracts_dir"):
        data_folder = Path("data/pdf")

        pdf_paths = list(data_folder.glob("*.pdf"))

        cargados = self.get_unique_sources_list(persistent_client)

        nuevos = 0
        for path in pdf_paths:
            if path.as_posix() not in cargados:
                nuevos += 1
                self.add_file_to_vectordb(path.as_posix(), embedding, llm, full_chroma_db_dir, abtracts_chorma_db_dir)
        print(f"Se cargaron {nuevos} nuevos documentos de PDFs.")

if __name__ == "__main__":
    
    persistent_client = chromadb.PersistentClient(path="./chroma_db_dir")

    newFile = create_new_files()
    files = newFile.get_unique_sources_list(persistent_client)
    print(len(files))
