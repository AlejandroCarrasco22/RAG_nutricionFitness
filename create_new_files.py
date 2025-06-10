from langchain_chroma import Chroma 
from langchain.docstore.document import Document
from langchain_community.document_loaders import PyMuPDFLoader
from pathlib import Path
import re
from langchain_text_splitters import RecursiveCharacterTextSplitter


class create_new_files:
    """
    Clase para añadir archivos a la base de datos de vectores.
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
        
        # Un prompt de alta calidad para la tarea de resumir
        summarization_prompt_template = """
        Eres un asistente experto en la síntesis de documentos técnicos y académicos.
        A continuación, te proporciono el texto de un documento. Tu tarea es generar un
        resumen conciso (un 'abstract') en español, de no más de 250 palabras.

        El resumen debe capturar las ideas principales, la metodología (si la hay) y
        las conclusiones clave del texto. Los que hablen de suplementación añade en el resumen toda 
        la suplementación de la que se hable. Debe ser denso en información y estar
        escrito en un estilo formal y objetivo.

        Texto del documento:
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

                if start_match:
                    start_index = start_match.end()
                    end_match = re.search(r'\b(Introduction|Introducción|Keywords|Palabras clave)\b', full_text[start_index:], re.IGNORECASE)
                    if end_match:
                        end_index = start_index + end_match.start()
                        abstract_text = full_text[start_index:end_index].strip()
                    else:
                        abstract_text = full_text[start_index:start_index+2500].strip()

                # 2. SI NO SE EXTRAJO, GENERARLO
                if not abstract_text or len(abstract_text) < 100: # Si es muy corto, probablemente no sea un buen abstract
                    print(f"No se encontró abstract en '{path}'. Generando uno nuevo con Azure OpenAI...")

                    # gpt-4o-mini tiene 128k tokens
                    # 4 caracteres ~= 1 token. 100,000 caracteres ~= 25,000 tokens, más que seguro.
                    max_chars = 100000
                    truncated_text = full_text[:max_chars]

                    prompt_para_resumir = summarization_prompt_template.format(document_text=truncated_text)
                    
                    generated_abstract = self.generate_from_azure_openai(prompt_para_resumir, llm_instance)
                    
                    if "ERROR_GENERACION" not in generated_abstract:
                        abstract_text = generated_abstract
                        print("Abstract generado con éxito.")
                    else:
                        print(f"Fallo al generar abstract para {path}. Saltando documento.")
                        continue # Saltamos al siguiente documento
                else:
                    print("Abstract extraído con éxito del documento.")

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