from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_chroma import Chroma
from pathlib import Path
from keys import LLM, EMBBEDING, CHROMA_ABSTRACTS_DB_DIR, CHROMA_DB_DIR
from langchain.memory import ConversationBufferMemory

class RAGResolver:
    def __init__(self,
                 llm=LLM,
                 embedding=EMBBEDING, 
                 db_full_path=CHROMA_DB_DIR, 
                 db_abstracts_path=CHROMA_ABSTRACTS_DB_DIR,
                 memory=None):
        self.llm = llm
        self.embedding = embedding
        self.db_full_path = Path(db_full_path).resolve()
        self.db_abstracts_path = Path(db_abstracts_path).resolve()
        self.memory = memory
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", """Tú eres un asistente experto en nutrición deportiva, suplementación y entrenamiento
            físico, diseñado para responder preguntas con la máxima precisión. Utiliza los siguientes fragmentos 
            de contexto recuperado para responder a la pregunta del usuario. Si la información no está presente 
            en el contexto proporcionado, indica claramente que no tienes la respuesta, respondiendo No sé. Limita tu respuesta a un 
            máximo de cinco oraciones, asegurando que sea concisa y directamente relevante para el ámbito del 
            deporte y la salud.

            {context}"""),
            ("human", "{input}"),
        ])
        self.negative_keywords = [
            "no tengo la respuesta", 
            "no se menciona", 
            "no encuentro información",
            "información no está presente",
            "contexto proporcionado no",
            "no sé la respuesta",
            "no sé"
        ]

    def resolver(self, pregunta: str):
        """
        Responde una pregunta utilizando un RAG de dos pasos.
        """
        print("Paso 1: Buscando en la base de datos de abstracts...")
        db_abstracts = Chroma(persist_directory=str(self.db_abstracts_path), embedding_function=self.embedding)
        retriever_abstracts = db_abstracts.as_retriever(search_kwargs={"k": 5})
        relevant_abstracts = retriever_abstracts.invoke(pregunta)

        relevant_sources = list(set([doc.metadata['source'] for doc in relevant_abstracts]))
        if not relevant_sources:
            return {"answer": "No pude encontrar documentos relevantes en los resúmenes para responder la pregunta."}

        print(f"Documentos relevantes encontrados: {relevant_sources}")
        
        print("Paso 2: Buscando en la base de datos principal...")
        db_full = Chroma(persist_directory=str(self.db_full_path), embedding_function=self.embedding)
        retriever_full_filtered = db_full.as_retriever(
            search_kwargs={
                "k": 8,
                "filter": {"source": {"$in": relevant_sources}}
            }
        )

        chain = create_stuff_documents_chain(self.llm, self.prompt_template)
        rag_chain_filtered = create_retrieval_chain(retriever_full_filtered, chain)
        result = rag_chain_filtered.invoke({"input": pregunta})

        if any(k in result['answer'].lower() for k in self.negative_keywords):
            print("PASO 3: Realizando búsqueda de respaldo en toda la base de datos...")
            retriever_full = db_full.as_retriever(search_kwargs={"k": 10})
            rag_chain_full = create_retrieval_chain(retriever_full, chain)
            result = rag_chain_full.invoke({"input": pregunta})

        return result

