from langchain_core.prompts import ChatPromptTemplate
import pandas as pd
from keys import LLM 
from langchain.memory import ConversationBufferMemory
from typing import Union, Dict, List, Any

class EndToEndResolver:
    def __init__(self, router, sql_resolver, rag_resolver, llm = LLM, memory=None):
        self.router = router
        self.sql_resolver = sql_resolver
        self.rag_resolver = rag_resolver
        self.llm = llm
        # Inicializa la memoria de conversación si no se pasa una
        self.memory = memory or ConversationBufferMemory(return_messages=True)
        
        # Prompt para generar respuestas naturales a partir de datos estructurados (SQL)
        self.natural_response_prompt = ChatPromptTemplate.from_template(
            """Eres un asistente experto en nutrición y alimentación que responde preguntas sobre 
            la composición nutricional de los alimentos.

            Historial de la conversación:
            {historial}

            Responde de forma clara, informativa y natural basándote únicamente en el siguiente contexto:

            {contexto}

            Pregunta del usuario: {pregunta}

            GUIDELINES:
            - Si los datos muestran valores numéricos, interprétalos y menciona las unidades apropiadas.
            - Sé específico con las cantidades y nombres de alimentos.
            - Si hay múltiples resultados, puedes ordenarlos o filtrarlos según la relevancia para la pregunta.
            - Usa un tono profesional pero accesible.
            """
        )
        
        self.natural_response_chain = self.natural_response_prompt | self.llm

    def _actualizar_memoria(self, pregunta: str, respuesta: str) -> None:
        # Añade el turno a la memoria
        self.memory.chat_memory.add_user_message(pregunta)
        self.memory.chat_memory.add_ai_message(respuesta)

    def resolver(self, pregunta: str) -> Union[str, Dict[str, Any]]:
        """
        Función principal que coordina todo el flujo de resolución:
        1. Enrutamiento (SQL vs RAG)
        2. Recuperación de información
        3. Generación de respuesta natural
        
        Returns:
            Union[str, Dict[str, Any]]: Puede devolver:
                - str: Respuesta directa para consultas SQL
                - Dict[str, Any]: Para consultas RAG, contiene:
                    - answer: str - La respuesta generada
                    - retrieved_docs: List[str] - Documentos recuperados
        """
        try:
            # Paso 1: Enrutamiento
            tipo = self.router.router(pregunta)
            
            # Paso 2: Recuperación según tipo
            if tipo == "sql":
                resultado = self.sql_resolver.resolver(pregunta)
                consulta_sql = resultado.get("consulta", "")
                datos = resultado.get("resultado", [])

                if isinstance(datos, list):
                    # Convertir resultado SQL a contexto textual
                    contexto = pd.DataFrame(datos).to_markdown(index=False)
                    historial = self.memory.buffer if hasattr(self.memory, 'buffer') else ""
                    
                    # Paso 3: Generar respuesta natural para datos estructurados
                    respuesta_final = self.natural_response_chain.invoke({
                        "pregunta": pregunta,
                        "contexto": contexto,
                        "historial": historial
                    })
                    respuesta_texto = str(respuesta_final.content)
                    self._actualizar_memoria(pregunta, respuesta_texto)
                    return respuesta_texto
                else:
                    # Si es un error u otro tipo de respuesta, devolverlo directamente
                    respuesta_texto = str(datos)
                    self._actualizar_memoria(pregunta, respuesta_texto)
                    return respuesta_texto
                    
            elif tipo == "rag":
                # Para RAG, ya tenemos respuesta natural directamente
                resultado = self.rag_resolver.resolver(pregunta)
                respuesta = str(resultado["answer"])
                documentos_recuperados = resultado.get("retrieved_docs", [])
                self._actualizar_memoria(pregunta, respuesta)
                return {
                    "answer": respuesta,
                    "retrieved_docs": documentos_recuperados
                }
                
            else:
                error_msg = f"Error: Tipo de consulta no reconocido ({tipo})"
                self._actualizar_memoria(pregunta, error_msg)
                return error_msg
                
        except Exception as e:
            error_msg = f"Error en el proceso de resolución: {str(e)}"
            self._actualizar_memoria(pregunta, error_msg)
            return error_msg