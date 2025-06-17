from keys import LLM
from langchain_core.prompts import ChatPromptTemplate

class RouterRagSql:
    def __init__(self):
        self.llm = LLM
        self.routing_prompt = ChatPromptTemplate.from_template(
            """Eres un sistema experto que enruta una pregunta de nutrición a la fuente de datos correcta.

                Clasifica cada pregunta en una de estas dos opciones:
                - "sql" → si la pregunta es sobre la composición nutricional específica de uno o más alimentos. Esto incluye consultas sobre cantidades exactas de nutrientes (ej. gramos de proteína, miligramos de vitamina C), calorías, minerales o grasas en un alimento. También se incluyen comparaciones directas o rankings de alimentos basados en estos datos numéricos.
                - "rag" → si la pregunta busca una explicación, recomendación, o información contextual sobre nutrición, suplementación o fitness. Esto abarca preguntas sobre la función de un suplemento (ej. "¿para qué sirve la creatina?"), beneficios, dosis recomendadas, dietas para objetivos específicos (ej. "dieta para diabetes tipo 2"), o consejos basados en un perfil de usuario (ej. "suplementos para un corredor").

                Responde únicamente con la palabra: sql o rag.

                Pregunta: {pregunta}
                """
        )
        self.routing_chain = self.routing_prompt | self.llm
    
    def router(self, pregunta):
        try:
            response = self.routing_chain.invoke({"pregunta": pregunta})
            # El resultado ahora viene directamente en el contenido del mensaje
            tipo = response.content.strip().lower()
            print(f"Tipo de consulta inferido: {tipo}")
            
            if tipo == "sql":
                return tipo
            elif tipo == "rag":
                return tipo
            else:
                return f"Ruta desconocida: '{tipo}'"
        except Exception as e:
            return f"Error al decidir la ruta: {e}"