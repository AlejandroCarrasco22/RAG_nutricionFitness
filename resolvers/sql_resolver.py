import sqlite3
import pandas as pd
from langchain.chains import LLMChain
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from fuzzywuzzy import process

class SQLResolver:
    def __init__(self, llm, db_path="../comida.db"):
        self.llm = llm
        self.db_path = db_path
        self.unique_categories = self.get_unique_categories()
        
        # Inicializa las chains
        self.sql_chain = self._build_sql_chain()
        self.translator_chain = self._build_translator_chain()
        
    
    def get_unique_categories(self):
        try:
            with sqlite3.connect(self.db_path) as conn:
                query = 'SELECT DISTINCT "category" FROM comida'
                return pd.read_sql_query(query, conn)["category"].tolist()
        except Exception as e:
            print(f"Error al cargar categorías: {e}")
            return []

    def _build_sql_chain(self):
        system_prompt = SystemMessagePromptTemplate.from_template("""Eres un asistente experto en bases de datos SQLite.

            Tu tarea es generar **ÚNICAMENTE una query SQL válida**, sin explicaciones, sin texto adicional y sin formato markdown (como triples comillas ```, la palabra 'sql', o cualquier otro envoltorio), a partir de una pregunta del usuario.

            Trabajas con una tabla llamada `comida`, el contenido de las tablas de texto viene en inglés, con las siguientes columnas. **¡IMPORTANTE! Cuando uses los nombres de las columnas que contienen puntos o guiones, DEBES ENVOLVERLOS ENTRE COMILLAS DOBLES (")** para que la consulta sea válida en SQLite.

            - "category" (TEXT) - Categoría general del alimento. **Contiene una gran variedad de valores únicos.**
            - "description" (TEXT) - Descripción completa del alimento y sus subcategorías (ej. 'Milk, human', 'Buttermilk, low fat (1%)').
            - "alpha_carotene" (REAL) - Pigmento vegetal. Unidad: microgramos (mcg).
            - "beta_carotene" (REAL) - Pigmento vegetal. Unidad: microgramos (mcg).
            - "beta_cryptoxanthin" (REAL) - Pigmento vegetal. Unidad: microgramos (mcg).
            - "carbohydrate" (REAL) - Azúcares y almidones. Unidad: gramos (g).
            - "cholesterol" (REAL) - Presente solo en productos animales. Unidad: miligramos (mg).
            - "choline" (REAL) - Nutriente esencial. Unidad: miligramos (mg).
            - "fiber" (REAL) - Parte vegetal no digerible. Unidad: gramos (g).
            - "lutein_and_zeaxanthin" (REAL) - Pigmento vegetal. Unidad: microgramos (mcg).
            - "lycopene" (REAL) - Pigmento vegetal. Unidad: microgramos (mcg).
            - "niacin" (REAL) - Vitamina B. Unidad: miligramos (mg).
            - "protein" (REAL) - Esencial para tejidos. Unidad: gramos (g).
            - "retinol" (REAL) - Vitamina A activa. Unidad: microgramos (mcg).
            - "riboflavin" (REAL) - Vitamina B. Unidad: miligramos (mg).
            - "selenium" (REAL) - Antioxidante. Unidad: microgramos (mcg).
            - "sugar_total" (REAL) - Azúcares simples. Unidad: gramos (g).
            - "thiamin" (REAL) - Vitamina B. Unidad: miligramos (mg).
            - "water" (REAL) - Cantidad de agua. Unidad: gramos (g).
            - "fat.monosaturated_fat" (REAL) - Grasas monoinsaturadas. Unidad: gramos (g).
            - "fat.polysaturated_fat" (REAL) - Grasas poliinsaturadas. Unidad: gramos (g).
            - "fat.saturated_fat" (REAL) - Grasas saturadas. Unidad: gramos (g).
            - "fat.total_lipid" (REAL) - Suma de todas las grasas. Unidad: gramos (g).
            - "major_minerals.calcium" (REAL) - Mineral. Unidad: miligramos (mg).
            - "major_minerals.copper" (REAL) - Mineral. Unidad: miligramos (mg).
            - "major_minerals.iron" (REAL) - Mineral. Unidad: miligramos (mg).
            - "major_minerals.magnesium" (REAL) - Mineral. Unidad: miligramos (mg).
            - "major_minerals.phosphorus" (REAL) - Mineral. Unidad: miligramos (mg).
            - "major_minerals.potassium" (REAL) - Mineral. Unidad: miligramos (mg).
            - "major_minerals.sodium" (REAL) - Mineral. Unidad: miligramos (mg).
            - "major_minerals.zinc" (REAL) - Mineral. Unidad: miligramos (mg).
            - "vitamins.vitamin_a_-_rae" (REAL) - Vitamina A. Unidad: microgramos (mcg).
            - "vitamins.vitamin_b12" (REAL) - Vitamina B12. Unidad: microgramos (mcg).
            - "vitamins.vitamin_b6" (REAL) - Vitamina B6. Unidad: miligramos (mg).
            - "vitamins.vitamin_c" (REAL) - Vitamina C. Unidad: miligramos (mg).
            - "vitamins.vitamin_e" (REAL) - Vitamina E. Unidad: miligramoscg).
            - "vitamins.vitamin_k" (REAL) - Vitamina K. Unidad: microgramos (mcg).
            - "calories" (REAL) - Calorías por 100g.

            GUIDELINES:

            **1. Cuando una columna específica (como calorías, proteína, etc.) sea solicitada, SIEMPRE incluye también las columnas "category" y "description" en la cláusula SELECT.**
            **2. Para preguntas que solicitan una propiedad general (ej. "alimentos con X calorías", "alimentos con más Y"), enfócate en la condición numérica o de ordenación en el WHERE/ORDER BY. Solo usa filtros de "category" o "description" (LIKE) si la pregunta especifica un TIPO de alimento a buscar.**
            **3. Devuelve ÚNICAMENTE una consulta SQL válida. No incluyas ningún texto extra, explicaciones o formato markdown (como triples comillas ```, la palabra 'sql', o cualquier otro envoltorio).**
            **4. Para filtrar por alimentos específicos, usa la columna "description" con el operador LIKE y el texto del alimento CAPITALIZADO (Camel Case) en inglés, dado que la base de datos está en inglés, por ejemplo: LIKE '%Chicken Breast%' o LIKE '%Avocado%'.**
            **5. Para filtrar por categorías generales de alimentos, usa la columna "category" con el operador '=' y el valor exacto de la categoría (en Camel Case) en inglés.**
            **6. Recuerda que los nombres de las columnas que contienen puntos o guiones DEBEN ENVOLVERSE ENTRE COMILLAS DOBLES (") en la consulta SQL (ej. "fat.total_lipid", "vitamins.vitamin_a_-_rae").**
            """
        )
        human_prompt = HumanMessagePromptTemplate.from_template("""{contexto}\nPregunta: {pregunta}""")
        prompt = ChatPromptTemplate.from_messages([system_prompt, human_prompt])
        return LLMChain(llm=self.llm, prompt=prompt)

    def _build_translator_chain(self):
        system_prompt = SystemMessagePromptTemplate.from_template(
            "Traduce el siguiente término de alimento o categoría del español a su equivalente en inglés más común, adecuado para una consulta de base de datos. Responde ÚNICAMENTE con el término en inglés, sin explicaciones ni texto adicional."
        )
        human_prompt = HumanMessagePromptTemplate.from_template("{term_spanish}")
        prompt = ChatPromptTemplate.from_messages([system_prompt, human_prompt])
        return LLMChain(llm=self.llm, prompt=prompt)
    
    def resolver_sql(self, pregunta: str):
        pregunta_normalizada = pregunta.lower()
        translated_phrase = self.translator_chain.run(term_spanish=pregunta).strip()

        is_specific_food_query = any(cat.lower() in translated_phrase.lower() for cat in self.unique_categories)
        
        general_keywords = [
            "calorias", "carbohidratos", "colesterol", "colina", "fibra", "niacina", 
            "proteina", "retinol", "riboflavina", "selenio", "azucar total", "tiamina", 
            "agua", "grasa monosaturada", "grasa polisaturada", "grasa saturada", 
            "lipido total", "calcio", "cobre", "hierro", "magnesio", "fosforo", 
            "potasio", "sodio", "zinc", "vitamina a", "vitamina b12", "vitamina b6", 
            "vitamina c", "vitamina e", "vitamina k", "alfa caroteno", "beta caroteno", 
            "beta criptoxantina", "luteina y zeaxantina", "licopeno"
        ]
        is_general_query = any(k in pregunta_normalizada for k in general_keywords) and not is_specific_food_query

        contexto_pistas = ""
        matched_cats = []
        matched_descs = []

        if is_specific_food_query:
            for cat in self.unique_categories:
                if cat.lower() in translated_phrase.lower():
                    matched_cats.append(cat)
                else:
                    score = process.extractOne(translated_phrase.lower(), [cat.lower()])
                    if score and score[1] > 90:
                        matched_cats.append(cat)

            camel = ' '.join(word.capitalize() for word in translated_phrase.split())
            matched_descs.append(camel)
        

        if matched_cats:
            contexto_pistas += f"El usuario podría referirse a las siguientes categorías: {', '.join(matched_cats[:3])}. "
        if matched_descs:
            contexto_pistas += f"El usuario podría referirse a alimentos como: {', '.join(matched_descs[:5])}. "

        consulta_sql = self.sql_chain.run(
            pregunta=pregunta,
            contexto=contexto_pistas
        ).strip()

        consulta_sql = self._clean_sql(consulta_sql)

        try:
            with sqlite3.connect(self.db_path) as conn:
                return pd.read_sql_query(consulta_sql, conn)
        except Exception as e:
            return f"Error ejecutando la consulta:\n{e}"

    def _clean_sql(self, consulta):
        if consulta.startswith("```sql"):
            return consulta.strip("```sql").strip("```").strip()
        elif consulta.startswith("```"):
            return consulta.strip("```").strip()
        return consulta.strip()