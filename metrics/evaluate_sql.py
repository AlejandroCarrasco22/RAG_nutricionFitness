import pandas as pd
from difflib import SequenceMatcher

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from resolvers.end_to_end_resolver import EndToEndResolver
from resolvers.sql_resolver import SQLResolver
from resolvers.rag_resolver import RAGResolver
from routing import RouterRagSql

# --- DATOS DE EVALUACIÓN DEFINIDOS EN EL SCRIPT ---
evaluacion_data = [
    {
        "pregunta": "¿Qué alimentos tienen menos de 10 calorias, dime los 5 con menos?",
        "sql_esperado": 'SELECT "category", "description", "calories" FROM comida WHERE "calories" < 10 ORDER BY "calories" ASC LIMIT 5;',
        "resultado_esperado": [
            {"category": "Tea", "description": "Tea, hot, leaf, green, decaffeinated","calories": 0.0},
            {"category": "Tea","description": "Tea, iced, instant, green, unsweetened","calories": 0.0},
            {"category": "Tea", "description": "Tea, hot, hibiscus", "calories": 0.0},
            {"category": "Tea","description": "Tea, iced, brewed, green, decaffeinated, unsweetened", "calories": 0.0 },
            {"category": "Tea","description": "Tea, iced, bottled, black, unsweetened", "calories": 0.0}
        ]
    },
    {
        "pregunta": "¿Cuántas calorías tiene un aguacate?",
        "sql_esperado": 'SELECT "category", "description", "calories" FROM comida WHERE "description" LIKE \'%Avocado%\';',
        "resultado_esperado": [
            {"category": "Sushi roll", "description": "Sushi roll, avocado","calories": 91.31},
            {"category": "Avocado","description": "Avocado, raw","calories": 174.06},
            {"category": "Lettuce", "description": "Lettuce, salad with avocado, tomato, and/or carrot, with or without other vegetables, no dressing", "calories": 57.19},
            {"category": "Avocado dressing", "description": "Avocado dressing","calories": 427.01},
            {"category": "Avocado","description": "Avocado, for use on a sandwich","calories": 174.06}
        ]
    },
    {
        "pregunta": "¿Cuántos gramos de proteína hay en 100g de pechuga de pollo? Devuelve 5",
        "sql_esperado": 'SELECT "category", "description", "protein" FROM comida WHERE "description" LIKE \'%Chicken Breast%\' LIMIT 5;',
        "resultado_esperado": [
            {"category": "Chicken breast", "description": "Chicken breast, NS as to cooking method, skin eaten", "protein": 26.37},
            {"category": "Chicken breast", "description": "Chicken breast, NS as to cooking method, skin not eaten", "protein": 28.04},
            {"category": "Chicken breast", "description": "Chicken breast, baked, broiled, or roasted, skin eaten, from raw", "protein": 26.25},
            {"category": "Chicken breast", "description": "Chicken breast, baked, broiled, or roasted, skin not eaten, from raw", "protein": 30.22},
            {"category": "Chicken breast", "description": "Chicken breast, baked or broiled, skin eaten, from pre-cooked", "protein": 26.37}
        ]
    },
    {
        "pregunta": "¿Cuál es el contenido de hierro en las lentejas? Devuelve 5",
        "sql_esperado": 'SELECT "category", "description", "major_minerals.iron" FROM comida WHERE "description" LIKE \'%Lentils%\' LIMIT 5;',
        "resultado_esperado": [
            {"category": "Lentils", "description": "Lentils, NFS", "major_minerals.iron": 3.11},
            {"category": "Lentils", "description": "Lentils, from dried, fat added", "major_minerals.iron": 3.11},
            {"category": "Lentils", "description": "Lentils, from dried, no added fat", "major_minerals.iron": 3.31},
            {"category": "Lentils", "description": "Lentils, from canned", "major_minerals.iron": 3.10},
            {"category": "Pasta with tomato-based sauce and beans or lentils", "description": "Pasta with tomato-based sauce and beans or lentils", "major_minerals.iron": 1.73}
        ]
    },
    {
        "pregunta": "Dime los microgramos de Vitamina B12 que tiene el salmón. Devuelve 5",
        "sql_esperado": 'SELECT "category", "description", "vitamins.vitamin_b12" FROM comida WHERE "description" LIKE \'%Salmon%\' LIMIT 5;',
        "resultado_esperado": [
            {"category": "Salmon", "description": "Salmon, raw", "vitamins.vitamin_b12": 4.15},
            {"category": "Salmon", "description": "Salmon, cooked, NS as to cooking method", "vitamins.vitamin_b12": 4.57},
            {"category": "Salmon", "description": "Salmon, baked or broiled, made with oil", "vitamins.vitamin_b12": 4.57},
            {"category": "Salmon", "description": "Salmon, baked or broiled, made with butter", "vitamins.vitamin_b12": 4.57},
            {"category": "Salmon", "description": "Salmon, baked or broiled, made with margarine", "vitamins.vitamin_b12": 4.57}
        ]
    },
    {
        "pregunta": "Enumera los 5 alimentos con más potasio de tu base de datos.",
        "sql_esperado": 'SELECT "category", "description", "major_minerals.potassium" FROM comida ORDER BY "major_minerals.potassium" DESC LIMIT 5;',
        "resultado_esperado": [
            {"category": "Tea", "description": "Tea, iced, instant, black, unsweetened, dry", "major_minerals.potassium": 6040.0},
            {"category": "Coffee", "description": "Coffee, instant, not reconstituted", "major_minerals.potassium": 3535.0},
            {"category": "Coffee", "description": "Coffee, instant, 50% less caffeine, not reconstituted", "major_minerals.potassium": 3535.0},
            {"category": "Coffee", "description": "Coffee, instant, decaffeinated, not reconstituted", "major_minerals.potassium": 3501.0},
            {"category": "Sun-dried tomatoes", "description": "Sun-dried tomatoes", "major_minerals.potassium": 3427.0}
        ]
    },
    {
        "pregunta": "¿Qué tiene más grasa saturada, el queso cheddar o la mantequilla?",
        "sql_esperado": 'SELECT "category", "description", "fat.saturated_fat" FROM comida WHERE "description" LIKE \'%Cheddar Cheese%\' OR "description" LIKE \'%Butter%\' ORDER BY "fat.saturated_fat" DESC LIMIT 1;',
        "resultado_esperado": [
            {"category": "Ghee", "description": "Ghee, clarified butter", "fat.saturated_fat": 61.924}
        ]
    }
]


router = RouterRagSql()
sql_resolver = SQLResolver()
rag_resolver = RAGResolver()
resolver = EndToEndResolver(router, sql_resolver, rag_resolver)

def comparar_sql(generado, esperado):
    """Similitud entre SQL generada y esperada (0 a 1)"""
    return SequenceMatcher(None, generado.strip().lower(), esperado.strip().lower()).ratio()

def comparar_resultado(df_generado, df_esperado):
    """Compara resultados SQL sin importar orden"""
    try:
        df_generado = pd.DataFrame(df_generado)
        df_esperado = pd.DataFrame(df_esperado)
        return df_generado.reset_index(drop=True).equals(df_esperado.reset_index(drop=True))
    except Exception as e:
        print(f"❌ Error comparando resultados: {e}")
        return False

resultados = []

for i, item in enumerate(evaluacion_data):
    pregunta = item["pregunta"]
    sql_esperado = item["sql_esperado"]
    resultado_esperado = item["resultado_esperado"]

    print(f"\n🔍 Evaluando pregunta {i+1}: {pregunta}")
    print(f"   SQL esperado: {sql_esperado}")
    print(f"   Resultado esperado: {resultado_esperado}")

    try:

        output = sql_resolver.resolver(pregunta)
        print(output)

        consulta_generada = output["consulta"]
        resultado_generado = output["resultado"]

        similitud_sql = comparar_sql(consulta_generada, sql_esperado)
        resultado_ok = comparar_resultado(resultado_generado, resultado_esperado)

        resultados.append({
            "pregunta": pregunta,
            "sql_esperado": sql_esperado,
            "sql_generado": consulta_generada,
            "similitud_sql": similitud_sql,
            "resultado_esperado": resultado_esperado,
            "resultado_generado": resultado_generado,
            "resultado_correcto": resultado_ok
        })

        print(f"✅ Similitud SQL: {similitud_sql:.2f} | Resultado correcto: {resultado_ok}")

    except Exception as e:
        print(f"❌ Error evaluando: {e}")
        resultados.append({
            "pregunta": pregunta,
            "error": str(e)
        })

# --- RESUMEN ---
comparables = [r for r in resultados if "resultado_correcto" in r]
correctos = sum(r["resultado_correcto"] for r in comparables)
similitudes = [r["similitud_sql"] for r in resultados if "similitud_sql" in r]

print("\n--- RESULTADO GLOBAL ---")
print(f"Total evaluadas: {len(evaluacion_data)}")
print(f"Consultas SQL correctas (resultado): {correctos}/{len(comparables)}")
if similitudes:
    print(f"Similitud media SQL: {sum(similitudes)/len(similitudes):.2f}")
