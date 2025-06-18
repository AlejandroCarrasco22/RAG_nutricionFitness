import os
import sys
import pandas as pd
from typing import Dict, List, Any, Union
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from resolvers.end_to_end_resolver import EndToEndResolver
from resolvers.sql_resolver import SQLResolver
from resolvers.rag_resolver import RAGResolver
from routing import RouterRagSql
from metrics.advanced_metrics import AdvancedMetrics

# Datos de evaluación para SQL
sql_evaluation_data = [
    {
        "pregunta": "¿Qué alimentos tienen menos de 10 calorias, dime los 5 con menos?",
        "contexto": "Consulta sobre alimentos bajos en calorías",
        "sql_esperado": 'SELECT "category", "description", "calories" FROM comida WHERE "calories" < 10 ORDER BY "calories" ASC LIMIT 5;',
        "resultado_esperado": """Los alimentos que tienen menos de 10 calorías y que se encuentran en la lista son los siguientes:

            1. **Té caliente, hoja, verde, descafeinado** - 0 calorías
            2. **Té helado, instantáneo, verde, sin azúcar** - 0 calorías
            3. **Té caliente, hibisco** - 0 calorías
            4. **Té helado, preparado, verde, descafeinado, sin azúcar** - 0 calorías
            5. **Té helado, embotellado, negro, sin azúcar** - 0 calorías

            Todos estos tipos de té tienen 0 calorías, lo que los convierte en opciones ideales si buscas bebidas con un contenido calórico muy bajo."""

    },
    {
        "pregunta": "¿Cuántas calorías tiene un aguacate?",
        "contexto": "Consulta sobre calorías en aguacates",
        "sql_esperado": 'SELECT "category", "description", "calories" FROM comida WHERE "description" LIKE \'%Avocado%\';',
        "resultado_esperado": """Un aguacate tiene aproximadamente 174.06 calorías por cada 100 gramos. Este valor puede variar ligeramente dependiendo del tamaño y la variedad del aguacate, pero en general, esa es la cantidad calórica que se puede esperar. Si necesitas información sobre porciones específicas o diferentes preparaciones, no dudes en preguntar."""
    }
]

# Datos de evaluación para RAG
rag_evaluation_data = [
    {
        "pregunta": "¿Para qué sirve la creatina?",
        "contexto": "Información sobre suplementación deportiva",
        "respuesta_esperada": "La creatina es un compuesto natural que se encuentra principalmente en los músculos y cuya función esencial es la producción rápida de energía. Sirve para regenerar el ATP, la principal molécula energética del cuerpo, especialmente durante ejercicios de alta intensidad y corta duración. Su suplementación aumenta la fuerza, la potencia y la masa muscular, mejora el rendimiento deportivo y acelera la recuperación. También puede tener beneficios para la función cerebral y la salud ósea."
    },
    {
        "pregunta": "¿Qué beneficios tiene la cafeína en el deporte?",
        "contexto": "Información sobre suplementación deportiva",
        "respuesta_esperada": "La cafeína en el deporte actúa principalmente como un estimulante del sistema nervioso central, bloqueando los receptores de adenosina y reduciendo la percepción de fatiga y dolor. Esto permite aumentar la resistencia, la fuerza muscular y la potencia, mejorando el rendimiento en deportes aeróbicos y anaeróbicos. Además, puede optimizar la movilización de grasas para ser utilizadas como energía, lo que ayuda a preservar el glucógeno muscular y retrasar el agotamiento. También se ha observado que mejora la concentración, el estado de alerta y el tiempo de reacción. Para obtener estos beneficios, la dosis recomendada suele oscilar entre 3-6 mg/kg de peso corporal, consumida aproximadamente 30-60 minutos antes del ejercicio."
    }
]

def evaluar_sql(resolver: EndToEndResolver, metrics: AdvancedMetrics, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Evalúa las consultas SQL usando las métricas avanzadas"""
    resultados: List[Dict[str, Any]] = []
    
    for item in data:
        try:
            pregunta = item["pregunta"]
            contexto = item["contexto"]
            sql_esperado = item["sql_esperado"]
            resultado_esperado = item["resultado_esperado"]
            
            # Obtener respuesta del sistema
            respuesta = resolver.resolver(pregunta)
            
            # Extraer consulta SQL y resultado
            consulta_generada = ""
            resultado_generado: List[Any] = []
            
            if isinstance(respuesta, dict):
                consulta_generada = respuesta.get("consulta", "")
                resultado_generado = respuesta.get("resultado", [])
            elif isinstance(respuesta, str):
                consulta_generada = respuesta
                resultado_generado = [respuesta]
            
            # Evaluar NLI y Faithfulness
            metricas = metrics.evaluate_response(contexto, str(resultado_generado))
            
            resultados.append({
                "pregunta": pregunta,
                "sql_esperado": sql_esperado,
                "sql_generado": consulta_generada,
                "resultado_esperado": resultado_esperado,
                "resultado_generado": resultado_generado,
                "metricas": metricas
            })
            
        except Exception as e:
            print(f"❌ Error evaluando SQL: {str(e)}")
            resultados.append({
                "pregunta": pregunta,
                "error": str(e)
            })
    
    return resultados

def evaluar_rag(resolver: EndToEndResolver, metrics: AdvancedMetrics, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Evalúa las consultas RAG usando las métricas avanzadas"""
    resultados: List[Dict[str, Any]] = []
    
    for item in data:
        try:
            pregunta = item["pregunta"]
            contexto = item["contexto"]
            respuesta_esperada = item["respuesta_esperada"]
            
            # Obtener respuesta del sistema
            respuesta = resolver.resolver(pregunta)
            
            # Extraer respuesta generada
            respuesta_generada = ""
            
            if isinstance(respuesta, dict):
                respuesta_generada = respuesta.get("answer", str(respuesta))
            else:
                respuesta_generada = str(respuesta)
            
            # Evaluar NLI y Faithfulness
            metricas = metrics.evaluate_response(contexto, respuesta_generada)
            
            resultados.append({
                "pregunta": pregunta,
                "respuesta_esperada": respuesta_esperada,
                "respuesta_generada": respuesta_generada,
                "metricas": metricas
            })
            
        except Exception as e:
            print(f"❌ Error evaluando RAG: {str(e)}")
            resultados.append({
                "pregunta": pregunta,
                "error": str(e)
            })
    
    return resultados

def main():
    try:
        # Inicializar componentes
        router = RouterRagSql()
        sql_resolver = SQLResolver()
        rag_resolver = RAGResolver()
        resolver = EndToEndResolver(router, sql_resolver, rag_resolver)
        metrics = AdvancedMetrics()
        
        # Evaluar SQL
        print("\n🔍 Evaluando consultas SQL...")
        resultados_sql = evaluar_sql(resolver, metrics, sql_evaluation_data)
        
        # Evaluar RAG
        print("\n🔍 Evaluando consultas RAG...")
        resultados_rag = evaluar_rag(resolver, metrics, rag_evaluation_data)
        
        # Guardar resultados
        resultados = {
            "sql": resultados_sql,
            "rag": resultados_rag
        }

        
        # Mostrar resumen
        print("\n📊 Resumen de métricas SQL:")
        for result in resultados_sql:
            if isinstance(result, dict) and "error" in result:
                print(f"\n❌ Error en pregunta: {result.get('pregunta', 'Desconocida')}")
                print(f"Error: {result.get('error', 'Desconocido')}")
                continue
                
            print(f"\nPregunta: {result.get('pregunta', '')}")
            print(f"SQL Esperado: {result.get('sql_esperado', '')}")
            print(f"SQL Generado: {result.get('sql_generado', '')}")
            print("Métricas:")
            metricas = result.get('metricas', {})
            if isinstance(metricas, dict):
                for k, v in metricas.items():
                    print(f"  {k}: {v}")
        
        print("\n📊 Resumen de métricas RAG:")
        for result in resultados_rag:
            if isinstance(result, dict) and "error" in result:
                print(f"\n❌ Error en pregunta: {result.get('pregunta', 'Desconocida')}")
                print(f"Error: {result.get('error', 'Desconocido')}")
                continue
                
            print(f"\nPregunta: {result.get('pregunta', '')}")
            print(f"Respuesta Esperada: {result.get('respuesta_esperada', '')[:100]}...")
            print(f"Respuesta Generada: {result.get('respuesta_generada', '')[:100]}...")
            print("Métricas:")
            metricas = result.get('metricas', {})
            if isinstance(metricas, dict):
                for k, v in metricas.items():
                    print(f"  {k}: {v}")
                
    except Exception as e:
        print(f"❌ Error general en la evaluación: {str(e)}")

if __name__ == "__main__":
    main() 