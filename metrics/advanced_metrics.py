import evaluate
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from ragas.metrics import faithfulness, answer_relevancy
from ragas import evaluate as ragas_evaluate
from datasets import Dataset
import pandas as pd
import re

class AdvancedMetrics:
    def __init__(self):
        # Cargar modelos de evaluación
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
    def extract_entities_simple(self, text: str) -> List[str]:
        """Extrae entidades nombradas usando patrones simples"""
        # Patrones para identificar entidades comunes en nutrición y deporte
        patterns = [
            r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b',  # Palabras capitalizadas
            r'\b(?:creatina|cafeína|proteína|vitamina|magnesio|antioxidantes|BCAA|glutamina|electrolitos|omega-3|DHA|EPA)\b',
            r'\b(?:ATP|BCAA|DHA|EPA)\b',  # Acrónimos
            r'\b\d+(?:\.\d+)?\s*(?:mg|g|kg|ml|kcal|calorías)\b',  # Cantidades con unidades
        ]
        
        entities = []
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            entities.extend([match.lower() for match in matches])
        
        return list(set(entities))  # Eliminar duplicados
    
    def extract_key_phrases_simple(self, text: str) -> List[str]:
        """Extrae frases clave usando patrones gramaticales simples"""
        # Dividir en oraciones
        sentences = re.split(r'[.!?]+', text)
        key_phrases = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence.split()) < 3:
                continue
                
            # Buscar patrones de frases clave
            patterns = [
                r'\b(?:es|son|actúa|ayuda|mejora|aumenta|reduce|previene|optimiza)\s+\w+(?:\s+\w+)*\b',
                r'\b(?:para|durante|tras|antes|después)\s+\w+(?:\s+\w+)*\b',
                r'\b(?:fundamental|esencial|clave|importante|crucial)\s+\w+(?:\s+\w+)*\b',
                r'\b(?:entre|aproximadamente|alrededor de)\s+\d+(?:\.\d+)?\s*\w+\b',
            ]
            
            for pattern in patterns:
                matches = re.findall(pattern, sentence, re.IGNORECASE)
                key_phrases.extend([match.lower() for match in matches])
        
        return list(set(key_phrases))
    
    def calculate_nli_score(self, contexto: str, respuesta: str) -> float:
        """
        Calcula el score NLI basado en similitud semántica y coherencia lógica
        """
        # Calcular similitud semántica usando embeddings
        context_embedding = self.model.encode(contexto)
        response_embedding = self.model.encode(respuesta)
        
        semantic_similarity = cosine_similarity(
            context_embedding.reshape(1, -1),
            response_embedding.reshape(1, -1)
        )[0][0]
        
        # Extraer entidades y verificar coherencia
        context_entities = set(self.extract_entities_simple(contexto))
        response_entities = set(self.extract_entities_simple(respuesta))
        
        if len(context_entities) == 0:
            entity_coherence = 1.0
        else:
            entity_overlap = len(context_entities.intersection(response_entities)) / len(context_entities)
            entity_coherence = entity_overlap
        
        # Combinar similitud semántica y coherencia de entidades
        nli_score = 0.7 * semantic_similarity + 0.3 * entity_coherence
        
        return float(nli_score)
    
    def calculate_faithfulness_score(self, contexto: str, respuesta: str) -> float:
        """
        Calcula el score de Faithfulness basado en la verificación de afirmaciones
        """
        # Extraer frases clave del contexto
        context_phrases = self.extract_key_phrases_simple(contexto)
        response_phrases = self.extract_key_phrases_simple(respuesta)
        
        if len(context_phrases) == 0:
            return 0.5  # Score neutral si no hay frases clave
        
        # Calcular qué porcentaje de frases clave del contexto están presentes en la respuesta
        supported_phrases = 0
        for context_phrase in context_phrases:
            # Verificar si la frase clave está presente (con cierta tolerancia)
            phrase_words = context_phrase.split()
            if len(phrase_words) >= 2:
                # Buscar al menos 2 palabras de la frase clave
                matching_words = sum(1 for word in phrase_words if word in respuesta.lower())
                if matching_words >= min(2, len(phrase_words)):
                    supported_phrases += 1
        
        faithfulness_score = supported_phrases / len(context_phrases)
        
        # Penalizar si hay muchas frases en la respuesta que no están en el contexto
        if len(response_phrases) > 0:
            extra_phrases = len(response_phrases) - supported_phrases
            if extra_phrases > len(context_phrases):
                faithfulness_score *= 0.8  # Penalización del 20%
        
        return float(max(0.0, min(1.0, faithfulness_score)))
        
    def evaluate_response(self, contexto: str, respuesta: str) -> Dict[str, float]:
        """
        Evalúa una respuesta usando métricas avanzadas de ragas.
        
        Args:
            contexto: El contexto o documento de referencia
            respuesta: La respuesta generada a evaluar
            
        Returns:
            Dict con las métricas calculadas
        """
        try:
            # Crear dataset para ragas
            data = {
                "question": ["Evaluación de respuesta"],
                "answer": [respuesta],
                "contexts": [[contexto]]
            }
            dataset = Dataset.from_dict(data)
            
            # Evaluar con ragas
            results = ragas_evaluate(
                dataset,
                metrics=[
                    faithfulness,
                    answer_relevancy
                ]
            )
            
            # Extraer scores
            faithfulness_score = results['faithfulness']
            answer_relevancy_score = results['answer_relevancy']
            
            # Calcular NLI score usando similitud semántica como fallback
            context_embedding = self.model.encode(contexto)
            response_embedding = self.model.encode(respuesta)
            
            nli_score = cosine_similarity(
                context_embedding.reshape(1, -1),
                response_embedding.reshape(1, -1)
            )[0][0]
            
            return {
                "nli_score": float(nli_score),
                "faithfulness_score": float(faithfulness_score)
            }
            
        except Exception as e:
            print(f"Error usando ragas, usando fallback: {e}")
            # Fallback a implementación simple si ragas falla
            return self._fallback_evaluation(contexto, respuesta)
    
    def _fallback_evaluation(self, contexto: str, respuesta: str) -> Dict[str, float]:
        """
        Evaluación de fallback usando similitud de coseno
        """
        context_embedding = self.model.encode(contexto)
        response_embedding = self.model.encode(respuesta)
        
        similarity_score = cosine_similarity(
            context_embedding.reshape(1, -1),
            response_embedding.reshape(1, -1)
        )[0][0]
        
        return {
            "nli_score": float(similarity_score),
            "faithfulness_score": float(similarity_score * 0.8)  # Faithfulness ligeramente más estricto
        } 