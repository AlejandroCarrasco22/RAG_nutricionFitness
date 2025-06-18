import evaluate
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class AdvancedMetrics:
    def __init__(self):
        # Cargar modelos de evaluación
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
    def evaluate_response(self, contexto: str, respuesta: str) -> Dict[str, float]:
        """
        Evalúa una respuesta usando métricas avanzadas.
        
        Args:
            contexto: El contexto o documento de referencia
            respuesta: La respuesta generada a evaluar
            
        Returns:
            Dict con las métricas calculadas
        """
        # Evaluar NLI usando similitud de coseno
        context_embedding = self.model.encode(contexto)
        response_embedding = self.model.encode(respuesta)
        
        nli_score = cosine_similarity(
            context_embedding.reshape(1, -1),
            response_embedding.reshape(1, -1)
        )[0][0]
        
        # Evaluar Faithfulness usando similitud de coseno también
        faithfulness_score = nli_score  # Por ahora usamos la misma métrica
        
        return {
            "nli_score": float(nli_score),
            "faithfulness_score": float(faithfulness_score)
        } 