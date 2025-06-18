import evaluate
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from resolvers.rag_resolver import RAGResolver

# Referencias (respuestas esperadas, por ejemplo humanas o correctas)
reference_answers = [
    "La creatina es un compuesto natural que se encuentra principalmente en los músculos y cuya función esencial es la producción rápida de energía. Sirve para regenerar el ATP, la principal molécula energética del cuerpo, especialmente durante ejercicios de alta intensidad y corta duración. Su suplementación aumenta la fuerza, la potencia y la masa muscular, mejora el rendimiento deportivo y acelera la recuperación. También puede tener beneficios para la función cerebral y la salud ósea.",
    "La cafeína en el deporte actúa principalmente como un estimulante del sistema nervioso central, bloqueando los receptores de adenosina y reduciendo la percepción de fatiga y dolor. Esto permite aumentar la resistencia, la fuerza muscular y la potencia, mejorando el rendimiento en deportes aeróbicos y anaeróbicos. Además, puede optimizar la movilización de grasas para ser utilizadas como energía, lo que ayuda a preservar el glucógeno muscular y retrasar el agotamiento. También se ha observado que mejora la concentración, el estado de alerta y el tiempo de reacción. Para obtener estos beneficios, la dosis recomendada suele oscilar entre 3-6 mg/kg de peso corporal, consumida aproximadamente 30-60 minutos antes del ejercicio.",
    "Para optimizar la recuperación muscular tras entrenamientos intensos, los suplementos clave incluyen la proteína de suero para la reparación muscular rápida, carbohidratos de rápida absorción para reponer las reservas de glucógeno y energía, y la creatina que, además de mejorar el rendimiento, acelera la resíntesis de ATP. Los BCAA y la glutamina contribuyen a la síntesis proteica y reducen el daño muscular, mientras que los electrolitos son esenciales para mantener la hidratación y la función muscular adecuadas, reponiendo los minerales perdidos durante el ejercicio.",
    "La suplementación con magnesio es crucial para el rendimiento deportivo, ya que este mineral interviene en más de 300 reacciones bioquímicas. Es fundamental para la producción de ATP (energía), la contracción y relajación muscular, y la síntesis de proteínas. Ayuda a prevenir calambres y espasmos, reduce la fatiga y el daño muscular, y mejora la recuperación. Además, optimiza la función nerviosa y la calidad del sueño, factores esenciales para un rendimiento óptimo y una recuperación eficaz en atletas.",
    "La evidencia científica sobre la suplementación con antioxidantes en atletas de resistencia es compleja y, en ocasiones, contradictoria. Si bien el ejercicio intenso genera estrés oxidativo que puede contribuir a la fatiga y al daño muscular, el uso de suplementos antioxidantes ha mostrado resultados mixtos en la mejora directa del rendimiento. Algunos estudios sugieren que dosis elevadas de antioxidantes como las vitaminas C y E podrían incluso atenuar las adaptaciones positivas al entrenamiento. Sin embargo, una dieta rica en antioxidantes naturales es fundamental para la salud general del atleta y para apoyar los sistemas antioxidantes endógenos del cuerpo, que son la primera línea de defensa.",
    "La mejor estrategia de hidratación para corredores de larga distancia implica una combinación de pre-hidratación, ingesta durante la carrera y recuperación post-esfuerzo. Es crucial comenzar bien hidratado, consumiendo aproximadamente 400-500 ml de agua o bebida con electrolitos unas dos horas antes de la carrera. Durante el recorrido, se recomienda alternar agua con bebidas deportivas que contengan electrolitos (especialmente sodio) y carbohidratos, bebiendo pequeños sorbos cada 15-20 minutos o cada 3-4 km, adaptando la ingesta a la tasa de sudoración individual y las condiciones climáticas. Tras la carrera, es vital reponer líquidos y sales, buscando ingerir 1.2 a 1.5 veces el peso perdido en el ejercicio para una recuperación óptima y sostenida.",
    "Para deportistas veganos que buscan aumentar masa muscular, es fundamental asegurar una ingesta adecuada de proteínas. Las proteínas veganas en polvo, a menudo combinando fuentes como guisante, arroz, soja o cáñamo, ofrecen un perfil de aminoácidos completo y de fácil absorción. La creatina monohidrato (asegurándose de que sea de origen sintético y certificada como vegana) es un suplemento clave para mejorar la fuerza y la potencia. Adicionalmente, la vitamina B12 es indispensable, ya que no se encuentra en alimentos vegetales en cantidades suficientes, y la suplementación con Omega-3 (DHA y EPA) derivado de algas marinas es recomendable para apoyar la recuperación y la salud general."
]

# Preguntas de evaluación
questions = [
    "¿Para qué sirve la creatina?",
    "¿Qué beneficios tiene la cafeína en el deporte?",
    "¿Qué suplementos son ideales para mejorar la recuperación muscular después de entrenamientos intensos?",
    "¿Cómo afecta la suplementación con magnesio al rendimiento deportivo?",
    "¿Qué evidencia científica respalda el uso de antioxidantes en atletas de resistencia?",
    "¿Cuál es la mejor estrategia de hidratación para corredores de larga distancia?",
    "¿Qué suplementos son recomendados para deportistas veganos que buscan aumentar masa muscular?"
]

# Inicializar el resolver
resolver = RAGResolver()  # Asegúrate de que esté correctamente inicializado

# Generar respuestas del sistema
generated_answers = [resolver.resolver(q)["answer"] for q in questions]

# Mostrar respuestas
for q, gen, ref in zip(questions, generated_answers, reference_answers):
    print(f"❓ Pregunta: {q}\n🤖 Respuesta generada: {gen}\n✅ Referencia: {ref}\n")

# Evaluar ROUGE
rouge = evaluate.load("rouge")
rouge_scores = rouge.compute(predictions=generated_answers, references=reference_answers)
print("📊 ROUGE:", rouge_scores)

# Evaluar BLEU
bleu = evaluate.load("bleu")
bleu_scores = bleu.compute(predictions=generated_answers, references=[[r] for r in reference_answers])
print("📊 BLEU:", bleu_scores)
