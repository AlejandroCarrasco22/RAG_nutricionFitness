# 🥗 NutriBot - Asistente de Nutrición y Fitness

Un chatbot inteligente especializado en nutrición, suplementación deportiva y fitness, con una interfaz web moderna y atractiva.

## 🚀 Características

- **Interfaz Web Moderna**: Diseño responsive con colores vibrantes apropiados para nutrición y fitness
- **Chat Inteligente**: Sistema RAG (Retrieval-Augmented Generation) para respuestas precisas
- **Base de Datos Nutricional**: Información detallada sobre composición de alimentos
- **Suplementación Deportiva**: Información científica sobre suplementos y rendimiento
- **Memoria de Conversación**: Mantiene contexto entre preguntas
- **Sugerencias Rápidas**: Botones con preguntas frecuentes y mejor padding
- **Diseño Responsive Avanzado**: Se adapta perfectamente a cualquier dispositivo

## 🎨 Paleta de Colores

- **Verde Esmeralda** (#10B981) - Salud y bienestar
- **Naranja Vibrante** (#F59E0B) - Energía y motivación  
- **Azul Profundo** (#1E40AF) - Confianza y profesionalismo
- **Blanco** (#FFFFFF) - Limpieza y claridad
- **Gris Claro** (#F3F4F6) - Neutralidad y equilibrio

## 📋 Requisitos

- Python 3.8 o superior
- Conexión a internet (para modelos de IA)
- Navegador web moderno

## 🛠️ Instalación

1. **Clonar el repositorio**:
   ```bash
   git clone <tu-repositorio>
   cd RAG_implementation
   ```


2. **Instalar dependencias**:
   Preferiblemente en un entorno .venv.
   ```bash
   pyhton -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```
   ```bash
   pip install -r requirements.txt
   ```

4. **Configurar claves API**:
   - Edita el archivo `keys.py` con tus claves de API necesarias
   - Asegúrate de tener configurado el LLM y embeddings

## 🚀 Uso

### Ejecutar el Chatbot

```bash
python main.py
```

El sistema automáticamente:
1. Iniciará el servidor Flask en `http://localhost:5000`
2. Abrirá tu navegador predeterminado
3. Mostrará la interfaz web del chatbot

### Funcionalidades

- **Preguntas Libres**: Escribe cualquier pregunta sobre nutrición o fitness
- **Sugerencias Rápidas**: Usa los botones de preguntas frecuentes con mejor padding
- **Chat Continuo**: El bot mantiene memoria de la conversación
- **Respuestas Científicas**: Basadas en la base de datos de documentos especializados

### Ejemplos de Preguntas

- "¿Qué beneficios tiene la creatina?"
- "¿Cuántas proteínas necesito al día?"
- "¿Qué comer antes del entrenamiento?"
- "¿Es buena la dieta vegana para deportistas?"
- "¿Cuáles son los mejores suplementos para ganar músculo?"

## 🏗️ Arquitectura

```
RAG_implementation/
├── app.py                 # Aplicación Flask
├── main.py               # Punto de entrada principal
├── templates/
│   └── index.html        # Interfaz web (HTML limpio)
├── static/
│   └── css/
│       └── styles.css    # Estilos CSS separados
├── resolvers/            # Lógica de resolución
├── data/                 # Datos y documentos
├── chroma_db_dir/        # Base de datos vectorial
└── requirements.txt      # Dependencias
```

## 🎯 Tecnologías Utilizadas

- **Backend**: Python, Flask, LangChain
- **Frontend**: HTML5, CSS3, JavaScript
- **IA**: OpenAI GPT, Sentence Transformers
- **Base de Datos**: ChromaDB (vectorial), SQLite
- **Estilo**: Diseño moderno con gradientes y animaciones

## 📱 Diseño Responsive Avanzado

### Breakpoints Optimizados:

| Dispositivo | Ancho | Altura | Título | Texto | Características |
|-------------|-------|--------|--------|-------|-----------------|
| Móvil pequeño | 320px | 98vh | 20px | 14px | Compacto, touch-friendly |
| Móvil grande | 480px | 95vh | 22px | 15px | Balanceado |
| Tablet | 768px | 90vh | 26px | 16px | Intermedio |
| Laptop | 1200px | 88vh | 28px | 16px | Estándar |
| Escritorio | 1600px+ | 80vh | 36px | 20px | Espacioso |

### Mejoras en Botones de Sugerencias:
- **Padding mejorado**: Más espacio interno para mejor usabilidad
- **Efectos hover**: Animaciones suaves al pasar el mouse
- **Sombras sutiles**: Profundidad visual mejorada
- **Centrado automático**: Mejor distribución en pantallas pequeñas

## 🔧 Configuración Avanzada

### Cambiar Puerto
Edita `main.py` línea 67:
```python
app.run(debug=False, host='0.0.0.0', port=5000)
```

### Personalizar Colores
Edita `static/css/styles.css` para cambiar la paleta de colores.

### Modificar Estilos
Los estilos están organizados en `static/css/styles.css` para fácil mantenimiento.

## 📱 Responsive Design

La interfaz se adapta automáticamente a:
- **Pantallas muy grandes** (1600px+) - Experiencia espaciosa
- **Escritorio** (1200px+) - Tamaño estándar
- **Laptop** (768px+) - Tamaño intermedio
- **Tablet** (768px) - Optimizado para touch
- **Móvil grande** (480px+) - Compacto pero funcional
- **Móvil pequeño** (320px+) - Mínimo pero usable
- **Orientación landscape** - Optimizado para pantalla horizontal

## 🛑 Detener el Servidor

Presiona `Ctrl+C` en la terminal donde ejecutaste `main.py`.

---

**¡Disfruta usando NutriBot para mejorar tu nutrición y fitness! 💪🥗** 
