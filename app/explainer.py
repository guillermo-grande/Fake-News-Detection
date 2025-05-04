import shap
from utils import preprocess_text
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain.schema import HumanMessage

# Variable global para el modelo de explicabilidad
explainability_model = None

# Carga del modelo de explicabilidad
def load_explainability_model(keras_model_path: str):
    global explainability_model
    explainability_model = tf.keras.models.load_model(keras_model_path)
    print("Modelo de explicabilidad de Tensorflow cargado correctamente.")

# Crear el modelo de explicabilidad
def create_explainer(background_data_path: str):
    global explainability_model
    if explainability_model is None:
        raise ValueError("Model for explainability is not loaded. Please load the model first.")
    background_data = np.load(background_data_path)
    explainer = shap.GradientExplainer(explainability_model, [background_data])
    return explainer

# Crearemos una función para facilitar el proceso
def get_top_shap_tokens(explainer, vectorizer, emb_no_mask, text, true_label, top_n=10):
    """
    Given a SHAP explainer, a raw text string and its true label (0 or 1),
    returns the top_n tokens pushing the model output towards that label.
    """
    # 1) Tokenize and embed the input text
    tok_ids = vectorizer([text]).numpy().astype(np.int32)          # (1, seq_len)
    tok_emb = emb_no_mask(tok_ids).numpy()                         # (1, seq_len, emb_dim)
    # 2) Compute SHAP values on embedding
    shap_emb = explainer.shap_values(tok_emb)[0]                   # (1, seq_len, emb_dim)
    # 3) Sum across embedding dimensions to get one score per token
    shap_token_vals = np.sum(shap_emb[0], axis=1)                  # (seq_len,)
    # 4) Emparejamos tokens con sus valores SHAP, evitando overflow
    vocab = vectorizer.get_vocabulary()
    token_ids = tok_ids[0]
    token_shaps = [
        (vocab[token_id], float(shap_val))
        for token_id, shap_val in zip(token_ids, shap_token_vals)
        if token_id != 0 and vocab[token_id] != "[UNK]"
    ]
    # 5) Separate positive vs. negative contributions
    pos = sorted([ts for ts in token_shaps if ts[1] > 0], key=lambda x: x[1], reverse=True)
    neg = sorted([ts for ts in token_shaps if ts[1] < 0], key=lambda x: x[1])
    # 6) Select top_n tokens driving towards the true_label
    selected = pos[:top_n] if true_label == 1 else neg[:top_n]

    # 7) Plot pushes of those tokens
    tokens, scores = zip(*selected)
    y_pos = list(range(len(tokens)))
    colors = ['green' if s > 0 else 'red' for s in scores]

    plt.figure(figsize=(8, 4))
    plt.barh(y_pos, scores, color=colors)
    plt.yticks(y_pos, tokens)
    plt.axvline(0, color='black', linewidth=0.8)
    plt.gca().invert_yaxis()
    plt.title(f"Las {top_n} palabras más influyentes para clasificar la noticia como {'real' if true_label==1 else 'falsa'}")
    plt.xlabel("SHAP value")
    plt.tight_layout()

    # 8) Return the list of (token, score)
    return selected, plt

# Prompt template
template = "{raw_prompt}"
prompt = PromptTemplate(input_variables=["raw_prompt"], template=template)

# Definimos el modelo a utilizar
load_dotenv() # Cargamos la API KEY
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.5)

# Cadena de LangChain (Modelo + Prompt)
chain = prompt | llm


def generate_prompt(token_vals: list[tuple[str, float]], label: int) -> str:
    """
    Genera un prompt para un LLM que explique, en formato Markdown,
    por qué cada token influyó en la clasificación de una noticia como verdadera o falsa.
    """
    # Mapeo de la etiqueta a texto
    label_text = "verdadera" if label == 1 else "falsa"

    # Cabecera del prompt
    prompt = (
        f"Eres un analista de ML que debe explicar por qué ciertos tokens "
        f"influyeron en la clasificación de una noticia como **{label_text}**.\n\n"
        "A continuación tienes los 10 tokens más influyentes con sus valores SHAP:\n"
    )

    # Lista de tokens con sus valores
    for token, shap in token_vals:
        prompt += f"- Token: \"{token}\", SHAP value: {shap:.4f}\n"

    # Instrucciones para el LLM: generar Markdown
    prompt += (
        "\nPor cada token, proporciona una explicación de una línea de por qué "
        f"posiblemente influyó en la clasificación hacia **{label_text}**. "
        "Devuélvelo exclusivamente en formato Markdown usando listas:\n"
        "- **token** (shap_value): explicación.\n"
    )

    return prompt

def generate_explanation(token_vals: list[tuple[str, float]], label: int, text: str) -> tuple[str, str]:
    """
    Genera una explicación Markdown basada en los valores SHAP y una
    explicación breve (máx. 3 líneas) de por qué la noticia original fue
    clasificada como verdadera o falsa.
    """
    global llm  # Instancia de ChatOpenAI
    global chain  # Cadena existente para valores SHAP

    # Generar explicación basada en los valores SHAP
    shap_prompt = generate_prompt(token_vals, label)
    explanation_shap = chain.invoke(input=shap_prompt)

    # Crear prompt para explicación breve según el texto y la etiqueta
    label_str = "falsa" if label == 0 else "real"
    summary_prompt = (
        f"Dado el siguiente texto de una noticia que ha sido clasificada como {label_str} por un modelo de inteligencia artificial:\n\n{text}\n\n"
        f"Explica en un máximo de cuatro líneas las razones principales por las que la noticia es {label_str}. "
        "No repitas el texto, solo proporciona una explicación directa y razonada en español, sin rodeos ni títulos. "
        "Evita basarte en tu conocimiento sobre los hechos, únicamente céntrate en el texto proporcionado. "
        "Usa un lenguaje claro y objetivo."
    )

    # Llamar al modelo directamente para obtener la explicación breve
    explanation_summary = llm.invoke(summary_prompt)

    return explanation_shap.content, explanation_summary.content
