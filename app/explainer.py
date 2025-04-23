import shap
from utils import preprocess_text

# SHAP explainer para Random Forest
# def get_shap_values(text: str):
#     explainer = shap.TreeExplainer(rf_model)
#     x = text_to_vector(preprocess_text(text))
#     shap_vals = explainer.shap_values(x)
#     # obtener tokens y valores
#     tokens = preprocess_text(text).split()
#     # Simplificación: usar shap_vals[1][0] para la clase positiva
#     vals = shap_vals[1][0][:len(tokens)]
#     return list(zip(tokens, vals))

def get_shap_values(text: str):
    return "hola"

# Explicación con LLM
def generate_explanation(token_vals: list) -> str:
    return "hola"