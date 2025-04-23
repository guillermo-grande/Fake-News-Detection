import streamlit as st
from model import load_model, predict
from explainer import get_shap_values, generate_explanation

# Cargar recursos
load_model('models/best_bilstm.pth')

st.title("Clasificador de Noticias 🤖")

text = st.text_area("Introduce el texto de la noticia:", height=350)
if st.button("Clasificar noticia"):
    if not text:
        st.warning("Por favor, introduce una noticia.")
    else:
        preds = predict(text)
        st.markdown(f"**Resultado:** {'Real ✅' if preds==1 else 'Falsa ❌'}")

        # # Explicabilidad
        # token_vals = get_shap_values(text)
        # st.subheader("Tokens más influyentes (SHAP)")
        # for t, v in token_vals:
        #     st.write(f"{t}: {v:.4f}")

        # explanation = generate_explanation(token_vals)
        # st.subheader("Explicación LLM")
        # st.write(explanation)

# Navegación a resultados de entrenamiento
st.markdown("---")
st.markdown("Ve a la página **Resultados** en el menú lateral para ver métricas de entrenamiento y evaluación.")