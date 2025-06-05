import streamlit as st
from model import load_model, predict, extract_vectorizer_layer, extract_embedding_layer
from explainer import load_explainability_model, create_explainer, get_top_shap_tokens, generate_explanation
import os
import matplotlib.pyplot as plt

# Cargar recursos
with st.spinner("Cargando modelos, por favor espera..."):
    load_model('models/best_bilstm.keras')
    load_explainability_model('models/model_explainability.keras')
    vectorizer = extract_vectorizer_layer()
    emb_no_mask = extract_embedding_layer()
    explainer = create_explainer('models/background_data.npy')

st.title("Clasificador de Noticias 🤖")

text = st.text_area("Introduce el texto de la noticia:", height=300)
if st.button("Clasificar noticia 🕵️‍♂️"):
    if not text:
        st.warning("Por favor, introduce una noticia.")
    else:
        st.markdown("---")
        with st.spinner("Calculando la predicción..."):
            label, processed_text, probability = predict(text)
        st.subheader(f"La noticia introducida es: {'Real ✅' if label == 1 else 'Falsa ❌'}")

        # Display preprocessed text in an expander
        with st.expander("Ver detalles de la clasificación"):
            st.write(f"**Texto preprocesado:** {processed_text}")
            st.write(f"**Modelo utilizado:** BiLSTM - TensorFlow")
            st.write(f"**Probabilidad devuelta por el modelo:** {probability:.4f}")
            st.write(f"**Etiqueta binaria:** {label}")

        st.markdown("---")

        # Explicabilidad
        with st.spinner("Cargando modelos de explicabilidad y generando resultados..."):
            token_vals, plt = get_top_shap_tokens(explainer, vectorizer, emb_no_mask, text=processed_text, true_label=label, top_n=10)
        st.subheader("Gráfico de las palabras más influyentes para la clasificación (SHAP)")
        col1, col2 = st.columns([3, 1])
        with col1:
            st.pyplot(plt)

        st.markdown("---")

        # Verificar si existe la clave de API de OpenAI
        if "OPENAI_API_KEY" not in os.environ:
            st.error("Se requiere una clave de API de OpenAI para generar explicaciones. Por favor, configura la variable de entorno 'OPENAI_API_KEY'.")
        else:
            # Explicación LLM 
            with st.spinner("Conectando con la API del LLM y generando resultados..."):
                explanation = generate_explanation(token_vals, label, text)
            st.subheader("Explicación intuitiva basada en LLM")
            st.markdown("##### Basándonos en las palabras más influyentes devueltas por SHAP y en la propia noticia:")
            st.write(explanation)

