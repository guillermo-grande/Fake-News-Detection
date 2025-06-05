import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.title("Resultados 📋")

st.markdown("---")



# Subtítulo para las métricas de evaluación de los modelos
st.subheader("Métricas de los Modelos entrenados")
# Leemos y mostramos el dataset desde un archivo CSV
df = pd.read_csv("static/model_metrics.csv")

# Convertimos la columna de Precisión (Test) a float para poder calcular el máximo
df['Precisión en Test (%)'] = (df['Precisión en Test (%)'].str.rstrip(' %').astype(float).round(4))
# Índice de la fila ganadora
winner_idx = df['Precisión en Test (%)'].idxmax()

# Función de estilo que solo pinta la columna "Precisión en Test (%)" de la fila ganadora
def highlight_precision_cell(row):
    styles = [''] * len(row)
    if row.name == winner_idx:
        # localizamos la posición de la columna
        col = 'Precisión en Test (%)'
        col_idx = row.index.get_loc(col)
        styles[col_idx] = 'background-color: #a8e6c1; font-weight: bold'
    return styles

# Convertimos la columna de Precisión (Test) de nuevo a string con el símbolo '%'
df['Precisión en Test (%)'] = df['Precisión en Test (%)'].astype(str) + ' %'

# Aplicamos el estilo
styled_df = df.style.apply(highlight_precision_cell, axis=1)

st.dataframe(styled_df)


st.markdown("---")


# Subtítulo para la información del dataset y preprocesamiento
st.subheader("Datos del Dataset Preprocesado")

# Añadimos espacio
st.write("")

# Creamos columnas
col1, col2 = st.columns([2, 3])  # Ajustamos las proporciones para que ambas columnas sean iguales
with col1:
    st.image("static/classes_distribution.png", use_container_width=True)

with col2:
    st.image("static/length_distribution.png", use_container_width=True)
