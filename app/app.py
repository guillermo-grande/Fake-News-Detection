import streamlit as st

def main():

    st.set_page_config(page_title="Clasificador de Noticias 🤖", layout="wide")

    # Diferentes páginas de la aplicación
    classifier_page = st.Page("pages/classifier.py", title="Clasificador", icon="🧠")
    results_page = st.Page("pages/results.py", title="Resultados", icon="📋")
    
    # Crear navegación
    pg = st.navigation({"Navegación":[classifier_page, results_page]}, position="sidebar")

    pg.run()

if __name__ == "__main__":
    main()