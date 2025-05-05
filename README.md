# Fake News Detection using Artificial Intelligence

This project implements a complete pipeline for extracting, preprocessing, and automatically classifying fake news using Artificial Intelligence models. It includes:

* **Data Acquisition**: Utilizes the *Fake and Real News Dataset* from Kaggle, comprising approximately 20,000 fake news articles and 20,000 real news articles.
* **Preprocessing**: Includes cleaning, tokenization, lemmatization, and vectorization techniques such as TF–IDF, Word2Vec, and Sentence Transformers.
* **Supervised models**:
    * Random Forest
    * Multilayer Perceptron (MLP)
    * LSTM (standard, bidirectional, and with Self-Attention)
* **Explainability**: generating interpretations with SHAP and Deep SHAP.
* **Web Application**: A Streamlit-based interface that integrates the best-performing model. It provides an interactive platform to view evaluation metrics, test predictions with custom text inputs, and explore explainability features through LLM-powered responses for enhanced user experience.

---

## Installation

1. Clone the repository:

     ```bash
     git clone https://github.com/guillermo-grande/Fake-News-Detection.git
     cd Fake-News-Detection
     ```
2. Create and activate a virtual environment (example using `venv`):

     ```bash
     python3 -m venv venv
     source venv/bin/activate    # Linux / macOS
     venv\Scripts\activate       # Windows
     ```
3. Install the dependencies:

     ```bash
     pip install -r app/requirements.txt
     ```

---

## Execution

1. Navigate to the app folder:

     ```bash
     cd app
     ```
2. Start the Streamlit application:

     ```bash
     streamlit run app.py
     ```
3. Open your browser at the URL displayed in the console (default is `http://localhost:8501`).

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

