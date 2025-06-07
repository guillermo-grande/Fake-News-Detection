# Fake News Detection using Artificial Intelligence 📰❌

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

## Installation and Execution ▶️

You can run this project using either a **manual setup** or a fully automated **Docker 🐳** environment.

---

### 🔧 Option 1: Manual Setup and Run

1. **Clone the repository and navigate to the app directory**:

    ```bash
    git clone https://github.com/guillermo-grande/Fake-News-Detection.git
    cd Fake-News-Detection/app
    ```

2. **Create and activate a virtual environment** (using `venv` as an example):

    ```bash
    python3 -m venv .venv
    source .venv/bin/activate     # Linux / macOS
    .venv/Scripts/activate        # Windows
    ```

3. **Install the required dependencies**:

    ```bash
    pip install -r requirements.txt
    ```

4. **Start the Streamlit application**:

    ```bash
    streamlit run app.py
    ```

5. **Open your browser** at the URL shown in the terminal (usually [http://localhost:8501](http://localhost:8501)).

---

### 🐳 Option 2: Docker (All-in-One)

Skip manual setup with Docker! This option runs the entire app in a containerized environment.

> ⚠️ **Note:** The first build may take **10–15 minutes** to install all scientific dependencies. Grab a coffee ☕ and relax!

1. From the root of the project (where `docker-compose.yml` is located), run:

    ```bash
    docker-compose up -d
    ```

2. Once the container is ready, access the app at: [http://localhost:8501](http://localhost:8501)

---

## License 📄

This project is licensed under the MIT License. See the `LICENSE` file for more details.

