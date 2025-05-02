import shap
from utils import preprocess_text
import matplotlib.pyplot as plt
import numpy as np

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
    plt.title(f"Mejores {top_n} tokens influyendo en la clasificación hacia {'Real' if true_label==1 else 'Falsa'}")
    plt.xlabel("SHAP value")
    plt.tight_layout()

    # 8) Return the list of (token, score)
    return selected, plt

# Explicación con LLM
def generate_explanation(token_vals: list) -> str:
    return "hola"