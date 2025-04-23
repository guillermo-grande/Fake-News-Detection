import pickle
import torch
from utils import preprocess_text
import torch
import torch.nn as nn

def load_model(bilstm_path: str):
    global bilstm_model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bilstm_model.load_state_dict(torch.load(bilstm_path))
    bilstm_model.to(device)
    bilstm_model.eval()

# Predicción con ambos modelos y votación

def predict(text: str) -> dict:
    proc = preprocess_text(text)
    # BiLSTM: implementar tokenización idéntica al entrenamiento
    tokens = proc.split()
    # Aquí debería ir conversión a ids, padding, torch tensor...
    with torch.no_grad():
        inputs = torch.tensor(...)  # placeholder
        logits = bilstm_model(inputs)
        bilstm_pred = torch.argmax(logits, dim=1).item()

    # Voting
    return bilstm_pred

class BiLSTM(nn.Module):
    def __init__(self, input_dim=10000, embedding_dim=64, hidden_dim_1=64, hidden_dim_2=32):
        super(BiLSTM, self).__init__()
        
        self.embedding = nn.Embedding(
            num_embeddings=input_dim,
            embedding_dim=embedding_dim,
            padding_idx=0
        )
        
        self.bilstm1 = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim_1,
            batch_first=True,
            bidirectional=True
        )
        
        self.bilstm2 = nn.LSTM(
            input_size=hidden_dim_1*2,  # Porque es bidireccional (64 * 2)
            hidden_size=hidden_dim_2,
            batch_first=True,
            bidirectional=True
        )

        self.fc1 = nn.Linear(hidden_dim_2 * 2, 16)  # Salida de la segunda BiLSTM
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.embedding(x)  # (batch_size, seq_len) → (batch_size, seq_len, 64)
        out, _ = self.bilstm1(x)  # (batch_size, seq_len, 128)
        out, _ = self.bilstm2(out)  # (batch_size, seq_len, 64)
        # Tomamos el último paso de la secuencia
        out = out[:, -1, :]  # (batch_size, 64)
        out = self.fc1(out) # (batch_size, 16)
        out = self.relu(out) 
        out = self.fc2(out) # (batch_size, 1)
        return self.sigmoid(out)  # Para clasificación binaria
    
# Carga de modelos preentrenados
bilstm_model = BiLSTM()