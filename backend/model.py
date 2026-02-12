"""
Deep Learning Model for Emoji Prediction
Implements LSTM-based classifier with attention mechanism.
"""
import torch
import torch.nn as nn
import numpy as np


class LSTMAttentionClassifier(nn.Module):
    """
    LSTM with Attention mechanism for text classification.
    Used for multi-class emoji prediction.
    """
    
    def __init__(self, embedding_dim, hidden_dim, num_classes, num_layers=2, dropout=0.3):
        super(LSTMAttentionClassifier, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Attention layer
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def attention_pooling(self, lstm_output):
        """Apply attention mechanism to LSTM outputs."""
        attention_weights = self.attention(lstm_output)
        attention_weights = torch.softmax(attention_weights, dim=1)
        context = torch.sum(attention_weights * lstm_output, dim=1)
        return context, attention_weights.squeeze(-1)
    
    def forward(self, x):
        """
        Forward pass.
        x: (batch_size, seq_len, embedding_dim)
        """
        # LSTM
        lstm_output, _ = self.lstm(x)
        
        # Attention
        context, attention_weights = self.attention_pooling(lstm_output)
        
        # Classification
        logits = self.classifier(context)
        
        return logits, attention_weights


class GRUClassifier(nn.Module):
    """
    GRU-based classifier as an alternative to LSTM.
    """
    
    def __init__(self, embedding_dim, hidden_dim, num_classes, num_layers=2, dropout=0.3):
        super(GRUClassifier, self).__init__()
        
        self.gru = nn.GRU(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, x):
        """Forward pass."""
        output, hidden = self.gru(x)
        # Use last hidden state
        hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        logits = self.classifier(hidden)
        return logits


class TextCNN(nn.Module):
    """
    CNN for text classification.
    Uses multiple kernel sizes for n-gram feature extraction.
    """
    
    def __init__(self, embedding_dim, num_classes, num_filters=100, filter_sizes=[3, 4, 5], dropout=0.5):
        super(TextCNN, self).__init__()
        
        self.convs = nn.ModuleList([
            nn.Conv1d(embedding_dim, num_filters, kernel_size=fs)
            for fs in filter_sizes
        ])
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(filter_sizes), num_classes)
    
    def forward(self, x):
        """
        Forward pass.
        x: (batch_size, seq_len, embedding_dim)
        """
        x = x.permute(0, 2, 1)  # (batch_size, embedding_dim, seq_len)
        
        conv_outputs = []
        for conv in self.convs:
            c = torch.relu(conv(x))
            c = torch.max_pool1d(c, c.size(2)).squeeze(2)
            conv_outputs.append(c)
        
        concat = torch.cat(conv_outputs, dim=1)
        concat = self.dropout(concat)
        logits = self.fc(concat)
        
        return logits


class EmojiPredictor:
    """
    High-level wrapper for emoji prediction using transformer embeddings.
    Uses sentence-transformers for embedding and semantic similarity.
    """
    
    def __init__(self, model, emoji_df, emoji_embeddings):
        self.model = model
        self.emoji_df = emoji_df
        self.emoji_embeddings = emoji_embeddings
    
    def predict(self, text_embedding, top_k=5):
        """
        Predict top-k emojis using cosine similarity.
        """
        from sentence_transformers import util
        
        cosine_scores = util.cos_sim(text_embedding, self.emoji_embeddings)[0]
        top_results = cosine_scores.argsort(descending=True)[:top_k]
        
        results = []
        for idx in top_results:
            idx = int(idx)
            results.append({
                "emoji": self.emoji_df.iloc[idx]['emoji'],
                "name": self.emoji_df.iloc[idx]['name'],
                "score": float(cosine_scores[idx])
            })
        
        return results
    
    def predict_with_softmax(self, text_embedding, top_k=5):
        """
        Predict using softmax probabilities for multi-class classification.
        """
        from sentence_transformers import util
        import torch.nn.functional as F
        
        cosine_scores = util.cos_sim(text_embedding, self.emoji_embeddings)[0]
        probabilities = F.softmax(cosine_scores, dim=0)
        
        top_results = probabilities.argsort(descending=True)[:top_k]
        
        results = []
        for idx in top_results:
            idx = int(idx)
            results.append({
                "emoji": self.emoji_df.iloc[idx]['emoji'],
                "name": self.emoji_df.iloc[idx]['name'],
                "score": float(probabilities[idx]),
                "raw_similarity": float(cosine_scores[idx])
            })
        
        return results
