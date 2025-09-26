"""
ECG Arrhythmia Classifier

This module implements the main ECG classifier with 1D CNN + Bi-GRU architecture
and temporal attention mechanism for interpretable arrhythmia detection.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import numpy as np
import logging

logger = logging.getLogger(__name__)

class TemporalAttention(nn.Module):
    """
    Temporal attention mechanism for highlighting critical time windows.
    """
    
    def __init__(self, hidden_size: int, attention_size: int = 64):
        """
        Initialize temporal attention module.
        
        Args:
            hidden_size: Size of hidden states
            attention_size: Size of attention layer
        """
        super(TemporalAttention, self).__init__()
        
        self.hidden_size = hidden_size
        self.attention_size = attention_size
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, attention_size),
            nn.Tanh(),
            nn.Linear(attention_size, 1)
        )
        
    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply temporal attention to hidden states.
        
        Args:
            hidden_states: Hidden states from GRU (batch, seq_len, hidden_size)
            
        Returns:
            Tuple of (attended_output, attention_weights)
        """
        # Calculate attention weights
        attention_weights = self.attention(hidden_states)  # (batch, seq_len, 1)
        attention_weights = F.softmax(attention_weights, dim=1)
        
        # Apply attention weights
        attended_output = torch.sum(hidden_states * attention_weights, dim=1)  # (batch, hidden_size)
        
        return attended_output, attention_weights.squeeze(-1)


class LeadAttention(nn.Module):
    """
    Lead-specific attention mechanism for multi-lead ECG analysis.
    """
    
    def __init__(self, num_leads: int, feature_size: int):
        """
        Initialize lead attention module.
        
        Args:
            num_leads: Number of ECG leads
            feature_size: Size of feature vectors
        """
        super(LeadAttention, self).__init__()
        
        self.num_leads = num_leads
        self.feature_size = feature_size
        
        # Lead-specific attention weights
        self.lead_attention = nn.Parameter(torch.randn(num_leads, feature_size))
        self.lead_attention.data.uniform_(-0.1, 0.1)
        
    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply lead attention to features.
        
        Args:
            features: Feature tensor (batch, num_leads, feature_size)
            
        Returns:
            Tuple of (attended_features, lead_weights)
        """
        # Calculate lead attention weights
        lead_weights = torch.softmax(self.lead_attention, dim=0)  # (num_leads, feature_size)
        
        # Apply lead attention
        attended_features = torch.sum(features * lead_weights.unsqueeze(0), dim=1)  # (batch, feature_size)
        
        return attended_features, lead_weights


class ECGClassifier(nn.Module):
    """
    Main ECG classifier with interpretable architecture.
    
            Architecture: 1D CNN per lead -> Bi-GRU -> Temporal Attention -> Classification
    """
    
    def __init__(self, 
                 num_leads: int = 2,
                 input_size: int = 1080,  # 3 seconds at 360 Hz
                 num_classes: int = 5,
                 cnn_channels: List[int] = [32, 64, 128],
                 gru_hidden_size: int = 128,
                 dropout_rate: float = 0.3,
                 use_lead_attention: bool = True):
        """
        Initialize ECG classifier.
        
        Args:
            num_leads: Number of ECG leads
            input_size: Input sequence length
            num_classes: Number of arrhythmia classes
            cnn_channels: List of CNN channel sizes
            gru_hidden_size: Hidden size for GRU
            dropout_rate: Dropout rate
            use_lead_attention: Whether to use lead attention
        """
        super(ECGClassifier, self).__init__()
        
        self.num_leads = num_leads
        self.input_size = input_size
        self.num_classes = num_classes
        self.use_lead_attention = use_lead_attention
        
        # Lead-specific CNN encoders
        self.lead_encoders = nn.ModuleList([
            self._build_cnn_encoder(cnn_channels) for _ in range(num_leads)
        ])
        
        # Calculate CNN output size
        cnn_output_size = cnn_channels[-1]
        
        # Bi-GRU for temporal modeling
        self.gru = nn.GRU(
            input_size=cnn_output_size,
            hidden_size=gru_hidden_size,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout_rate if 2 > 1 else 0
        )
        
        # Temporal attention
        self.temporal_attention = TemporalAttention(
            hidden_size=gru_hidden_size * 2  # Bidirectional
        )
        
        # Lead attention (optional)
        if use_lead_attention:
            self.lead_attention = LeadAttention(
                num_leads=num_leads,
                feature_size=gru_hidden_size * 2
            )
        
        # Classification layers
        classifier_input_size = gru_hidden_size * 2
        if use_lead_attention:
            classifier_input_size = gru_hidden_size * 2
        
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(classifier_input_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
        
    def _build_cnn_encoder(self, channels: List[int]) -> nn.Module:
        """
        Build CNN encoder for a single lead.
        
        Args:
            channels: List of channel sizes
            
        Returns:
            CNN encoder module
        """
        layers = []
        in_channels = 1
        
        for i, out_channels in enumerate(channels):
            layers.extend([
                nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2, stride=2),
                nn.Dropout(0.2)
            ])
            in_channels = out_channels
        
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.GRU):
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        nn.init.orthogonal_(param)
                    elif 'bias' in name:
                        nn.init.constant_(param, 0)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor (batch, num_leads, sequence_length)
            
        Returns:
            Dictionary containing outputs and attention weights
        """
        batch_size = x.size(0)
        
        # Process each lead through its CNN encoder
        lead_features = []
        for i in range(self.num_leads):
            # Extract lead data and add channel dimension
            lead_data = x[:, i, :].unsqueeze(1)  # (batch, 1, sequence_length)
            
            # Apply CNN encoder
            lead_feat = self.lead_encoders[i](lead_data)  # (batch, channels, seq_len)
            
            # Transpose for GRU input
            lead_feat = lead_feat.transpose(1, 2)  # (batch, seq_len, channels)
            lead_features.append(lead_feat)
        
        # Concatenate lead features along sequence dimension
        # This allows the GRU to learn cross-lead temporal patterns
        combined_features = torch.cat(lead_features, dim=1)  # (batch, total_seq_len, channels)
        
        # Apply Bi-GRU
        gru_output, _ = self.gru(combined_features)  # (batch, seq_len, hidden_size*2)
        
        # Apply temporal attention
        attended_output, temporal_weights = self.temporal_attention(gru_output)
        
        # Apply lead attention if enabled
        if self.use_lead_attention:
            # Reshape for lead attention (assuming equal sequence lengths per lead)
            seq_len_per_lead = gru_output.size(1) // self.num_leads
            lead_reshaped = gru_output.view(batch_size, self.num_leads, seq_len_per_lead, -1)
            lead_avg = torch.mean(lead_reshaped, dim=2)  # Average over time per lead
            
            attended_output, lead_weights = self.lead_attention(lead_avg)
        else:
            lead_weights = None
        
        # Classification
        logits = self.classifier(attended_output)
        probabilities = F.softmax(logits, dim=1)
        
        return {
            'logits': logits,
            'probabilities': probabilities,
            'temporal_attention': temporal_weights,
            'lead_attention': lead_weights,
            'features': attended_output
        }
    
    def get_attention_maps(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Get attention maps for interpretability analysis.
        
        Args:
            x: Input tensor
            
        Returns:
            Dictionary with attention maps
        """
        with torch.no_grad():
            outputs = self.forward(x)
            
            # Reshape temporal attention for visualization
            batch_size = x.size(0)
            seq_len_per_lead = outputs['temporal_attention'].size(1) // self.num_leads
            
            temporal_attention_reshaped = outputs['temporal_attention'].view(
                batch_size, self.num_leads, seq_len_per_lead
            )
            
            return {
                'temporal_attention': temporal_attention_reshaped,
                'lead_attention': outputs['lead_attention'],
                'probabilities': outputs['probabilities']
            }


class ECGClassifierLightweight(nn.Module):
    """
    Lightweight version of ECG classifier for faster training and inference.
    """
    
    def __init__(self, 
                 num_leads: int = 2,
                 input_size: int = 1080,
                 num_classes: int = 5,
                 hidden_size: int = 64,
                 dropout_rate: float = 0.3):
        """
        Initialize lightweight ECG classifier.
        
        Args:
            num_leads: Number of ECG leads
            input_size: Input sequence length
            num_classes: Number of arrhythmia classes
            hidden_size: Hidden size for the model
            dropout_rate: Dropout rate
        """
        super(ECGClassifierLightweight, self).__init__()
        
        self.num_leads = num_leads
        self.input_size = input_size
        
        # Simple CNN for feature extraction
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(num_leads, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=4),
            nn.Dropout(dropout_rate),
            
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(dropout_rate),
            
            nn.Conv1d(64, hidden_size, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size // 2, num_classes)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the lightweight model.
        
        Args:
            x: Input tensor (batch, num_leads, sequence_length)
            
        Returns:
            Dictionary containing outputs
        """
        # Input is already in correct format (batch, num_leads, sequence_length)
        # No need to transpose
        
        # Feature extraction
        features = self.feature_extractor(x)  # (batch, hidden_size, 1)
        features = features.squeeze(-1)  # (batch, hidden_size)
        
        # Classification
        logits = self.classifier(features)
        probabilities = F.softmax(logits, dim=1)
        
        return {
            'logits': logits,
            'probabilities': probabilities,
            'features': features
        }


def create_model(model_type: str = 'full', **kwargs) -> nn.Module:
    """
    Factory function to create ECG classifier models.
    
    Args:
        model_type: Type of model ('full' or 'lightweight')
        **kwargs: Additional arguments for model initialization
        
    Returns:
        Initialized model
    """
    if model_type == 'full':
        return ECGClassifier(**kwargs)
    elif model_type == 'lightweight':
        return ECGClassifierLightweight(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def get_model_summary(model: nn.Module) -> str:
    """
    Get a summary of the model architecture.
    
    Args:
        model: PyTorch model
        
    Returns:
        Model summary string
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    summary = f"Model: {model.__class__.__name__}\n"
    summary += f"Total parameters: {total_params:,}\n"
    summary += f"Trainable parameters: {trainable_params:,}\n"
    
    return summary


def main():
    """Example usage of the ECG classifier models."""
    # Create a sample input
    batch_size = 4
    num_leads = 2
    sequence_length = 1080  # 3 seconds at 360 Hz
    
    x = torch.randn(batch_size, num_leads, sequence_length)
    
    # Test full model
    print("Testing full ECG classifier...")
    full_model = create_model('full', num_leads=num_leads, input_size=sequence_length)
    full_output = full_model(x)
    
    print(f"Full model output shape: {full_output['logits'].shape}")
    print(f"Temporal attention shape: {full_output['temporal_attention'].shape}")
    print(f"Lead attention shape: {full_output['lead_attention'].shape}")
    
    # Test lightweight model
    print("\nTesting lightweight ECG classifier...")
    light_model = create_model('lightweight', num_leads=num_leads, input_size=sequence_length)
    light_output = light_model(x)
    
    print(f"Lightweight model output shape: {light_output['logits'].shape}")
    
    # Print model summaries
    print(f"\n{get_model_summary(full_model)}")
    print(f"\n{get_model_summary(light_model)}")


if __name__ == "__main__":
    main()
