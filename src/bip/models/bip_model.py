#!/usr/bin/env python3
"""
BIP Temporal Invariance Model

Architecture:
- Encoder: Pretrained transformer (freezable)
- Disentangler: Splits representation into z_bond (invariant) and z_label (temporal)
- Adversarial time classifier: Tries to predict era from z_bond (we want it to fail)
- Hohfeldian classifier: Predicts moral state from z_bond
- Reconstruction decoder: Optional, ensures information preservation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from typing import Dict, Optional, Tuple
import math

# =============================================================================
# MODEL COMPONENTS
# =============================================================================

class GradientReversal(torch.autograd.Function):
    """
    Gradient reversal layer for adversarial training.

    Forward: identity
    Backward: negate gradient and scale by lambda
    """
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None

def gradient_reversal(x, lambda_=1.0):
    return GradientReversal.apply(x, lambda_)


class BIPEncoder(nn.Module):
    """
    Encode passages into latent space.

    Uses pretrained transformer + projection layers.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-mpnet-base-v2",
        d_model: int = 768,
        freeze_encoder: bool = False
    ):
        super().__init__()

        self.encoder = AutoModel.from_pretrained(model_name)
        self.d_model = d_model

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        # Get encoder hidden size
        self.encoder_dim = self.encoder.config.hidden_size

        # Projection if dimensions don't match
        if self.encoder_dim != d_model:
            self.projection = nn.Linear(self.encoder_dim, d_model)
        else:
            self.projection = nn.Identity()

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Encode passages.

        Returns: [batch, d_model] pooled representation
        """
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)

        # Mean pooling
        hidden = outputs.last_hidden_state  # [batch, seq, hidden]
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)

        return self.projection(pooled)


class BIPDisentangler(nn.Module):
    """
    Disentangle representation into:
    - z_bond: Bond-level features (should be temporally invariant)
    - z_label: Label-level features (should capture temporal variation)

    Uses variational approach for regularization.
    """

    def __init__(
        self,
        d_model: int = 768,
        d_bond: int = 128,
        d_label: int = 64,
        dropout: float = 0.1
    ):
        super().__init__()

        self.d_model = d_model
        self.d_bond = d_bond
        self.d_label = d_label

        # Bond space projection (variational)
        self.bond_mean = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_bond)
        )
        self.bond_logvar = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_bond)
        )

        # Label space projection (variational)
        self.label_mean = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_label)
        )
        self.label_logvar = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_label)
        )

    def reparameterize(self, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick for VAE."""
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mean + eps * std
        else:
            return mean

    def forward(self, h: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Disentangle representation.

        Args:
            h: [batch, d_model] encoder output

        Returns:
            dict with z_bond, z_label, and VAE parameters
        """
        # Bond space
        bond_mean = self.bond_mean(h)
        bond_logvar = self.bond_logvar(h)
        z_bond = self.reparameterize(bond_mean, bond_logvar)

        # Label space
        label_mean = self.label_mean(h)
        label_logvar = self.label_logvar(h)
        z_label = self.reparameterize(label_mean, label_logvar)

        return {
            'z_bond': z_bond,
            'z_label': z_label,
            'bond_mean': bond_mean,
            'bond_logvar': bond_logvar,
            'label_mean': label_mean,
            'label_logvar': label_logvar
        }


class TimeClassifier(nn.Module):
    """
    Adversarial time period classifier.

    Used to ensure z_bond is time-invariant.
    We want this to FAIL when given z_bond.
    """

    def __init__(self, d_input: int, n_periods: int = 9, hidden_dim: int = 128):
        super().__init__()

        self.classifier = nn.Sequential(
            nn.Linear(d_input, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, n_periods)
        )

    def forward(self, z: torch.Tensor, reverse_grad: bool = False, lambda_: float = 1.0) -> torch.Tensor:
        """
        Predict time period.

        Args:
            z: latent representation
            reverse_grad: if True, reverse gradients (adversarial)
            lambda_: gradient reversal scale
        """
        if reverse_grad:
            z = gradient_reversal(z, lambda_)
        return self.classifier(z)


class HohfeldianClassifier(nn.Module):
    """
    Classify passages into Hohfeldian states.

    States: Right (0), Obligation (1), Liberty (2), No-Right (3)
    """

    def __init__(self, d_input: int, n_classes: int = 4, hidden_dim: int = 64):
        super().__init__()

        self.classifier = nn.Sequential(
            nn.Linear(d_input, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, n_classes)
        )

    def forward(self, z_bond: torch.Tensor) -> torch.Tensor:
        return self.classifier(z_bond)


class BondClassifier(nn.Module):
    """
    Classify primary bond type.

    Multi-label classification for bond types present.
    """

    def __init__(self, d_input: int, n_bond_types: int = 10, hidden_dim: int = 64):
        super().__init__()

        self.classifier = nn.Sequential(
            nn.Linear(d_input, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, n_bond_types)
        )

    def forward(self, z_bond: torch.Tensor) -> torch.Tensor:
        return self.classifier(z_bond)  # Apply sigmoid for multi-label


# =============================================================================
# FULL MODEL
# =============================================================================

class BIPTemporalInvarianceModel(nn.Module):
    """
    Complete model for BIP temporal invariance testing.
    """

    def __init__(self, config: dict):
        super().__init__()

        self.config = config

        # Components
        self.encoder = BIPEncoder(
            model_name=config.get('encoder', 'sentence-transformers/all-mpnet-base-v2'),
            d_model=config.get('d_model', 768),
            freeze_encoder=config.get('freeze_encoder', False)
        )

        self.disentangler = BIPDisentangler(
            d_model=config.get('d_model', 768),
            d_bond=config.get('d_bond', 128),
            d_label=config.get('d_label', 64),
            dropout=config.get('dropout', 0.1)
        )

        self.time_classifier = TimeClassifier(
            d_input=config.get('d_bond', 128),
            n_periods=config.get('n_time_periods', 9)
        )

        self.time_classifier_label = TimeClassifier(
            d_input=config.get('d_label', 64),
            n_periods=config.get('n_time_periods', 9)
        )

        self.hohfeld_classifier = HohfeldianClassifier(
            d_input=config.get('d_bond', 128),
            n_classes=config.get('n_hohfeld_classes', 4)
        )

        self.bond_classifier = BondClassifier(
            d_input=config.get('d_bond', 128),
            n_bond_types=10
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        time_labels: Optional[torch.Tensor] = None,
        adversarial_lambda: float = 1.0
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Returns dict with all predictions and latent representations.
        """
        # Encode
        h = self.encoder(input_ids, attention_mask)

        # Disentangle
        disentangled = self.disentangler(h)
        z_bond = disentangled['z_bond']
        z_label = disentangled['z_label']

        # Predictions
        # Time from z_bond (adversarial - we want this to fail)
        time_pred_bond = self.time_classifier(z_bond, reverse_grad=True, lambda_=adversarial_lambda)

        # Time from z_label (should succeed)
        time_pred_label = self.time_classifier_label(z_label)

        # Hohfeldian state from z_bond
        hohfeld_pred = self.hohfeld_classifier(z_bond)

        # Bond types from z_bond
        bond_pred = self.bond_classifier(z_bond)

        return {
            'z_bond': z_bond,
            'z_label': z_label,
            'bond_mean': disentangled['bond_mean'],
            'bond_logvar': disentangled['bond_logvar'],
            'label_mean': disentangled['label_mean'],
            'label_logvar': disentangled['label_logvar'],
            'time_pred_bond': time_pred_bond,
            'time_pred_label': time_pred_label,
            'hohfeld_pred': hohfeld_pred,
            'bond_pred': bond_pred
        }

    def encode_bond(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Get bond embedding only (for evaluation)."""
        h = self.encoder(input_ids, attention_mask)
        disentangled = self.disentangler(h)
        return disentangled['z_bond']


# =============================================================================
# LOSS FUNCTIONS
# =============================================================================

class BIPLoss(nn.Module):
    """
    Combined loss for BIP training.

    Components:
    1. Adversarial time loss (maximize entropy of time prediction from z_bond)
    2. Time prediction loss (z_label should predict time well)
    3. KL divergence (regularization)
    4. Hohfeldian classification loss
    5. BIP contrastive loss (bond-isomorphic pairs should have similar z_bond)
    """

    def __init__(self, config: dict):
        super().__init__()

        self.lambda_adv = config.get('lambda_adversarial', 1.0)
        self.lambda_kl = config.get('lambda_kl', 0.1)
        self.lambda_hohfeld = config.get('lambda_hohfeld', 1.0)
        self.lambda_bip = config.get('lambda_bip', 2.0)

        self.ce_loss = nn.CrossEntropyLoss()
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        time_labels: torch.Tensor,
        hohfeld_labels: Optional[torch.Tensor] = None,
        bond_labels: Optional[torch.Tensor] = None,
        isomorphic_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute total loss.
        """
        losses = {}

        # 1. Adversarial time loss on z_bond
        # We want maximum entropy = classifier is confused
        time_probs = F.softmax(outputs['time_pred_bond'], dim=-1)
        entropy = -torch.sum(time_probs * torch.log(time_probs + 1e-8), dim=-1)
        # Maximize entropy = minimize negative entropy
        losses['adv'] = -entropy.mean() * self.lambda_adv

        # 2. Time prediction loss on z_label (should succeed)
        losses['time'] = self.ce_loss(outputs['time_pred_label'], time_labels)

        # 3. KL divergence regularization
        kl_bond = -0.5 * torch.mean(
            1 + outputs['bond_logvar'] - outputs['bond_mean'].pow(2) - outputs['bond_logvar'].exp()
        )
        kl_label = -0.5 * torch.mean(
            1 + outputs['label_logvar'] - outputs['label_mean'].pow(2) - outputs['label_logvar'].exp()
        )
        losses['kl'] = (kl_bond + kl_label) * self.lambda_kl

        # 4. Hohfeldian classification loss
        if hohfeld_labels is not None:
            losses['hohfeld'] = self.ce_loss(outputs['hohfeld_pred'], hohfeld_labels) * self.lambda_hohfeld
        else:
            losses['hohfeld'] = torch.tensor(0.0, device=outputs['z_bond'].device)

        # 5. Bond type classification loss
        if bond_labels is not None:
            losses['bond'] = self.bce_loss(outputs['bond_pred'], bond_labels.float())
        else:
            losses['bond'] = torch.tensor(0.0, device=outputs['z_bond'].device)

        # 6. BIP contrastive loss (if isomorphic pairs provided)
        if isomorphic_mask is not None:
            z_bond = outputs['z_bond']
            # Normalize for cosine similarity
            z_norm = F.normalize(z_bond, dim=-1)
            sim_matrix = torch.mm(z_norm, z_norm.t())

            # Positive pairs: high similarity
            # Negative pairs: low similarity
            pos_loss = -torch.log(torch.exp(sim_matrix / 0.07) * isomorphic_mask + 1e-8).mean()
            losses['bip'] = pos_loss * self.lambda_bip
        else:
            losses['bip'] = torch.tensor(0.0, device=outputs['z_bond'].device)

        # Total loss
        total = sum(losses.values())

        # Convert to floats for logging
        loss_dict = {k: v.item() for k, v in losses.items()}
        loss_dict['total'] = total.item()

        return total, loss_dict


# =============================================================================
# TOKENIZER WRAPPER
# =============================================================================

def get_tokenizer(model_name: str = "sentence-transformers/all-mpnet-base-v2"):
    """Get tokenizer for the model."""
    return AutoTokenizer.from_pretrained(model_name)


if __name__ == "__main__":
    # Quick test
    config = {
        'encoder': 'sentence-transformers/all-mpnet-base-v2',
        'd_model': 768,
        'd_bond': 128,
        'd_label': 64,
        'n_time_periods': 9,
        'n_hohfeld_classes': 4
    }

    model = BIPTemporalInvarianceModel(config)
    tokenizer = get_tokenizer()

    # Test input
    texts = ["This is a test passage about saving a life.",
             "Another passage about obligations."]
    encoded = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')

    # Forward pass
    with torch.no_grad():
        outputs = model(encoded['input_ids'], encoded['attention_mask'])

    print("Model test passed!")
    print(f"z_bond shape: {outputs['z_bond'].shape}")
    print(f"z_label shape: {outputs['z_label'].shape}")
