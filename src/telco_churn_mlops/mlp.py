"""PyTorch MLP model definition for churn classification."""

from __future__ import annotations

try:
    import torch
    from torch import nn
except ImportError:
    # O ambiente de desenvolvimento pode ainda nao ter PyTorch instalado.
    # Mantemos o modulo importavel para testes e mensagens de erro amigaveis.
    torch = None
    nn = None

BaseModule = nn.Module if nn is not None else object


def ensure_torch_is_available() -> None:
    """Raise a clear error when PyTorch is not installed."""
    if torch is None:
        raise RuntimeError(
            "PyTorch nao esta instalado neste ambiente. "
            "Instale com `pip install torch` antes de treinar a MLP."
        )


class ChurnMLP(BaseModule):
    """Simple multilayer perceptron for binary churn classification."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        dropout: float = 0.2,
    ) -> None:
        ensure_torch_is_available()
        super().__init__()

        # A MLP recebe as features pre-processadas e passa por camadas densas.
        # ReLU adiciona nao linearidade; Dropout ajuda a reduzir overfitting.
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, x):
        """Return raw logits for binary classification."""
        # O ultimo layer nao usa sigmoid aqui porque BCEWithLogitsLoss ja aplica
        # sigmoid internamente de forma numericamente mais estavel.
        return self.network(x).squeeze(1)
