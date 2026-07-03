"""PyTorch MLP model definition for churn classification."""

from __future__ import annotations

from typing import Any

torch: Any = None
nn: Any = None
TORCH_IMPORT_ERROR: Exception | None = None


def ensure_torch_is_available() -> None:
    """Raise a clear error when PyTorch cannot be imported."""
    global TORCH_IMPORT_ERROR, nn, torch

    if torch is None:
        try:
            import torch as torch_module
            from torch import nn as nn_module
        except (ImportError, OSError) as torch_error:
            # O ambiente pode nao ter PyTorch instalado ou pode falhar ao
            # carregar DLL nativa. A mensagem fica clara para quem for treinar.
            TORCH_IMPORT_ERROR = torch_error
            details = f" Detalhe tecnico: {TORCH_IMPORT_ERROR}"
            raise RuntimeError(
                "PyTorch nao esta disponivel neste ambiente. "
                "Instale ou ajuste a instalacao com `pip install torch` antes de treinar a MLP."
                f"{details}"
            ) from torch_error

        torch = torch_module
        nn = nn_module


class ChurnMLP:
    """Factory that returns the PyTorch MLP used for binary churn classification."""

    def __new__(
        cls,
        input_size: int,
        hidden_size: int = 64,
        dropout: float = 0.2,
    ) -> Any:
        ensure_torch_is_available()

        class _ChurnMLP(nn.Module):
            """Concrete PyTorch module created only after torch is available."""

            def __init__(self) -> None:
                super().__init__()

                # A MLP recebe as features pre-processadas e passa por camadas
                # densas. ReLU adiciona nao linearidade; Dropout reduz overfitting.
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
                # O ultimo layer nao usa sigmoid porque BCEWithLogitsLoss ja
                # aplica sigmoid internamente de forma numericamente estavel.
                return self.network(x).squeeze(1)

        return _ChurnMLP()
