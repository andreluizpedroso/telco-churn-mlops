import pytest

from telco_churn_mlops.mlp import ChurnMLP, torch
from telco_churn_mlops.train_mlp import MLPTrainingConfig, split_train_validation


def test_mlp_training_config_defaults_are_valid():
    # O config concentra hiperparametros que serao usados no treino da MLP.
    config = MLPTrainingConfig()

    assert config.hidden_size > 0
    assert 0 <= config.dropout < 1
    assert config.learning_rate > 0
    assert config.batch_size > 0
    assert config.patience > 0


def test_split_train_validation_keeps_both_classes(sample_churn_dataframe):
    # A validacao tambem precisa preservar exemplos de churn e nao churn.
    train_df, validation_df = split_train_validation(sample_churn_dataframe, validation_size=0.25)

    assert set(train_df["Churn"]) == {0, 1}
    assert set(validation_df["Churn"]) == {0, 1}


@pytest.mark.skipif(torch is None, reason="PyTorch is not installed")
def test_churn_mlp_forward_shape_when_torch_is_available():
    # Quando PyTorch esta instalado, validamos que a rede devolve um logit por linha.
    model = ChurnMLP(input_size=4, hidden_size=8, dropout=0.1)
    batch = torch.zeros((3, 4), dtype=torch.float32)

    output = model(batch)

    assert output.shape == (3,)
