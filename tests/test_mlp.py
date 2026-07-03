import pytest

from telco_churn_mlops import mlp as mlp_runtime
from telco_churn_mlops.mlp import ChurnMLP, ensure_torch_is_available
from telco_churn_mlops.train_mlp import MLPTrainingConfig, split_train_validation


def is_torch_available() -> bool:
    # O PyTorch e importado sob demanda para evitar falhas de DLL durante a
    # coleta dos testes em ambientes onde ele ainda nao esta configurado.
    try:
        ensure_torch_is_available()
    except RuntimeError:
        return False
    return True


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


@pytest.mark.skipif(not is_torch_available(), reason="PyTorch is not available")
def test_churn_mlp_forward_shape_when_torch_is_available():
    # Quando PyTorch esta instalado, validamos que a rede devolve um logit por linha.
    model = ChurnMLP(input_size=4, hidden_size=8, dropout=0.1)
    batch = mlp_runtime.torch.zeros((3, 4), dtype=mlp_runtime.torch.float32)

    output = model(batch)

    assert output.shape == (3,)
