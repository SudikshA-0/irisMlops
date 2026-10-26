from src.train import train_model


def test_training_produces_good_model():
    metadata = train_model()
    assert metadata["accuracy"] >= 0.9
    assert metadata["classes"] == ["setosa", "versicolor", "virginica"]
