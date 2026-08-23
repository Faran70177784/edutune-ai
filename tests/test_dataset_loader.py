from training.dataset_loader import load_training_datasets


def test_training_datasets_load():
    datasets = load_training_datasets()

    assert "train" in datasets
    assert "validation" in datasets
    assert "test" in datasets


def test_training_dataset_sizes():
    datasets = load_training_datasets()

    assert len(datasets["train"]) == 51
    assert len(datasets["validation"]) == 6
    assert len(datasets["test"]) == 7


def test_required_training_columns():
    datasets = load_training_datasets()

    required = {"prompt", "response"}

    for dataset in datasets.values():
        assert required.issubset(set(dataset.column_names))


def test_training_dataset_is_not_empty():
    datasets = load_training_datasets()

    for dataset in datasets.values():
        assert len(dataset) > 0