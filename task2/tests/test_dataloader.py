from task1.data.dataloader import CustomDataLoader

test_dataloader = CustomDataLoader()


def test_loading():
    """Tests loading dataset.

    Ensures that dataloader can extract features.
    """
    test_dataloader.load_data()


def test_get_target():
    """Tests loading target.

    Ensures target is accessible.
    """
    test_dataloader.target_column()
