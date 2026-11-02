def pytest_configure(config):
    config.addinivalue_line(
        'markers',
        'gpu: test requires a GPU or a real model',
    )
