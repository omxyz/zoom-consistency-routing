def pytest_addoption(parser):
    parser.addoption(
        "--optional",
        action="store_true",
        default=False,
        help="run tests marked @pytest.mark.optional",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--optional"):
        config.option.markexpr = ""
