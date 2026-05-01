def pytest_addoption(parser):
    parser.addoption("--incumbent", default=None, help="Incumbent model spec for benchmark_compare")
    parser.addoption("--candidate", default=None, help="Candidate model spec for benchmark_compare")
