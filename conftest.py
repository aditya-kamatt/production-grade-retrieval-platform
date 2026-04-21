import gc
import pytest


@pytest.fixture(autouse=True)
def cleanup_after_test():
    """Clean up memory after each test to prevent exhaustion."""
    yield
    gc.collect()  # Force garbage collection
