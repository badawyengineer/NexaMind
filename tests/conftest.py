import pytest


class FakeProvider:
    def __init__(self, replies):
        self.replies = iter(replies)
        self.prompts = []

    def generate(self, prompt, *, system="", json_mode=False):
        self.prompts.append(prompt)
        return next(self.replies)

    def stream(self, prompt, *, system=""):
        yield "mock"


@pytest.fixture
def fake_provider():
    return FakeProvider
