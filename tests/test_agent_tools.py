import pytest

from genai_engineering.core.errors import ToolError
from genai_engineering.task3_agent.tools.calculator import calculate
from genai_engineering.task3_agent.tools.file_reader import SafeFileReader


def test_calculator_supports_arithmetic_and_functions():
    assert calculate("sqrt(81) + 2**3") == 17


def test_calculator_rejects_code_execution():
    with pytest.raises(ValueError):
        calculate("__import__('os').system('whoami')")


def test_file_reader_confines_paths(tmp_path):
    outside = tmp_path.parent / "outside.txt"
    outside.write_text("secret")
    reader = SafeFileReader(str(tmp_path))
    with pytest.raises(ToolError):
        reader.read("../outside.txt")
