import types


def executor(code: str) -> None:
    """Execute train.py source code in an isolated module context."""
    module = types.ModuleType("__main__")
    module.__file__ = "train.py"
    exec(compile(code, "train.py", "exec"), module.__dict__)
