import os
from pathlib import Path

_default_cartridges_dir = str(Path(__file__).parent.parent.absolute())

if os.environ.get("CARTRIDGES_DIR") is None:
    os.environ["CARTRIDGES_DIR"] = _default_cartridges_dir

if os.environ.get("CARTRIDGES_OUTPUT_DIR") is None:
    # Use an 'output' folder in the default dir as a sensible default
    os.environ["CARTRIDGES_OUTPUT_DIR"] = os.path.join(_default_cartridges_dir, "output")
    os.makedirs(os.environ["CARTRIDGES_OUTPUT_DIR"], exist_ok=True)
