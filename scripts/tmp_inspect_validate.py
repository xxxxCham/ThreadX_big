from pathlib import Path
import sys

sys.path.insert(0, str(Path("d:/ThreadX/src").resolve()))
try:
    import threadx.data.validate as v

    print("module:", getattr(v, "__file__", "no-file"))
    print("has validate_dataset:", hasattr(v, "validate_dataset"))
    print(
        "dir sample:",
        [a for a in dir(v) if "validate" in a.lower() or a == "validate_dataset"],
    )
except Exception as e:
    import traceback

    print("ERROR:", type(e).__name__, e)
    traceback.print_exc()
