import os
import sys
from pathlib import Path

# # Enforce UTF-8 for Windows console stdio
# if sys.platform == "win32":
#     os.environ["PYTHONIOENCODING"] = "utf-8"
#     if hasattr(sys.stdout, "reconfigure"):
#         sys.stdout.reconfigure(encoding="utf-8", errors="replace")
#     if hasattr(sys.stderr, "reconfigure"):
#         sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Add src/ to sys.path so rag_app can be imported directly
root_dir = Path(__file__).parent.resolve()
src_dir = root_dir / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from rag_app.cli import main

if __name__ == "__main__":
    sys.exit(main())
