import sys
from pathlib import Path

# Add repo root and apps/api so that "api" and "ml" and "worker" are importable
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "apps" / "api"))
sys.path.insert(0, str(ROOT / "apps" / "worker"))
