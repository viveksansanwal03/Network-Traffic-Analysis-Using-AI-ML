"""One-command pipeline: generate data -> train -> sample predictions.

Run:  python run_all.py
Then: streamlit run app.py
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent
steps = [
    ["src/generate_data.py", "--samples", "30000"],
    ["src/train.py"],
    ["src/predict.py", "--samples", "500"],
]
for step in steps:
    print(f"\n=== python {' '.join(step)} ===")
    code = subprocess.call([sys.executable, *step], cwd=ROOT)
    if code != 0:
        sys.exit(code)
print("\nDone. Launch the dashboard with:  streamlit run app.py")
