# dna-hamming-qc
Coherent Hamming-weight circuits for approximate DNA alignment (+ hwb case study).

## Setup
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pytest -q                                              # should print: 4 passed
```
Open the folder in VS Code, install the **Claude Code** and **Google Colab** (+ Jupyter) extensions, open `notebooks/01_verification_v2.ipynb`.
