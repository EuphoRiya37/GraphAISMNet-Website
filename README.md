# GraphAISMNet — Web App

Streamlit deployment for GraphAISMNet, a Graph Attention Network (GAT) model
predicting anti-inflammatory activity of small molecules from SMILES input.

🔗 **Live app:** [graphaismnet-app.streamlit.app](https://graphaismnet-app.streamlit.app)

## What it does
Takes a molecule's SMILES string, builds its molecular graph, and predicts
anti-inflammatory activity using a trained GAT model.

## Files
- `app.py` — Streamlit app (UI + inference)
- `model_gat.pt` — trained GAT model weights
- `GRAPH.png` — visual asset used in the app
- `requirements.txt` / `packages.txt` — Python and system dependencies for Streamlit Cloud

## Related
- Model training code and research: [`Anti-inflammatory-small-molecule-prediction-Research-Paper`](https://github.com/EuphoRiya37/Anti-inflammatory-small-molecule-prediction-Research-Paper)

## Tech stack
Streamlit, PyTorch, PyTorch Geometric, RDKit
