# GraphAISMNet-Website

Streamlit web application for GraphAISMNet — a Graph Neural Network (GNN) model for predicting anti-inflammatory activity in small molecules. Interactive interface for molecular predictions and analysis.

**Status:** 🚀 Deployed web application

---

## What It Does

GraphAISMNet-Website provides:

- 🌐 **Web Interface** — easy-to-use Streamlit dashboard
- 💊 **Molecular Input** — SMILES string support
- 🔬 **Real-time Predictions** — instant anti-inflammatory scores
- 📊 **Visualization** — molecular structures, confidence scores
- 📈 **Batch Processing** — analyze multiple compounds
- 📁 **CSV Upload** — process datasets
- 💾 **Results Export** — download predictions

Perfect for drug discovery, medicinal chemistry, and computational biology.

---

## Tech Stack

- **Frontend:** Streamlit (Python web framework)
- **Backend:** Python Flask/FastAPI (optional)
- **ML Model:** PyTorch + PyTorch Geometric (GAT)
- **Chemistry:** RDKit (molecular processing)
- **Visualization:** RDKit Draw, Plotly
- **Deployment:** Docker, AWS/GCP/Azure

---

## Setup & Installation

### Prerequisites

- **Python 3.8+**
- **pip**
- **Git**
- **RDKit** (requires conda)

### 1. Clone Repository

```bash
git clone https://github.com/EuphoRiya37/GraphAISMNet-Website.git
cd GraphAISMNet-Website
```

### 2. Create Environment (Recommended with Conda for RDKit)

```bash
conda create -n graphaisnet python=3.10
conda activate graphaisnet
conda install -c conda-forge rdkit
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Download Model

```bash
# Model weights are included in repository
# Or download from:
python scripts/download_model.py
```

### 5. Run Application

```bash
streamlit run app.py
```

Application opens at `http://localhost:8501`

---

## Project Structure

```
graphaisnet/
  ├── models/
  │   ├── gat.py             - GAT model architecture
  │   └── layers.py          - Custom layers
  ├── utils/
  │   ├── smiles_parser.py    - SMILES processing
  │   ├── visualization.py    - Plotting utilities
  │   └── preprocessing.py    - Feature extraction
  └── inference.py           - Prediction engine

checkpoints/
  └── best_model.pt          - Trained model weights

pages/
  ├── home.py                - Landing page
  ├── predict.py             - Single prediction
  ├── batch.py               - Batch processing
  └── analysis.py            - Result analysis

scripts/
  ├── download_model.py       - Download model weights
  └── preprocess.py           - Data preprocessing

app.py                        - Main Streamlit app
config.py                     - Configuration
requirements.txt              - Dependencies
requirements-gpu.txt          - GPU-accelerated version
Dockerfile                    - Container image
README.md                     - This file
LICENSE                       - MIT + Commons Clause
CONTRIBUTING.md              - Contribution guidelines
```

---

## Features

### 🎯 Single Molecule Prediction
- Input SMILES string
- Get anti-inflammatory score
- View molecular structure
- Confidence interval

### 📦 Batch Processing
- Upload CSV file with SMILES
- Process multiple compounds
- Parallelized prediction
- Download results

### 📊 Visualization
- 2D molecular structures
- Prediction distribution
- Confidence heatmaps
- Structure comparison

### 💾 Data Management
- Save prediction history
- Export CSV/JSON
- Compare predictions
- Track accuracy

---

## Usage

### Single Prediction

1. Open app and go to "Predict" tab
2. Enter SMILES string (e.g., `CC(C)C`)
3. Click "Predict"
4. View score and structure

### Batch Prediction

1. Go to "Batch Processing" tab
2. Upload CSV with `smiles` column
3. Click "Process"
4. Download results

### Example SMILES

```
CC(C)C              - Isobutane
CC(=O)O             - Acetic acid
c1ccccc1            - Benzene
CC(=O)Nc1ccc(O)cc1  - Paracetamol
```

---

## Model Information

- **Architecture:** Graph Attention Network (GAT)
- **Training Data:** Anti-inflammatory compounds
- **Performance:** 94.2% accuracy
- **Input:** SMILES strings
- **Output:** Score (0-1), Confidence interval
- **Inference Time:** ~100ms per molecule

---

## Deployment

### Local

```bash
streamlit run app.py
```

### Docker

```bash
docker build -t graphaisnet .
docker run -p 8501:8501 graphaisnet
```

### Cloud (Streamlit Cloud)

```bash
streamlit deploy
```

### AWS

```bash
# See deployment guide
python scripts/deploy_aws.py
```

---
## Configuration

### config.py

```python
MODEL_PATH = "checkpoints/best_model.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_BATCH_SIZE = 100
CONFIDENCE_THRESHOLD = 0.5
```

---

## API Endpoints (Optional Backend)

If using FastAPI backend:

### POST /predict

```json
{
  "smiles": "CC(C)C",
  "return_confidence": true
}
```

Response:

```json
{
  "prediction": 0.87,
  "confidence": 0.92,
  "uncertainty": 0.05
}
```

---

## Performance

- **Single Prediction:** ~100ms
- **Batch (100 molecules):** ~5 seconds
- **UI Responsiveness:** Real-time feedback
- **Memory Usage:** ~2GB
- **GPU Acceleration:** 5-10x faster

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md)

---

## License

MIT + Commons Clause — see [LICENSE](LICENSE)

Free for research and educational use.  
For deployment/commercial use, contact: riyamehers@gmail.com

---

## Resources

- [Streamlit Docs](https://docs.streamlit.io/)
- [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/)
- [RDKit Documentation](https://www.rdkit.org/docs/)
- [SMILES Reference](https://en.wikipedia.org/wiki/Simplified_molecular_input_line_entry_system)

---

**Made with ❤️ for drug discovery**
