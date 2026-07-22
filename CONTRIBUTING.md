# Contributing to GraphAISMNet-Website

Thank you for helping improve the molecular prediction platform!

## Setup for Development

### 1. Clone and Setup

```bash
git clone https://github.com/EuphoRiya37/GraphAISMNet-Website.git
cd GraphAISMNet-Website
```

### 2. Create Environment

```bash
conda create -n graphaisnet python=3.10
conda activate graphaisnet
conda install -c conda-forge rdkit
pip install -r requirements.txt
```

### 3. Run Development

```bash
streamlit run app.py --logger.level=debug
```

## What to Work On

- Improve UI/UX
- Add new visualization options
- Optimize inference speed
- Implement result caching
- Add export formats (SDF, MOL2)
- Create tutorials
- Improve documentation
- Add accessibility features

## Before Submitting

- Test with various SMILES strings
- Run linter: `pylint graphaisnet/`
- Check for memory leaks
- Test CSV upload with edge cases
- Update documentation
- Add comments for complex logic

## Reporting Issues

When reporting bugs:
- Describe the issue clearly
- Include error messages/screenshots
- Provide example SMILES
- Mention your environment (Python version, OS)

## Questions?

Open an issue or discussion in the repository!

---

**License:** MIT + Commons Clause (see LICENSE file)
