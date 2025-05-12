# Supervised Big-data FMRIB’s Linked Independent Component Analysis (SuperBigFLICA)

SuperBigFLICA is a semi-supervised, multi-modal data fusion framework designed to extract interpretable spatial components from neuroimaging data while jointly predicting clinical outcomes. This implementation supports multimodal brain imaging analysis and is tailored for studies in aging, dementia, and related neurological conditions.

## Features

- **Blends multiple brain‑scan types in one model** – MRI, fMRI, DTI or other measures can be analysed together.
- **Finds common patterns across people** – the code learns a small set of “scores” for each participant that capture how strongly they show those brain patterns.
- **Predicts clinical or behavioural outcomes at the same time** – training is guided both by how well the scans are reconstructed and how well the model forecasts the target variable (e.g., symptom severity).
- **Keeps a separate map for each scan type** – you still get modality‑specific brain maps that are easy to visualise.
- **Flexible and hardware‑friendly** – adjustable hyper‑parameters, runs on CPU or GPU, and includes ready‑made dataset and training utilities.

## File Structure

- `SBF.py`: Main training and testing script for the SuperBigFLICA model
- `utils.py`: Supporting functions
  
## Requirements

- Python ≥ 3.11 (tested with 3.11.9)
- PyTorch
- NumPy
- Pandas
- NiBabel
- SciPy
- scikit-learn
- Matplotlib
- Joblib

## Getting Started

1. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
