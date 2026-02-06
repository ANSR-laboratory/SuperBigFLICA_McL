# Model Card: SuperBigFLICA (SBF) for Predicting Clinical Dementia Rating (CDR-SOB)

## Model Details
- **Model Name:** ADNI SuperBigFLICA (SBF) Data Fusion Model  
- **Developers:** Lily Cheng, Lisa Nickerson  
- **Model Date/Version:** Developed in 2025 with SuperBigFLICA v1  
- **Model Type:** Semi-supervised multimodal fusion model  
- **Training Algorithm:** Joint decomposition of 5 neuroimaging modalities with supervised prediction of CDR-SOB  
- **Implementation:** Python 3.9, PyTorch 2.1.2  
- **Contact:**  
  - Lily Cheng (ycheng23@mgh.harvard.edu)  
  - Lisa Nickerson (lnickerson@mclean.harvard.edu)  
- **License:** Research use only  

---

## Intended Use
- **Primary Intended Uses:** Research tool to identify multimodal neuroimaging signatures predictive of cognitive decline.  
- **Primary Intended Users:** Neuroscientists, clinicians, and ML researchers in aging and dementia research.  
- **Out-of-Scope Uses:** Not for clinical diagnosis, treatment decisions, or real-time prediction outside of research datasets (e.g., ADNI-like).  

---

## Training Data
- **Dataset:** Alzheimer’s Disease Neuroimaging Initiative Phase 3 (ADNI-3).  
- **Modalities fused (5):**
  1. Cortical thickness (CT)  
  2. Grey matter volume (GM)  
  3. Amyloid PET (AMY)  
  4. Tau PET (TAU)  
  5. Pial surface area (PSA)  
- **Target variable:** Clinical Dementia Rating – Sum of Boxes (CDR-SOB).  
- **Inclusion criteria:** Participants with complete T1 MRI, amyloid PET, tau PET, demographic data (age, sex), and CDR-SOB.  
- **Exclusion criteria:** Major neurologic/psychiatric conditions, substance abuse, systemic disease, or MRI contraindications.  
- - **Preprocessing:**  
  - **MRI:** fMRIPrep 20.2.7 with FreeSurfer cortical surfaces.  
  - **Amyloid PET:** SUVR maps motion-corrected, averaged, co-registered to T1w MRI, intensity-normalized to cerebellum, warped to MNI152 space.  
    - Input data were **masked to retain only gray matter voxels** to minimize nonspecific white-matter signal.  
  - **Tau PET:** SUVR maps processed similarly (motion correction, cerebellar reference, MNI space).  **masked to retain only gray matter regions** and exclude white matter and cerebellum.
  - **GM density maps:** Derived from structural MRI; **masked to retain only gray matter regions** and exclude white matter.  
  - **Amyloid PET harmonization:** Centiloid scaling applied.  

---

## Evaluation
- **Performance metrics:** pearson correlation between predicted and observed CDR-SOB.  
- **Data splits:** 70% train / 15% validation / 15% test, with participants from the same site grouped to avoid leakage.  
- **Decision thresholds (Output Spatial Maps):** The output spatial maps were thresholded using **mixture-model–based thresholding** in FSL MELODIC (`mmthresh=0.5`). This method estimates the distribution of signal and noise and applies a probabilistic threshold, producing both thresholded maps and posterior probability maps for interpretation. Thresholding was used only for visualization and interpretation, not during model training.  
For more details, see the [FSL MELODIC documentation](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/MELODIC).
- **Approaches to uncertainty:** Randomized splits. During the training, uncertainty was assessed via an internal validation split (held-out participants within ADNI).

---

## Artifacts (written to `output_dir`)

### 1) Weights (checkpoints)
- **`SBF_best_model.pth`** — Model weights at the **best validation epoch** (selected by the highest validation correlation).  
  *Use this for reporting results and applying the model to new data.*
- **`SBF_final_model.pth`** — Model weights at the **end of training** (last epoch).  
  *Kept for reproducibility/audit; not used for reported performance.*

---

### 2) Predictions
- **`pred_train.csv`** — Model predictions on the **training** split (same order as training subjects).  
- **`pred_valid.csv`** — Predictions on the **validation/early-stopping** split (used to select `SBF_best_model.pth`).  
- **`pred_test.csv`** — Predictions on the **held-out test** split (unseen during training/selection).

*Format:* rows = subjects; columns = target(s) (e.g., CDR-SOB). Values are **back-transformed** to the original target scale.

---

### 3) Latent variables (subject loadings)
- **`lat_train.csv`** — **Subject × component** matrix of latent loadings for the **training** subjects.  
- **`lat_test.csv`** — **Subject × component** matrix for the **test** subjects.

*Use these to analyze which components track clinical severity, do clustering, or make figures.*

---

### 4) Model parameters (learned weights)
- **`modality_weights.csv`** — Relative weights linking **modalities → latent components**  
  (*e.g., contribution of CT/GM/AMY/TAU/PSA to each component; shape typically `n_components × n_modalities`*).
- **`prediction_weights.csv`** — Weights linking **latent components → target(s)**  
  (*e.g., component coefficients for predicting CDR-SOB; shape typically `n_components × n_targets`*).

*These files explain how modalities feed into components and how components predict outcomes.*

---

### 5) Performance & logs
- **`best_corr.txt`** — Single value summarizing the **best validation correlation** used to pick the checkpoint.  
- **`best_performance.csv`** — Key performance metrics saved by `get_model_param`  
  (*e.g., correlations/R²/MAE; contents may be a small vector—see code for exact order*).  
- **`loss_all_test.csv`** — Per-epoch loss components on the validation/held-out split  
  (*four columns correspond to the composite loss terms in `loss_SuperBigFLICA_regression`*).

---

### 6) Spatial maps (component images)
- For each **latent component × modality**, the model produces a **spatial map** that shows where in the brain that component expresses variance.  
- Maps are saved in neuroimaging formats (e.g., NIfTI/CIFTI), one per component per modality.  
- Two versions are written:
  - `*_mmthresh.nii.gz` — thresholded map using mixture-modeling (`mmthresh=0.5`), highlighting voxels most likely to represent true signal.  
  - `*_posterior.nii.gz` — posterior probability maps, giving voxelwise probabilities of belonging to the “signal” class.  
- These maps are for **interpretation and visualization** of multimodal components, not used during training.


### Results Visualization
Visualization of component and prediction weights is provided in [`../Visualizations/SBF_ADNI_CDRSOB_component_modality_visualization.pdf`](../Visualizations/SBF_ADNI_CDRSOB_component_modality_visualization.pdf).

---

### Notes
- **Shapes:**  
  - `lat_*`: `n_subjects × n_components`  
  - `modality_weights`: `n_components × n_modalities`  
  - `prediction_weights`: `n_components × n_targets`
- **Ordering:** Subject order in CSVs matches the split construction in your run.  
- **Privacy & Availability:** Access to [preprocessed input images](https://www.dropbox.com/home/You%20Cheng/SuperBigFLICA_McL/ADNI/Flica_inputs), [output spatial maps](https://www.dropbox.com/home/You%20Cheng/SuperBigFLICA_McL/ADNI/ADNI_SBF_CDR_Outputs), and [model weights](https://www.dropbox.com/home/You%20Cheng/SuperBigFLICA_McL/ADNI/ADNI_SBF_CDR_Outputs) (.pth files) is provided via [Dropbox](https://www.dropbox.com/home/You%20Cheng/SuperBigFLICA_McL) for authorized users under the ADNI DUA. [Code and instructions](https://github.com/ANSR-laboratory/SuperBigFLICA_McL/tree/main/Scripts) are publicly available for reproducibility.

---

## Factors
- **Relevant factors:** Age, sex, clinical diagnosis (CN, MCI, AD), and imaging site. These are known to affect brain structure and dementia severity.  
- **Evaluation factors:** Site bias was mitigated by stratified splitting. We verified that there were **no significant differences** in age, sex, education, or diagnosisdistribution across the training, validation, and test splits.
  
---

## Ethical Considerations
- **Data source:** ADNI participants gave informed consent; data are de-identified.  
- **Bias & fairness:** Model inherits ADNI’s demographic skew (primarily highly educated, white, U.S.-based participants aged 55–90). May not generalize globally or to underrepresented groups.  
- **Limitations:**  
  - Associations, not causal mechanisms.  
  - Requires multimodal imaging not widely available in clinics.  
  - Not validated for clinical deployment.  

---

## Citation
If you use this model, please cite:  
> [https://github.com/ANSR-laboratory/SuperBigFLICA_McL](https://github.com/ANSR-laboratory/SuperBigFLICA_McL)
