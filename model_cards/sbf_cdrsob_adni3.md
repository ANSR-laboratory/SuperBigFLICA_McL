# Model Card: SuperBigFLICA (SBF) for Predicting Clinical Dementia Rating (CDR-SOB)

## Model Details
- **Developers**: Neuroimaging researchers using ADNI-3 data  
- **Model Date/Version**: Developed in 2025 using SuperBigFLICA v1  
- **Model Type**: Semi-supervised multimodal fusion model  
- **Training Algorithm**: Joint decomposition of 5 neuroimaging modalities with supervised prediction of CDR-SOB  
- **Implementation**: Python 3.9, PyTorch 2.1.2  
- **Contact**: Lily Cheng (YCHENG23@mgh.harvard.edu), Lisa Nickerson (lisa_nickerson@hms.harvard.edu)
- **License**: Research use only  

## Intended Use
- **Primary Intended Uses**: Research tool to identify multimodal neuroimaging signatures predictive of cognitive decline
- **Primary Intended Users**: Neuroscientists, clinicians, and ML researchers in aging and dementia research  
- **Out-of-Scope Uses**: Not for clinical diagnosis, treatment decisions, or real-time prediction outside the context of ADNI-like datasets  

## Factors
- **Relevant Factors**: 
Model performance may vary by participant characteristics and data source. Key factors include:
	•	Age, sex, and clinical diagnosis (CN, MCI, AD), which influence brain structure and dementia severity.
	•	Imaging site, due to differences in scanner hardware and acquisition protocols.
These factors were selected based on known effects on brain imaging and clinical outcomes.

- **Evaluation Factors**:
To reduce site-related bias, participants from the same imaging site were kept within the same data split. We did not evaluate performance separately by age, sex, or diagnosis, but these are important for future analysis and fairness assessment.

## Metrics
- **Model Performance Measures**: Predictive accuracy of CDR-SOB (e.g., R², MAE to be reported)  
- **Decision Thresholds**: Spatial maps thresholded at top/bottom 5% (95th percentile)  
- **Approaches to Uncertainty and Variation**: Random data splits (70% train, 15% validation, 15% test); participants from the same site assigned to the same data split to prevent site-related leakage  

## Evaluation Data
- **Datasets**: Alzheimer’s Disease Neuroimaging Initiative Phase 3 (ADNI-3)  
- **Motivation**: ADNI-3 provides high-quality, multi-modal imaging and clinical data representative of cognitive aging and dementia  
- **Preprocessing**:  
  - MRI: fMRIPrep 20.2.7 with FreeSurfer-derived cortical surfaces  
  - Amyloid and tau PET: SUVR maps motion-corrected, averaged, co-registered to T1w MRI, intensity-normalized to cerebellum, warped to MNI152 space  
  - Amyloid PET harmonized to Centiloid units  

## Training Data
- **Source**: Same as evaluation (ADNI-3), one scan per participant with complete imaging and clinical data  
- **Inclusion Criteria**: Complete data for T1 MRI, amyloid PET, tau PET, age, sex, and CDR-SOB  
- **Exclusion Criteria**: Major neurologic or psychiatric disorders, substance abuse, significant systemic disease, skilled-nursing residence, or contraindications for MRI  

## Quantitative Analyses
- **To Be Added**: CDR-SOB prediction metrics across test set and subgroups  
- **Planned Analyses**: Stratification by sex, clinical diagnosis (CN, MCI, AD), and amyloid status  

## Ethical Considerations
- **Data Source**: ADNI participants provided informed consent; data are de-identified  
- **Bias/Fairness**: No explicit fairness constraints applied; subgroup evaluation recommended for demographic fairness assessment  

## Caveats and Recommendations
- **Limitations**:  
  - ADNI participants are U.S.-based, aged 55–90, and may not represent general clinical populations  
  - Results may not generalize to data acquired with different imaging protocols  
  - Model not intended for deployment in clinical settings  
- **Recommendations**:  
  - Evaluate model on external cohorts before clinical translation  
  - Incorporate disaggregated performance metrics and explore bias mitigation strategies in future iterations
