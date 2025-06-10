# SuperBigFLICA: Code Modifications and Applications

This folder contains the presentation slides and supporting materials from:

**SuperBigFLICA: Code Modifications and Applications**  
Presented at the 2nd Meeting of the Users’ Committee for the NWO-VICI project  
*“Big data for precision neuroscience: new tools for brain connectopics”*  
Donders Centre for Cognitive Neuroimaging (DCCN), Nijmegen, Netherlands & virtual · May 2025

---

## About

This presentation outlines key adaptations made to the [original SuperBigFLICA code repository](https://github.com/weikanggong/SuperBigFLICA).

---

## Summary of Code Modifications

The code has been adapted to better support:

- Data Load: flexibly import data in common imaging formats
- Added dictionary learning as a dimensionality reduction step
- Fixed underflow in data preprocessing (normalization step)
- Save imaging outputs in common image formats
- Automated results visualization
- Enhanced reproducibility


---

## Planned Extensions

Looking ahead, we plan to expand the SuperBigFLICA framework in several directions:

- **Surface-based Image Format Compatibility**: Add support for CIFTI file formats.
- **Classification Support**: Extend the model to handle categorical prediction tasks, such as diagnostic group classification.
- **Site Harmonization**: Implement scanner/site effect correction methods to enhance generalizability across pooled datasets.
- **Uncertainty-Aware Evaluation**: Replace single-point test set performance metrics with bootstrap-based estimates to report means and 95% confidence intervals—yielding more stable and statistically robust evaluation metrics.

---

## Acknowledgment

This work builds on the original [SuperBigFLICA GitHub repository](https://github.com/weikanggong/SuperBigFLICA) and is being actively extended as part of the NIA-funded project **RF1AG078304**.

For questions or contributions, please contact [Lily Cheng](YCHENG23@mgh.harvard.edu) (McLean Hospital / Harvard Medical School).
