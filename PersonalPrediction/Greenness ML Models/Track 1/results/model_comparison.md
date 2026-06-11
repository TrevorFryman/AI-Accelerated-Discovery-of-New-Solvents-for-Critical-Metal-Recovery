# GSK Solvent G-Score — Model Comparison Report

| Pipeline | Features | Train RMSE | Train R² | CV RMSE (Mean ± Std) | CV R² (Mean ± Std) | Test RMSE | Test R² |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Pipeline A (Morgan) | 2048 | 0.403 | 0.892 | 0.916 ± 0.115 | 0.399 ± 0.202 | 0.952 | 0.566 |
| Pipeline B (RDKit) | 217 | 0.057 | 0.998 | 0.784 ± 0.133 | 0.552 ± 0.200 | 0.800 | 0.694 |
| Pipeline C (ChemBERTa) | 768 | 0.056 | 0.998 | 1.177 ± 0.078 | 0.041 ± 0.157 | 1.415 | 0.042 |
