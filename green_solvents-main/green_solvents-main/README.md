
# Green Solvent Prediction  

This repository contains the model and data files for predicting solvent sustainability based on the GlaxoSmithKline (GSK) Solvent Sustainability Guide. Solvent sustainability is quantified using the Composite Greenness Score (G-Score), which incorporates Environment, Health, Safety, and Waste (EHSW) considerations and Life Cycle Assessment (LCA) factors.
  

## Repository Contents  

1. **GSK_dataset.csv**: This csv file contains the GSK-based Composite Greenness Score (G-score) data for 154 solvents.  
3. **Greener_Replacement_Candidate_Sets**: A folder containing greener replacements for the target undesirable solvent, ranked based on Hansen Solubility Parameter (HSP) based Relative Energy Difference (RED) along with Gaussian Process Regression ML predicted G-scores. A lower RED value indicates greater similarity to the target solvent, and a higher G-score indicates a more sustainable solvent.
   - Each alternative is labeled as “high-fidelity RED” (based on validated HSP dataset) or “low-fidelity RED” (based on estimated HSP dataset), serving as an indicator of HSP-derived confidence and to caution against potential inaccuracies associated with lower-fidelity data.
4. **GreenSolventDB**: A folder containing machine learning-predicted Greenness Scores (G-scores).

Please cite: Datta, R., Nistane, J., Sose, A., Sahu, H., & Ramprasad, R. Machine Learning for Green Solvents: Assessment, Selection and Substitution. Advanced Science, e16851 (2025). https://doi.org/10.1002/advs.202516851

## Source Code Access
All recipients must adhere to the terms and conditions outlined in the GTRC Academic Research Use License.
