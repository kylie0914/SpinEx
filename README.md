# SpinEx-Machine-Learning (SpinEx-ML)
[![DOI](https://sandbox.zenodo.org/badge/817967634.svg)](https://sandbox.zenodo.org/doi/10.5072/zenodo.80783)

<img width="504" alt="Fig1_overview of the SpinEx approach" src="https://github.com/kylie0914/SpinEx/assets/48717355/a6299851-843e-4ad6-bfde-f2ad4a6588ab">

## SpinEx
This is the github repo for the paper "Integrated disc-fluidics enables high-throughput sample processing for the analysis of extracellular vesicles". SpinEx-ML is a practical utility to evaluate SpinEx. We used the obtained data using SpinEx for cancer diagnostics and classification. We focused on five prevalent tumor types, breast, lung, liver, pancreas, and colon, as potential detection targets. In addition, non-cancer control samples were obtained from non-cancer donors.

The overall workflow of SpinEx-ML is shown in below

![Clinical application of SpinEx for cancer detection and classification](https://github.com/user-attachments/assets/2e9a5d74-3a53-40a0-80a4-33f7b7972d22)

## Simple Description
### 1. Cancer diagnostic model
Dx fold in SpinEx_ML contains whole part of ancer diagnostic model.
Three .py files in that folder: Utils.py, CrossValidation_Lasso_Dx.py, repeatedholdout_Lasso_Dx.py.
We recommend to use CrossValidation_Lasso_Dx.py for stratified K-fold evaluation.

### 2. Cancer classification model
Multi folder in SpinEx_ML contains whole part of cancer classification model.
Three .py files in that folder: Utils.py, CancerType_classification.py, Visualization.py.
Here, you can obtain the result of the CancerType_classification.py file and then visualize it as the result of applying t-sne/Umap to the Visualization.py file.

