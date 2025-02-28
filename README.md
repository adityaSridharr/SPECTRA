# SPECTRA: Semantic Perturbation-Based Counterfactuals and Training for Robustness Against Adversarial Attacks

## Overview
SPECTRA (Semantic Perturbation-based Counterfactuals for Robust Adversarial Training) is a novel framework that enhances interpretability and robustness in deep learning models. It generates counterfactuals by applying minimal perturbations in the embedding space of CNNs, focusing on fine-grained semantic attributes. This project utilizes the CUB-200 dataset and demonstrates improved model stability against adversarial attacks.


## Methodology
1. **Dataset**  
   - Uses the CUB-200 dataset with 312 semantic attributes across 200 bird species.
   - Enables fine-grained, interpretable perturbations.

2. **Model Architecture**
   - Implemented models include a linear classifier, simple CNN, complex CNN, and ResNet.
   - Attribute vector embeddings aid in semantic counterfactual generation.

3. **Counterfactual Generation**
   - Perturbations applied in embedding space to achieve minimal feature changes.
   - Uses optimization techniques to generate meaningful attribute shifts.

4. **Robustness Evaluation**
   - Introduces the **SPECTRA Attackability Metric**, measuring perturbation resistance.
   - Evaluates adversarial stability by computing minimal norm perturbations.

## Key Features
- **Semantic Counterfactuals**: Generates interpretable counterfactuals based on attribute shifts.
- **Robust Adversarial Training**: Improves model resilience against adversarial perturbations.
- **Minimal Perturbation Calculation**: Ensures targeted and localized feature changes.
- **Attackability Metric**: Provides a quantitative measure of model robustness.

## Results

### Quantitative Metrics:
- Attack Success Rate (ASR)
- Perturbation Noise Ratio
- Peak Signal-to-Noise Ratio (PSNR)

### Qualitative Analysis:
- Localized attribute modifications
- Visualization of perturbed features

## Acknowledgments

This project was developed by **Aditya Sridhar and Ananya Varshney** as part of research on adversarial robustness in deep learning.

## Contact
For inquiries, please reach out to:

- **Aditya Sridhar** - aditya.sridharr.11@gmail.com
- **GitHub**: [My GitHub profile](https://github.com/adityaSridharr)

