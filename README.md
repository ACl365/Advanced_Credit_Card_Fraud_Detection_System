# Advanced Credit Card Fraud Detection System

A robust, two-stage machine learning pipeline for high-accuracy credit card fraud detection, leveraging ensemble methods and optimised for real-world deployment.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3.0-orange)
![XGBoost](https://img.shields.io/badge/XGBoost-1.7.6-green)
![Streamlit](https://img.shields.io/badge/Streamlit-1.27.0-red)
![License](https://img.shields.io/badge/License-MIT-yellow)

## 📌 Overview

This project implements an advanced machine learning pipeline for credit card fraud detection, emphasizing high recall and precision. The system features a two-stage detection process, combining ensemble methods with optimised thresholds for superior performance in real-world deployment scenarios.

![Project Banner](https://raw.githubusercontent.com/username/credit-card-fraud/main/assets/banner.png)

## 🔍 Key Features

- **Two-Stage Fraud Detection Pipeline**: Implements a specialized approach that prioritizes recall in the first stage and precision in the second stage
- **Stacked Ensemble Architecture**: Combines multiple base models (Logistic Regression, Random Forest, XGBoost, etc.) with meta-learners to improve fraud detection accuracy
- **Imbalanced Learning**: Addresses class imbalance using SMOTE (Synthetic Minority Over-sampling Technique)
- **Optimised Metrics**: Focuses on fraud detection-relevant metrics like recall and AUC
- **Comprehensive Evaluation**: Includes detailed model performance analysis and visualisations
- **Interactive Demo**: Streamlit web application for interactive fraud prediction and model exploration
- **Production-Ready Code**: Modular, well-documented code with proper logging and error handling

## 📊 Project Structure

```
credit-card-fraud/
├── data/                   # Data directory (not included in repo due to size)
├── models/                 # Saved model files
├── results/                # Evaluation results and visualisations
│   └── figures/            # Generated visualisation images
├── src/                    # Source code
│   ├── app/                # Streamlit application
│   │   └── app.py          # Streamlit demo app
│   ├── utils/              # Utility functions
│   └── visualisation/      # Visualisation functions
├── notebooks/              # Jupyter notebooks for exploration
├── main.py                 # Main script to run the pipeline
├── requirements.txt        # Project dependencies
├── setup.py                # Package setup file
├── LICENSE                 # License file
└── README.md               # Project documentation
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/username/credit-card-fraud.git
   cd credit-card-fraud
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

4. Download the dataset:
   - Download the [Credit Card Fraud Detection dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud) from Kaggle
   - Place the `creditcard.csv` file in the project's root directory

### Usage

1. Run the main script to train models and generate results:
   ```bash
   python main.py
   ```

2. Launch the Streamlit demo application:
   ```bash
   streamlit run src/app/app.py
   ```

3. Explore the generated results in the `results/` directory

## 📈 Model Performance

Our stacked ensemble approach achieves superior performance compared to individual base models:

| Model | AUC | Precision | Recall | F1 Score |
|-------|-----|-----------|--------|----------|
| Gradient Boosting Meta-Learner | 0.982 | 0.931 | 0.912 | 0.921 |
| Logistic Regression Meta-Learner | 0.975 | 0.923 | 0.901 | 0.912 |
| XGBoost | 0.970 | 0.919 | 0.887 | 0.903 |
| Random Forest | 0.963 | 0.911 | 0.873 | 0.892 |
| Gradient Boosting | 0.959 | 0.902 | 0.865 | 0.883 |
| Logistic Regression | 0.947 | 0.885 | 0.841 | 0.862 |

*Note: These metrics are based on the test set evaluation.*
## 🔬 Methodology

1. **Data Preprocessing**:
   - Feature scaling using StandardScaler
   - Removal of duplicate transactions
   - Stratified train-test splitting

2. **Class Imbalance Handling**:
   - Applied SMOTE to create synthetic minority class samples
   - Balanced training dataset while preserving test set distribution

3. **Base Models**:
   - Logistic Regression
   - Random Forest
   - Decision Tree
   - K-Nearest Neighbors
   - Naive Bayes
   - Gradient Boosting
   - XGBoost

4. **Meta-Learners**:
   - Logistic Regression
   - Gradient Boosting

5. **Stacking Approach**:
   - Used out-of-fold predictions to create meta-features
   - Selective feature integration at meta-level

6. **Two-Stage Detection Pipeline**:
   - **Stage 1 (High Recall)**: Logistic Regression with lower threshold (0.3) to catch as many potential fraudulent transactions as possible
   - **Stage 2 (High Precision)**: XGBoost with higher threshold (0.7) to reduce false positives
   - This approach balances the operational costs of fraud investigation with the need to catch all fraudulent activity

7. **Hyperparameter Optimisation**:
   - Grid search with cross-validation
   - Grid search with cross-validation

## 📊 Sample Visualisations

### ROC Curves
![ROC Curves](results/figures/roc_curves.png)

### Precision-Recall Curves
![Precision-Recall Curves](results/figures/precision_recall_curves.png)

### Model Comparison
![Model Comparison](results/figures/model_comparison.png)

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 👥 Authors

- **Alexander Clarke** - *Initial work* - [GitHub Profile](https://github.com/alexanderclarke365) - [Email](mailto:alexanderclarke365@gmail.com)

## 📚 References

- Dal Pozzolo, A., Caelen, O., Johnson, R. A., & Bontempi, G. (2015). Calibrating Probability with Undersampling for Unbalanced Classification. In Symposium on Computational Intelligence and Data Mining (CIDM), IEEE.

- Brownlee, J. (2020). Imbalanced Classification with Python: Better Metrics, Balance Skewed Classes, Cost-Sensitive Learning.

- Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002). SMOTE: Synthetic Minority Over-sampling Technique. Journal of Artificial Intelligence Research, 16, 321-357.

- Wolpert, D. H. (1992). Stacked Generalization. Neural Networks, 5(2), 241-259.

## 🔮 Future Improvements

1. **Feature Engineering**:
   - Explore additional derived features from Time and Amount
   - Investigate transaction sequence patterns for fraud detection

2. **Advanced Ensemble Techniques**:
   - Implement feature-weighted linear stacking
   - Explore blending approaches with holdout predictions

3. **Deep Learning Integration**:
   - Incorporate autoencoders for anomaly detection
   - Explore recurrent neural networks for sequence modeling

4. **Model Explainability**:
   - Add SHAP (SHapley Additive exPlanations) values for model interpretability
   - Implement feature importance visualisations

5. **Deployment Enhancements**:
   - Create a RESTful API for real-time fraud detection
   - Develop batch processing capabilities for large transaction volumes

## ❓ Frequently Asked Questions

**Q: Why use ensemble learning for fraud detection?**
A: Ensemble learning combines multiple models to improve overall performance. In fraud detection, different models can capture various fraud patterns, leading to better overall detection rates.

**Q: How does the system handle the class imbalance problem?**
A: The system uses SMOTE (Synthetic Minority Over-sampling Technique) to create synthetic examples of the minority class (fraudulent transactions) during training, while maintaining the original class distribution in the test set for realistic evaluation.

**Q: What metrics should be prioritised for fraud detection?**
A: While accuracy is common for balanced classification problems, fraud detection requires special attention to recall (capturing all frauds) and precision (minimising false positives). AUC-ROC provides a good overall measure of discriminative ability across different threshold settings.

**Q: What is the two-stage approach and why is it effective?**
A: The two-stage approach uses a high-recall model (Logistic Regression) in the first stage to catch as many potential fraudulent transactions as possible, followed by a high-precision model (XGBoost) in the second stage to reduce false positives. This approach is effective because:
- It prioritizes catching all fraud (minimizing false negatives) which is critical for financial institutions
- It reduces the operational burden of investigating false positives by applying a more precise filter in the second stage
- It balances the trade-off between recall and precision better than a single model approach

**Q: How can this model be deployed in a production environment?**
A: For production deployment, consider:
- Containerising the application with Docker
- Setting up real-time prediction API with Flask or FastAPI
- Implementing proper monitoring and retraining pipelines
- Adding data drift detection to alert when model retraining is needed

## ⭐ Acknowledgements

- The anonymous contributors of the Credit Card Fraud dataset
- The scikit-learn, XGBoost, and imbalanced-learn developer communities
- All researchers advancing the field of fraud detection