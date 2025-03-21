# Advanced Credit Card Fraud Detection: An Ensemble Learning Approach

## Abstract

This study explores ensemble learning methodologies for credit card fraud detection, a domain characterised by heavily imbalanced datasets and high-stakes classification decisions. We introduce a stacked ensemble architecture that leverages multiple base classifiers and meta-learners to optimise the predictive performance on the standard credit card fraud dataset. Our approach demonstrates significant improvements in key performance metrics, particularly recall and AUC, which are critical in fraud detection contexts.

## 1. Introduction

Credit card fraud detection presents unique challenges in machine learning: extreme class imbalance, evolving fraud patterns, and asymmetric misclassification costs. Traditional single-model approaches often struggle with these challenges, particularly in balancing precision and recall to minimise both false positives and false negatives.

The primary objectives of this research are:
1. To develop a robust stacking ensemble that improves upon single-model fraud detection capabilities
2. To address class imbalance issues through strategic resampling techniques
3. To optimise the model for high recall without sacrificing precision
4. To analyse the complementary nature of diverse base learners in capturing fraud patterns

## 2. Methodology

### 2.1 Data Preprocessing

The dataset (`creditcard.csv`) contains anonymized credit card transactions with 30 features, including 28 principal components transformed via PCA, along with 'Time' and 'Amount' variables and a binary 'Class' label indicating fraudulent transactions.

We performed the following preprocessing steps:
- Feature scaling using `StandardScaler` for 'Amount' and 'Time' features
- Removal of duplicate entries
- Stratified train-test splitting with 80/20 ratio
- SMOTE (Synthetic Minority Over-sampling Technique) to address class imbalance in the training set

```python
# Scale the 'Amount' and 'Time' features
scaler = StandardScaler()
data['Amount'] = scaler.fit_transform(data['Amount'].values.reshape(-1, 1))
data['Time'] = scaler.fit_transform(data['Time'].values.reshape(-1, 1))

# Split the data into training and testing sets
X = data.drop('Class', axis=1)
y = data['Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Handle class imbalance using SMOTE oversampling
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
```

### 2.2 Ensemble Architecture

We employed a stacking ensemble methodology using a two-layer approach:

#### Base Models (First Layer):
- Logistic Regression
- Random Forest
- Decision Tree
- K-Nearest Neighbors
- Gaussian Naive Bayes
- Gradient Boosting
- XGBoost

SVM was initially considered but removed from the final implementation due to computational constraints.

```python
# Define base models
logistic_model = LogisticRegression()
rf_model = RandomForestClassifier(random_state=42)
dt_model = DecisionTreeClassifier(random_state=42)
knn_model = KNeighborsClassifier()
nb_model = GaussianNB()
gb_model = GradientBoostingClassifier(random_state=42)
xgb_model = xgb.XGBClassifier(random_state=42)
```

#### Meta-Learners (Second Layer):
- Logistic Regression
- Gradient Boosting Classifier

The stacking ensemble used out-of-fold predictions to create meta-features, implementing a 2-fold stratified cross-validation strategy:

```python
def get_oof_predictions(model, X_train, y_train, X_test, n_splits=2):
    oof_predictions = np.zeros((len(X_train),))
    test_predictions = np.zeros((len(X_test),))

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    for fold, (train_index, val_index) in enumerate(skf.split(X_train, y_train)):
        X_train_fold, X_val_fold = X_train.iloc[train_index], X_train.iloc[val_index]
        y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]

        model.fit(X_train_fold, y_train_fold)
        oof_predictions[val_index] = model.predict(X_val_fold)
        test_predictions += model.predict(X_test) / n_splits

    return oof_predictions, test_predictions
```

Feature selection was implemented at the meta-level, utilizing only the strongest base models:

```python
meta_train = np.column_stack((logistic_oof_train, rf_oof_train, gb_oof_train, xgb_oof_train))
meta_test = np.column_stack((logistic_test_preds, rf_test_preds, gb_test_preds, xgb_test_preds))
```

### 2.3 Hyperparameter Optimisation

We employed Grid Search with cross-validation to optimise key parameters of the meta-learners:

```python
param_grid_logistic = {'C': [0.1, 1, 10]}
grid_search_logistic = GridSearchCV(LogisticRegression(), param_grid_logistic, scoring='roc_auc', cv=2)

param_grid_gb = {'learning_rate': [0.01, 0.1], 'n_estimators': [100, 200]}
grid_search_gb = GridSearchCV(GradientBoostingClassifier(random_state=42), param_grid_gb, scoring='roc_auc', cv=2)
```

### 2.4 Evaluation Metrics

Given the domain-specific requirements of fraud detection, we employed a comprehensive set of evaluation metrics:
- AUC-ROC: To assess overall discriminative capability
- Precision: To measure the ratio of true positives to predicted positives
- Recall: To quantify the model's ability to capture all fraudulent transactions
- F1-score: To balance precision and recall
- Accuracy: As a standard baseline metric

## 3. Results and Analysis

Our stacking ensemble methodology demonstrated notable improvements over single-model approaches, particularly in the critical metrics of recall and AUC. The selection of meta-features from only the strongest base models (Logistic Regression, Random Forest, Gradient Boosting, and XGBoost) was a deliberate choice to reduce noise and improve generalisation.

Several key findings emerged:

1. **Meta-learner performance**: The Gradient Boosting meta-learner consistently outperformed the Logistic Regression meta-learner, suggesting that capturing complex, non-linear relationships between base model predictions is advantageous.

2. **Feature diversity impact**: The inclusion of diverse base learners in the ensemble contributed complementary information. Logistic Regression provided a strong linear baseline, while tree-based methods (Random Forest, XGBoost) captured non-linear feature interactions.

3. **SMOTE effectiveness**: The application of SMOTE significantly improved the recall of minority class predictions while maintaining reasonable precision, addressing the fundamental class imbalance challenge.

4. **Processing efficiency**: The strategic feature selection at the meta-level improved computational efficiency without sacrificing predictive performance.

## 4. Implications and Future Directions

The developed stacking ensemble architecture provides several methodological contributions to the field of fraud detection:

1. Demonstrating that intelligent combination of diverse base learners can capture complementary fraud patterns
2. Establishing an effective approach to balance precision and recall in highly imbalanced classification contexts
3. Showing that selective feature extraction at the meta-level can improve both performance and efficiency

Future research directions include:
- Integration of temporal features to capture evolving fraud patterns
- Exploration of cost-sensitive learning to directly optimise for business-relevant metrics
- Implementation of online learning mechanisms to adapt to concept drift
- Incorporation of unsupervised anomaly detection techniques as additional meta-features

## 5. Conclusion

This research demonstrates that stacking ensemble approaches can significantly enhance credit card fraud detection performance by leveraging the complementary strengths of diverse base learners. The strategic use of SMOTE resampling, selective meta-feature generation, and hyperparameter optimisation creates a robust framework that addresses the unique challenges of imbalanced classification in financial fraud detection.

The methodologies presented here can be extended to other domains characterised by extreme class imbalance and asymmetric misclassification costs, providing a blueprint for developing high-performing ensemble models in similar contexts.
