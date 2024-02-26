# Titanic Survival Prediction

This repository contains code for exploring different classification algorithms on the Titanic dataset from Kaggle. The goal of the project is to train various classifiers, optimize hyperparameters, and evaluate their performance on both the validation and test sets. Additionally, we aim to create an ensemble classifier that outperforms individual classifiers.

## Dataset

The Titanic dataset consists of information about passengers aboard the Titanic, including features such as age, sex, ticket class, and whether they survived or not.

The dataset is divided into train and test sets. The train set is used for training and validating models, while the test set is used for final evaluation.

## Files

- `Titanic Survival Prediction.ipynb`: Jupyter Notebook containing the code for data loading, preprocessing, model training, hyperparameter tuning, evaluation, and ensemble creation.
- `train.csv`: Train set of the Titanic dataset.
- `test.csv`: Test set of the Titanic dataset.

## Classification Algorithms Explored

1. **Multinomial Logistic Regression (Softmax Regression):**
   - Utilized softmax regression for multiclass classification.
   - Explored different hyperparameters such as regularization strength.
   - Evaluated performance on training, validation, and test sets.

2. **Support Vector Machines (SVM):**
   - Utilized SVM with various kernels (linear, polynomial, radial basis function).
   - Explored hyperparameters such as C (regularization parameter) and kernel parameters.
   - Analyzed performance on different kernels and their impact on classification.

3. **Random Forest Classifier:**
   - Implemented random forest classifier.
   - Analyzed feature importance to understand which features contribute most to the classification.
   - Tuned hyperparameters such as number of trees and maximum depth of trees.

## Ensemble Classifier

- Combined individual classifiers into an ensemble to improve performance.
- Used a simple averaging method to combine predictions.
- Evaluated ensemble performance on the validation set.

## Results and Discussion

- Performance metrics (accuracy, precision, recall, F1-score) for each classifier on training, validation, and test sets are reported.
- Impact of hyperparameters on model performance is discussed.
- Feature importance analysis for Random Forest Classifier is provided.
- Findings from ensemble classifier and comparison with individual classifiers are discussed.

## Requirements

- Python 3
- Jupyter Notebook
- Libraries: pandas, numpy, scikit-learn

## Usage

1. Clone the repository: `git clone https://github.com/sarfaraj-mohammad/Titanic-Survival-Prediction.git`
2. Open `Titanic Survival Prediction.ipynb` in Jupyter Notebook.
3. Run each cell sequentially to load data, train models, and evaluate performance.
4. Follow instructions and comments within the notebook for customization or further analysis.

## Contributors

Feel free to contribute to the project by opening issues or pull requests.
