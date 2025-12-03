# Myers-Briggs Personality Type Indicator (MBTI) Prediction

This project utilizes Natural Language Processing (NLP) and machine learning techniques to predict a person's Myers-Briggs personality type based on their social media posts. By analyzing the text content, sentiment, and grammatical structure of user posts, the model classifies individuals into one of the 16 MBTI personality types (e.g., INFJ, ENFP).

## Project Overview

The Myers-Briggs Type Indicator classifies personalities across 4 dichotomies:

  * **I**ntroversion (I) vs. **E**xtroversion (E)
  * **I**ntuition (N) vs. **S**ensing (S)
  * **T**hinking (T) vs. **F**eeling (F)
  * **J**udging (J) vs. **P**erceiving (P)

Instead of a single 16-class classification problem, this project tackles it as **four separate binary classification tasks**, one for each axis. This approach allows for more targeted and accurate predictions.

## Repository Structure

  * **`Data Cleaning.ipynb`**: The primary preprocessing hub. It handles data cleaning (removing URLs, handling emojis), lemmatization, and feature engineering (POS tagging, sentiment scores).
  * **`Model_Implementation_o.ipynb`**: Contains the training logic. It implements various models (Logistic Regression, SVM, XGBoost, etc.) and handles class imbalance techniques like SMOTE and undersampling.
  * **`model_testing.ipynb`**: A script to load saved models and run predictions on the test set or new custom strings.
  * **`sentimental_analysis and pos_tagging.ipynb`**: Dedicated notebook for extracting sentiment polarity (VADER) and Part-of-Speech tags as features.
  * **`Bag of words and TF-IDF.ipynb`**: Exploratory work on text vectorization methods.

## Feature Engineering & Preprocessing

To get the most out of the text data, several preprocessing steps were applied:

1.  **Text Cleaning**: Removal of URLs, emails, punctuation, and MBTI-specific words (to prevent data leakage).
2.  **Lemmatization**: Reducing words to their base root using NLTK's WordNetLemmatizer.
3.  **Sentiment Analysis**: Using VADER (Valence Aware Dictionary and sEntiment Reasoner) to calculate compound, positive, negative, and neutral sentiment scores for every user.
4.  **POS Tagging**: Counting the average usage of parts of speech (Nouns, Verbs, Adjectives, etc.) to capture linguistic style.
5.  **Structural Features**: Counts of emojis, question marks, exclamation marks, and unique words.

## Models & Methodology

The project compares several machine learning algorithms coupled with **TF-IDF** and **Count Vectorization**. Because the dataset has class imbalances (e.g., more Introverts than Extroverts), techniques like **Random Under Sampling**, **Random Over Sampling**, and **SMOTE** were employed to stabilize training.

**Models Evaluated:**

  * Logistic Regression (Standard, Lasso, Ridge)
  * Naive Bayes (Multinomial)
  * Support Vector Classifier (Linear SVC)
  * Random Forest
  * XGBoost

**Best Performance:**
XGBoost and Logistic Regression generally provided the most balanced results across the four axes, evaluated using ROC-AUC and Geometric Mean scores.

## Getting Started

### Prerequisites

You will need Python installed with the following libraries:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn nltk plotly imbalanced-learn xgboost vaderSentiment
```

### Running the Project

1.  **Data Prep**: Run `Data Cleaning.ipynb` first to process the raw `mbti_1.csv` into training and testing CSVs.
2.  **Training**: Execute `Model_Implementation_o.ipynb`. This will train the classifiers for each of the 4 personality axes and save the best models using `joblib`.
3.  **Prediction**: Use `model_testing.ipynb` to load the saved models and predict the personality type of new text inputs.

### Example Usage

```python
from joblib import load

# Load your models
EorI_model = load("clf_is_Extrovert.joblib")
# ... load other 3 models ...

# Predict on new text
prediction = EorI_model.predict(processed_text)
```

## Results

The models output 4 binary predictions which are concatenated to form the final MBTI type (e.g., 0-0-1-0 becomes "INTP").

  * **I vs E**: \~75% ROC-AUC
  * **N vs S**: \~80% ROC-AUC
  * **F vs T**: \~87% ROC-AUC
  * **J vs P**: \~68% ROC-AUC

*(Note: Scores vary depending on the specific sampling strategy and model used).*
