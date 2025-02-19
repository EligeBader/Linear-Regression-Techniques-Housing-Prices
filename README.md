# 🏡 Housing Prices Prediction Project 🏡

![Kaggle](https://img.shields.io/badge/Kaggle-Competition-blue) ![Python](https://img.shields.io/badge/Python-3.8%2B-brightgreen) ![Scikit-learn](https://img.shields.io/badge/Scikit--learn-0.24.2-yellow) ![License](https://img.shields.io/badge/License-MIT-orange) ![Regression](https://img.shields.io/badge/Regression-Model-red)

## 🏆 Overview
Welcome to my Housing Prices Prediction Project! This project is part of the Kaggle competition "Housing Prices Competition for Kaggle Learn Users," where the objective is to predict the final price of each home based on a variety of features.

![House](https://img.shields.io/badge/House-Pricing-yellow) ![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Regression-blue)

## 📜 Description
This project focuses on predicting house prices using various regression techniques and neural networks. The dataset used in this project contains 79 explanatory variables that describe different aspects of residential homes in Ames, Iowa. The goal is to predict the sale price of each house based on these features.

## 🏢 Benefits to the Company and Stakeholders

This project brings significant benefits to the company and stakeholders by leveraging machine learning for housing price predictions. By developing models that predict house prices accurately, the company can offer valuable insights to real estate agents, investors, and homebuyers. This enhances decision-making processes and increases market efficiency. Additionally, these models help in identifying trends and patterns in the real estate market, providing a competitive advantage.

Furthermore, implementing these models demonstrates the company's commitment to innovation and advanced technology. It showcases the company's ability to utilize cutting-edge machine learning techniques, thereby attracting potential clients and partners. Collaborating on this project fosters a culture of learning and technological advancement within the company, driving overall growth and success.

## 💾 Dataset
The competition dataset consists of the following files:
- `train.csv`: The training set containing the house features and their corresponding sale prices.
- `test.csv`: The test set for which you need to predict the sale prices.
- `data_description.txt`: A full description of each column in the dataset.
- `sample_submission.csv`: A sample submission file in the correct format.

### Data Fields
Here are some of the main data fields:
- `SalePrice`: The property's sale price in dollars. This is the target variable you're trying to predict.
- `MSSubClass`: The building class.
- `MSZoning`: The general zoning classification.
- `LotFrontage`: Linear feet of street connected to property.
- `LotArea`: Lot size in square feet.
- ... and many more.

## 🛠 Tools & Technologies
For this project, I used the following tools and technologies:
- **Python 3.8+**: The backbone of the project.
- **Pandas**: For data manipulation and analysis.
- **NumPy**: For numerical operations.
- **Scikit-learn**: For building and evaluating regression models.
- **TensorFlow/Keras**: For building and training neural network models.
- **Jupyter Notebook**: For interactive coding and exploration.

## 🔍 Workflow

1. **Data Loading and Preprocessing**:
   - Created functions to load, clean, and encode the data.
   - Handled missing values and encoded categorical variables.

2. **Feature Engineering**:
   - Created new features to enhance the model's performance.
   - Used various encoding techniques and power transformations.

3. **Model Building and Evaluation**:
   - Built multiple regression models including Linear Regression, Random Forest, and XGBoost.
   - Built and trained a neural network model using TensorFlow/Keras.
   - Evaluated the models using Root-Mean-Squared-Error (RMSE) between the logarithm of predicted values and the logarithm of observed sale prices.


4. **Best Model**:
   - Among the models trained, **XGBoost** performed the best in terms of RMSE on the validation set. This model was fine-tuned using GridSearchCV to find the best hyperparameters.
     
5. **Submission**:
   - Predicted sale prices for the test set.
   - Prepared the submission files in the required format.

## 📂 Project Structure
```
- Housing_Prices_Prediction
  - data/
    - train.csv
    - test.csv
    - data_description.txt
    - sample_submission.csv
  - notebooks/
    - training Regression & Neural Network Project.ipynb
    - testing Regression & Neural Network Project.ipynb
  - scripts/
    - define_function.py
  - models/
    - read_file.pickle
    - drop_features.pickle
    - split_data.pickle
    - clean_data.pickle
    - encode_data.pickle
    - transformed_data.pickle
    - trained_model_LR.pickle
    - trained_model_XG.pickle
    - trained_model_rf.pickle
    - trained_nn_model.pickle
  - predictions/
    - prediction_rf.csv
    - prediction_xg.csv
    - prediction_nn.csv
  - README.md
```

## 🎯 Results
After training and evaluating the models, I generated predictions for the test set and prepared the submission files. The results were evaluated based on RMSE between the logarithm of the predicted values and the logarithm of the observed sale prices.

## 🌟 Improvements
To further enhance this project, I can explore the following:
- **Experiment with Different Models**:
  - Try advanced regression techniques like Gradient Boosting.
  - Use ensemble methods to combine predictions from multiple models.

- **Optimize Hyperparameters**:
  - Use techniques like grid search to find the best hyperparameters for the models.

- **Feature Engineering**:
  - Explore additional feature engineering techniques to improve model performance.

## 🙏 Acknowledgements
A big thank you to Kaggle for providing the platform and the dataset for this competition. Special thanks to Dean De Cock for compiling the Ames Housing dataset, which has been instrumental for this project.
