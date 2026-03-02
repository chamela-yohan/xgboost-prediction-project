# 🏠 House Price Prediction using XGBoost

An end-to-end machine learning project designed to predict house prices using the **XGBoost (eXtreme Gradient Boosting)** algorithm. This project focuses on explaining the underlying mathematical concepts and providing a clear, step-by-step implementation.

---

## 🎯 Project Overview
The goal of this project is to build a robust regression model to predict housing prices based on features like median income, house age, and location. It is structured to be "Interview Ready," showcasing both coding skills and theoretical understanding.

### Key Highlights:
* **Algorithm:** XGBoost Regressor
* **Dataset:** California Housing Dataset
* **Key Focus:** Model Interpretability & Mathematical Foundations
* **Metrics:** RMSE, R-squared Score

---

## 🧠 Core Concepts & Mathematics

### What is XGBoost?
XGBoost is an ensemble learning method that builds multiple decision trees sequentially. Each new tree corrects the errors (residuals) made by the previous trees.

### The Objective Function
XGBoost minimizes the following objective function:
$$Obj(\theta) = L(\theta) + \Omega(\theta)$$

* **$L(\theta)$:** Loss function (e.g., Mean Squared Error).
* **$\Omega(\theta)$:** Regularization term to prevent **Overfitting**.

### Tree Splitting Logic
We use the **Similarity Score** and **Gain** to determine the best splits for each tree:
$$Similarity Score = \frac{(\sum Gradient)^2}{\sum Hessian + \lambda}$$

---

## 🛠️ Tech Stack
* **Language:** Python 3.x
* **Libraries:** `xgboost`, `scikit-learn`, `pandas`, `numpy`, `matplotlib`
* **Tool:** Jupyter Notebook

---

## 🚀 How to Run
1. Clone the repository:
   ```bash
   git clone [https://github.com/chamela-yohan/xgboost-prediction-project.git](https://github.com/chamela-yohan/xgboost-prediction-project.git)

2. Install dependencies:
    pip install -r requirements.txt

3. Open main_project.ipynb in Jupyter Notebook and run all cells.

## 📊 Results & Insights
* Model Accuracy: The model achieved an R-squared score of 0.8141.
* Top Features: Through Feature Importance (Gain), we found that Median Income is the strongest predictor of house prices.

## 📁 Project Structure
* main_project.ipynb: The main workflow with code and documentation.
* Project overview and documentation.
* Files to be ignored by Git.
