# 🔋 Energy Output Prediction Model

🚀 **Machine Learning model to predict energy output based on environmental conditions.**

## 📌 Overview
This project builds a **Random Forest Regressor** to predict **power plant energy output** (`PE`) using various environmental factors:
- **AT** (Ambient Temperature) 🌡️
- **V** (Exhaust Vacuum) 🔥
- **AP** (Ambient Pressure) 🌬️
- **RH** (Relative Humidity) 💧

🎯 **Goal**: Optimize energy efficiency and reduce operational costs by improving energy output predictions.

🔗 **Colab Notebook**: [Run in Google Colab](https://colab.research.google.com/drive/1yZ_mJ3MWhYQbYGA_SMqSLoUMR137ONES?usp=sharing)

---

## 🛠️ Tech Stack
- **Machine Learning:** Random Forest, Decision Trees, Linear Regression
- **Libraries:** Pandas, Scikit-learn, Matplotlib, Seaborn
- **Deployment:** Google Colab

---

## 📊 Data Preprocessing & Exploration
### **Step 1: Load and Explore Data**
- Load dataset from `CCPP_data.csv`
- Check for missing values and statistical summaries
- Plot a **correlation matrix** to analyze feature relationships

### **Step 2: Visualizations**
- Scatter plots for feature relationships with energy output (`PE`)
- Helps in understanding trends and dependencies

### **Step 3: Splitting Data**
- Dataset is split **80% training / 20% testing** for better generalization

```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

---

## 🤖 Model Training & Evaluation
### **Step 4: Train Multiple Models**
- **Linear Regression** *(Baseline model)*
- **Decision Tree Regressor** *(Handles non-linearity, may overfit)*
- **Random Forest Regressor** *(Final optimized model)*

```python
from sklearn.ensemble import RandomForestRegressor
final_forest_reg = RandomForestRegressor(n_estimators=200, max_depth=20, max_features='sqrt', min_samples_split=2, random_state=42)
final_forest_reg.fit(X_train, y_train)
```

### **Step 5: Hyperparameter Tuning**
- Used **Grid Search CV** to find optimal hyperparameters
- Improved model accuracy and reduced overfitting

```python
from sklearn.model_selection import GridSearchCV
param_grid = {'n_estimators': [50, 100, 200], 'max_features': ['sqrt', 'log2', None], 'max_depth': [10, 20, 30, None], 'min_samples_split': [2, 5, 10]}
grid_search = GridSearchCV(RandomForestRegressor(), param_grid, cv=5, verbose=2, n_jobs=-1)
grid_search.fit(X_train, y_train)
print("Best Parameters:", grid_search.best_params_)
```

---

## 📈 Final Model Results
- **Mean Absolute Error (MAE)**: Measures average prediction error
- **Mean Squared Error (MSE)**: Penalizes large errors more than MAE
- **R-squared Score (R²)**: Measures model accuracy

```python
final_y_pred = final_forest_reg.predict(X_test)
final_r2 = r2_score(y_test, final_y_pred)
print(f"Final Model R-squared: {final_r2:.2f}")
```

### **Feature Importance Plot** 📊
- Helps understand which variables impact predictions the most

```python
importances = final_forest_reg.feature_importances_
features = X_train.columns
plt.barh(features, importances, color='skyblue')
plt.xlabel('Importance')
plt.title('Feature Importance in Final Model')
plt.show()
```

---

## 📂 How to Use This Project
1. Open the **Google Colab Notebook**: [Run Here](https://colab.research.google.com/drive/1yZ_mJ3MWhYQbYGA_SMqSLoUMR137ONES?usp=sharing)
2. Upload the dataset (`CCPP_data.csv`) to Colab or host it on Google Drive
3. Run all code cells to preprocess data, train models, and evaluate results

---

## 📬 Contact
- **Author**: Darrel Nitereka
- **Email**: darrelnitereka@gmail.com
- **LinkedIn**: [linkedin.com/in/darrel-nitereka](https://linkedin.com/in/darrel-nitereka)
- **Portfolio**: [darreln15.github.io/](https://darreln15.github.io/)

---
⚡ Built for energy optimization and AI-driven insights. 🚀