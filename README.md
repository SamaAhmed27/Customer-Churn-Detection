# 📌 Customer Churn Detection

## 📖 Project Overview
Customer churn prediction helps businesses identify customers who are likely to leave a service. This project builds a machine learning model to predict customer churn using historical data.

## 📂 Dataset
The dataset used is **P1_Churn_Modelling.csv**, which includes:
- **Numerical Features**: `CreditScore`, `Age`, `Balance`, `EstimatedSalary`, etc.
- **Categorical Features**: `Gender`, `Geography`
- **Target Variable**: `Exited` (1 = Churned, 0 = Not Churned)

## 🛠️ Workflow
### 1️⃣ Data Preprocessing
- Handled missing values & removed unnecessary columns.
- Encoded categorical features (`Gender`, `Geography`).
- Standardized numerical features.

### 2️⃣ Exploratory Data Analysis (EDA)
- Visualized feature distributions.
- Analyzed correlation between features.
- Checked for class imbalance.

### 3️⃣ Feature Engineering
- One-Hot Encoding for `Geography`
- Label Encoding for `Gender`
- Applied **SMOTE** to handle class imbalance for the target feature.

### 4️⃣ Model Training & Evaluation
Trained & evaluated the following models:
- **K-Nearest Neighbors (KNN)**
- **Naive Bayes**
- **Support Vector Machine (SVM)**
- **Decision Tree (DT)**

### 5️⃣ Model Comparison
Compared models using:
- **Accuracy**
- **Precision & Recall**
- **F1-Score**
- **ROC-AUC Score**

## 🏆 Results & Best Model
- The best-performing model based on evaluation metrics was **SVM**.
- Fine-tuned models using **GridSearchCV**.
- Applied **a max depth and tuned hyperparameters for the decision tree model** to improve performance.

## 📂 How to Run the Project
1. Clone the repository:
   ```sh
   git clone https://github.com/SamaAhmed27/Customer-Churn-Detection.git
   cd Customer-Churn-Detection
   ```
2. Install dependencies:
   ```sh
   pip install -r requirements.txt

3. Run the code:


## 💾 Saving & Loading Models
Save models:
```python
import joblib
joblib.dump(model, "saved_models/model.pkl")
```
Load models:
```python
model = joblib.load("saved_models/model.pkl")
```

## 📝 Future Improvements
- Try **Deep Learning models (ANNs)**.
- Use **Hyperparameter Tuning** for optimization.
- Deploy model using **Flask or Streamlit**.

## 🤝 Contributing
Pull requests are welcome! Feel free to fork and contribute. 😊


---
📌 **Author**: Sama Ahmed
📧 **Contact**: samaabdelaal275@gmail.com
🌟 **GitHub**: SamaAhmed27
