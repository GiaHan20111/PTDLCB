streamlit run app.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import gdown
import joblib
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import streamlit as st

# Cài đặt cảnh báo
warnings.filterwarnings('ignore')

# Tải dữ liệu từ Google Drive
url = "https://drive.google.com/uc?id=1P6IUd8WQrTAlWZMshN5iV0uDwMIXOcW3"
output = "data.xlsx"
gdown.download(url, output, quiet=False)
data = pd.read_excel(output)

# Hiển thị dữ liệu
st.title("Click Prediction Model")
st.write("Data Preview:")
st.dataframe(data.head())

# Xử lý dữ liệu
data = data.drop(['id', 'full_name'], axis=1)
data = data.dropna()
X = data.drop(['click'], axis=1)
y = data['click']

# Phân chia dữ liệu thành train và test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Định nghĩa các mô hình
models = {
    'Logistic Regression': LogisticRegression(),
    'SVM': SVC(),
    'Nearest Neighbors': KNeighborsClassifier(),
    'Decision Tree': DecisionTreeClassifier(),
    'Naive Bayes': GaussianNB(),
    'Random Forest': RandomForestClassifier(),
    'AdaBoost': AdaBoostClassifier(),
    'Gradient Boosting': GradientBoostingClassifier()
}

# So sánh các mô hình
best_model = None
best_score = 0
best_model_name = ""

for model_name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    score = accuracy_score(y_test, predictions)

    if score > best_score:
        best_score = score
        best_model = model
        best_model_name = model_name

# Hiển thị kết quả trong Streamlit
st.subheader(f"Best Model: {best_model_name} with Accuracy: {best_score:.4f}")

# Hiển thị kết quả so sánh mô hình
st.subheader("Model Comparison")
model_scores = {model_name: accuracy_score(y_test, model.predict(X_test)) for model_name, model in models.items()}
st.write(model_scores)

# Lưu mô hình tốt nhất
save_model = st.checkbox("Save Best Model")
if save_model:
    joblib.dump(best_model, f"{best_model_name}_model.pkl")
    st.success(f"Model saved as '{best_model_name}_model.pkl'")

# Confusion Matrix for Best Model
st.subheader(f"Confusion Matrix for {best_model_name}")
best_model_predictions = best_model.predict(X_test)
cm = confusion_matrix(y_test, best_model_predictions)
fig, ax = plt.subplots(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
ax.set_title(f'Confusion Matrix - {best_model_name}')
st.pyplot(fig)

# Đánh giá mô hình tốt nhất
st.subheader(f"Classification Report for {best_model_name}")
st.text(classification_report(y_test, best_model_predictions))

# Đo lường ROC Curve cho Logistic Regression (hoặc mô hình khác nếu cần)
if best_model_name == 'Logistic Regression':
    from sklearn import metrics
    y_pred_proba = best_model.predict_proba(X_test)[::, 1]
    fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_proba)
    auc = metrics.roc_auc_score(y_test, y_pred_proba)
    st.subheader("ROC Curve")
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"AUC = {auc:.2f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc=4)
    st.pyplot()


     
