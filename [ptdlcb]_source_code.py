import joblib
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

# Chuẩn bị dữ liệu
st.title("Model Comparison for Click Prediction")

X = data.drop(['click'], axis=1)
y = data['click']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Các mô hình phân loại
models = {
    'Logistic Regression': LogisticRegression(),
    'SVM': SVC(),
    'Nearest Neighbors': KNeighborsClassifier(),
    'Linear SVM': SVC(kernel='linear'),
    'RBF SVM': SVC(kernel='rbf'),
    'Decision Tree': DecisionTreeClassifier(),
    'Naive Bayes': GaussianNB(),
    'Random Forest': RandomForestClassifier(),
    'AdaBoost': AdaBoostClassifier(),
    'Gradient Boosting': GradientBoostingClassifier()
}

best_model = None
best_score = 0

# Chạy tất cả các mô hình và tính độ chính xác
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

# Lưu mô hình tốt nhất nếu cần
save_model = st.checkbox("Save Best Model")
if save_model:
    joblib.dump(best_model, f"{best_model_name}_model.pkl")
    st.success(f"Model saved as '{best_model_name}_model.pkl'")

# Hiển thị các mô hình và độ chính xác của chúng
st.subheader("Model Comparison")
model_scores = {model_name: accuracy_score(y_test, model.predict(X_test)) for model_name, model in models.items()}
st.write(model_scores)
streamlit run app.py


     
