import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBClassifier, plot_importance
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

st.set_page_config(page_title="Bank Churn Predictor", layout="wide")

st.title("🔍 Bank Customer Churn Prediction with XGBoost")

# File upload
uploaded_file = st.file_uploader("Upload your dataset (CSV format)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("Raw Data")
    st.write(df.head())

    st.markdown("---")
    st.subheader("📊 Preprocessing")

    # Outlier removal
    for i in df:
        if df[i].dtype in ['int64', 'float64'] and i not in ['churn', 'customer_id']:
            Q1 = df[i].quantile(0.25)
            Q3 = df[i].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df = df[(df[i] >= lower_bound) & (df[i] <= upper_bound)]

    df = df[~((df['estimated_salary'].isin([0,200000])) & (df['balance'] == 250000) & (df['products_number'].isin([3.0, 4.0])))]

    df['age_bucket'] = pd.cut(df['age'], bins=[0,30,70,100], labels=['Young','Middle-aged','Old'])

    le = LabelEncoder()
    df['gender'] = le.fit_transform(df['gender'])

    df = pd.get_dummies(df, columns= ['country'], drop_first= True, dtype=int)
    df = pd.get_dummies(df, columns= ['age_bucket'], drop_first= True, dtype=int)

    df = df.drop(['customer_id', 'credit_card'], axis=1)

    scaler = StandardScaler()
    df[['credit_score','age','tenure','balance','estimated_salary']] = scaler.fit_transform(
        df[['credit_score','age','tenure','balance','estimated_salary']]
    )

    X = df.drop('churn', axis=1)
    y = df['churn']

    st.success("Preprocessing Complete!")

    st.markdown("---")
    st.subheader("🧠 Model Training & Evaluation")

    if st.button("Train Model"):
        xgb_model = XGBClassifier(
            objective='binary:logistic',
            eval_metric='logloss',
            use_label_encoder=False,
            gamma=0,
            learning_rate=0.05, 
            max_depth=6, 
            n_estimators=200,
            reg_alpha=0.01, 
            reg_lambda=1,
            scale_pos_weight=len(y[y==0]) / len(y[y==1]),
            random_state=42
        )

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        xgb_model.fit(X_train, y_train)

        y_pred = xgb_model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        st.metric("Accuracy", f"{acc:.4f}")
        st.metric("Precision", f"{prec:.4f}")
        st.metric("Recall", f"{rec:.4f}")
        st.metric("F1 Score", f"{f1:.4f}")

        st.markdown("---")
        st.subheader("🔥 Cross-Validation (Precision)")
        scores = cross_val_score(xgb_model, X, y, cv=5, scoring='precision')
        st.write(scores)
        st.metric("Mean Precision:", f"{scores.mean():.4f}")

        st.markdown("---")
        st.subheader("📌 Feature Importance")
        fig, ax = plt.subplots(figsize=(10, 6))
        plot_importance(xgb_model, importance_type='gain', max_num_features=10, height=0.5, ax=ax)
        st.pyplot(fig)