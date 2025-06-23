import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
    AdaBoostClassifier, AdaBoostRegressor,
    BaggingClassifier, BaggingRegressor
)
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import accuracy_score, r2_score
from streamlit_chat import message
import google.generativeai as genai
import io

# ---- Page Setup ----
st.set_page_config(page_title="🧠 MindFlux AI: Talk to Your Data. See the Future.", layout="centered")
st.title("🧠 MindFlux AI")
st.caption("An intelligent ML chatbot powered by CRISP-DM and advanced AI")

# ---- Gemini API Key ----
genai.configure(api_key="AIzaSyCSio6PYfC8qUb2G9guQSfzLjOnAr-vgTQ")  # Replace this with your actual API key securely in production
gemini_model = genai.GenerativeModel("gemini-2.0-flash")

# ---- Chat History ----
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ---- Upload Dataset ----
st.markdown("## 📁 Upload Your CSV Dataset")
file = st.file_uploader("Upload CSV", type=["csv"])

if file:
    df = pd.read_csv(file)
    st.success("✅ Dataset loaded successfully!")
    st.dataframe(df.head())

    # ---- Data Understanding ----
    st.markdown("## 🔍 Data Understanding")
    st.write(df.describe())
    st.write("Missing Value % per column:", df.isnull().mean() * 100)

    # ---- Data Preparation ----
    st.markdown("## 🧹 Data Preparation")
    encoding_choice = st.radio("Encoding Method:", ["Label Encoding", "One-Hot Encoding"])
    use_knn = st.checkbox("Use KNN Imputer instead of SimpleImputer")

    df_clean = df.copy()
    for col in df_clean.columns:
        if df_clean[col].isnull().any():
            null_pct = df_clean[col].isnull().mean()
            strategy = 'most_frequent' if df_clean[col].dtype == 'object' else 'mean' if null_pct <= 0.3 else 'median'
            if use_knn:
                imputer = KNNImputer()
                df_clean.iloc[:, :] = imputer.fit_transform(df_clean.select_dtypes(include=[np.number]))
                break
            else:
                imputer = SimpleImputer(strategy=strategy)
                df_clean[[col]] = imputer.fit_transform(df_clean[[col]])

    st.success("✅ Missing values imputed.")

    # ---- Target Variable ----
    target = st.selectbox("🎯 Select Target Variable", df_clean.columns)

    if target:
        X = df_clean.drop(columns=[target])
        y = df_clean[target]

        if encoding_choice == "Label Encoding":
            for col in X.select_dtypes(include='object'):
                X[col] = LabelEncoder().fit_transform(X[col].astype(str))
        else:
            X = pd.get_dummies(X)

        task_type = "Classification" if y.nunique() <= 15 and y.dtype != float else "Regression"
        st.info(f"Detected Task: **{task_type}**")

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # ---- Model Training ----
        st.markdown("## 🤖 Model Training & Evaluation")
        models = {}

        if task_type == "Classification":
            models = {
                "RandomForest": RandomForestClassifier(),
                "LogisticRegression": LogisticRegression(max_iter=1000),
                "KNN": KNeighborsClassifier(),
                "DecisionTree": DecisionTreeClassifier(),
                "GradientBoosting": GradientBoostingClassifier(),
                "AdaBoost": AdaBoostClassifier(),
                "Bagging": BaggingClassifier(),
                "GaussianNB": GaussianNB()
            }
        else:
            models = {
                "RandomForest": RandomForestRegressor(),
                "LinearRegression": LinearRegression(),
                "KNN": KNeighborsRegressor(),
                "DecisionTree": DecisionTreeRegressor(),
                "GradientBoosting": GradientBoostingRegressor(),
                "AdaBoost": AdaBoostRegressor(),
                "Bagging": BaggingRegressor()
            }

        results = {}
        for name, model in models.items():
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            score = accuracy_score(y_test, preds) if task_type == "Classification" else r2_score(y_test, preds)
            results[name] = score

        best_model_name = max(results, key=results.get)
        best_model = models[best_model_name]
        best_score = results[best_model_name]

        st.success(f"🏆 Best Model: **{best_model_name}** with score: **{best_score:.4f}**")

        st.markdown("## 💬 Chat with MindFlux AI")

        # Optional greeting
        if len(st.session_state.chat_history) == 0:
            st.info("👋 Hello! Send feature values below and I'll predict & explain.")

        user_input = st.text_input("Enter feature values (comma-separated):", key="chat_input")

        if st.button("🧠 Send"):
            try:
                input_list = [float(x.strip()) for x in user_input.split(",")]
                input_df = pd.DataFrame([input_list], columns=X.columns)
                pred = best_model.predict(input_df)[0]
                prediction = f"🔮 Prediction: **{pred}**"

                prompt = f"Explain why {best_model_name} is the best model for a {task_type} task with score {best_score:.4f}."
                gemini_response = gemini_model.generate_content(prompt)
                explanation = gemini_response.text

                full_response = f"{prediction}\n\n🧠 *Why this model?*\n{explanation}"
                st.session_state.chat_history.append({"user": user_input, "ai": full_response})
                st.rerun()

            except Exception as e:
                st.session_state.chat_history.append({"user": user_input, "ai": f"❌ Error: {str(e)}"})
                st.rerun()

        # ---- Display chat messages with feedback buttons ----
        for i, chat in enumerate(st.session_state.chat_history):
            message(chat["user"], is_user=True, key=f"user_{i}", avatar_style="personas")
            message(chat["ai"], is_user=False, key=f"ai_{i}", avatar_style="bottts")

            col1, col2 = st.columns([1, 5])
            with col1:
                if st.button("👍", key=f"like_{i}"):
                    st.toast("Thanks for the feedback! 😊")
                if st.button("👎", key=f"dislike_{i}"):
                    st.toast("We'll try to improve. 🙏")

        # ---- Clear chat history ----
        if st.button("🗑️ Clear Chat"):
            st.session_state.chat_history = []
            st.rerun()

        # ---- Export chat as CSV ----
        if st.button("📥 Download Chat as CSV"):
            if st.session_state.chat_history:
                chat_df = pd.DataFrame(st.session_state.chat_history)
                csv = chat_df.to_csv(index=False)
                st.download_button("📄 Click to Download", data=csv, file_name="mindflux_chat_history.csv", mime="text/csv")
