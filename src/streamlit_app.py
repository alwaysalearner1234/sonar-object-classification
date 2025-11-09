import streamlit as st
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

# -------------------------------
# 🧠 App Configuration
# -------------------------------
st.set_page_config(page_title="SONAR Object Classification", layout="centered")

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
DEFAULT_DATA = DATA_DIR / "sonar.csv"

# -------------------------------
# 🎯 App Title
# -------------------------------
st.title("🎯 SONAR Object Classification (Mines vs Rocks)")
st.markdown(
    "This Streamlit app classifies SONAR signals as **Mine (M)** or **Rock (R)** "
    "using a Random Forest Machine Learning model. "
    "You can upload your own dataset or use synthetic demo data."
)

# -------------------------------
# ⚙️ Sidebar Controls
# -------------------------------
st.sidebar.header("⚙️ Options")
uploaded = st.sidebar.file_uploader("Upload SONAR CSV (60 features + label M/R)", type=["csv"])
train_button = st.sidebar.button("Train / Retrain Model")
predict_mode = st.sidebar.radio("Prediction Mode", ["From Data Table", "Manual Input"])

# -------------------------------
# 📂 Load or Create Dataset (SAFE)
# -------------------------------
def load_data():
    """Load dataset from upload, default file, or generate synthetic data safely."""
    # 1️⃣ Check for uploaded file
    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded, header=None)
            if df.empty or df.shape[1] < 2:
                raise ValueError("Uploaded file is empty or invalid.")
            return df, True
        except Exception as e:
            st.error(f"❌ Error loading uploaded file: {e}")
            st.info("Using synthetic demo data instead.")

    # 2️⃣ Check for default sonar.csv
    if DEFAULT_DATA.exists():
        try:
            if DEFAULT_DATA.stat().st_size == 0:
                raise ValueError("Default sonar.csv file is empty.")
            df = pd.read_csv(DEFAULT_DATA, header=None)
            if df.empty or df.shape[1] < 2:
                raise ValueError("Default sonar.csv is invalid or corrupted.")
            return df, True
        except Exception as e:
            st.warning(f"⚠️ Problem with default sonar.csv: {e}")
            st.info("Switching to synthetic demo data.")

    # 3️⃣ Synthetic fallback
    np.random.seed(0)
    X = np.random.rand(208, 60)
    y = np.random.choice(['M', 'R'], size=208)
    df = pd.DataFrame(np.column_stack([X, y]))
    st.success("✅ Synthetic dataset generated successfully.")
    return df, False

# Load data safely
df, is_real = load_data()
if df is None:
    st.stop()

st.caption(f"📊 Data source: {'Uploaded/Real dataset' if is_real else 'Synthetic demo data'}")
st.dataframe(df.head())

# -------------------------------
# 🧩 Prepare Data
# -------------------------------
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values
le = LabelEncoder()
y_enc = le.fit_transform(y)

# -------------------------------
# 💾 Load Existing Model
# -------------------------------
scaler_path = MODELS_DIR / "scaler.pkl"
model_path = MODELS_DIR / "model.pkl"
scaler, model = None, None

if scaler_path.exists() and model_path.exists():
    scaler = joblib.load(scaler_path)
    model = joblib.load(model_path)
    st.success("✅ Pre-trained model and scaler loaded successfully!")

# -------------------------------
# 🧠 Train or Retrain Model
# -------------------------------
def train_model(X, y_enc):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.3, random_state=42, stratify=y_enc
    )
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = RandomForestClassifier(n_estimators=150, random_state=42)
    model.fit(X_train_s, y_train)
    preds = model.predict(X_test_s)
    acc = accuracy_score(y_test, preds)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, scaler_path)
    joblib.dump(model, model_path)

    report = classification_report(y_test, preds, target_names=["Rock(0)", "Mine(1)"])
    cm = confusion_matrix(y_test, preds)
    return acc, report, cm, scaler, model

# Train model if requested or not yet loaded
if train_button or (scaler is None or model is None):
    with st.spinner("Training model... Please wait ⏳"):
        acc, rep, cm, scaler, model = train_model(X, y_enc)

    st.success(f"🎉 Training complete! Model accuracy: **{acc*100:.2f}%**")
    st.text(rep)

    # Plot confusion matrix
    fig, ax = plt.subplots()
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title("Confusion Matrix", fontsize=13)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Rock", "Mine"])
    ax.set_yticklabels(["Rock", "Mine"])
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, v, ha='center', va='center', color="black")
    st.pyplot(fig)

# -------------------------------
# 🔮 Prediction Section
# -------------------------------
st.subheader("🔮 Predict Object Type")

if predict_mode == "From Data Table":
    st.caption("Select a row index from the dataset to classify.")
    row_idx = st.number_input("Row index", min_value=0, max_value=len(df)-1, value=0, step=1)
    if st.button("Predict Selected Row"):
        x = X[row_idx:row_idx+1]
        x_s = scaler.transform(x)
        pred = model.predict(x_s)[0]
        label = "Mine (1)" if pred == 1 else "Rock (0)"
        st.info(f"Prediction for row {row_idx}: **{label}**")

else:
    st.caption("Use manual input sliders (or defaults) to test predictions.")
    f1 = st.slider("Feature 1", 0.0, 1.0, 0.5, 0.01)
    x = np.zeros((1, 60))
    x[0, 0] = f1
    if st.button("Predict Manual Input"):
        x_s = scaler.transform(x)
        pred = model.predict(x_s)[0]
        label = "Mine (1)" if pred == 1 else "Rock (0)"
        st.success(f"Prediction: **{label}**")

# -------------------------------
# 🏁 Footer
# -------------------------------
st.markdown("---")
st.markdown("💡 *Developed as part of the SONAR Object Classification Project using Python, Scikit-learn, and Streamlit.*")
