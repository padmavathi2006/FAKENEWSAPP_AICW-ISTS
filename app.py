import streamlit as st
import pickle

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="Fake News Detection",
    page_icon="📰",
    layout="wide"
)

# -------------------------------------------------
        #BACKGROUND + STYLE
# -------------------------------------------------
st.markdown(
    """
    <style>

    /* Background image */
    .stApp {
        background-image: url("https://images.unsplash.com/photo-1504711434969-e33886168f5c");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }

    /* Dark overlay */
    .stApp::before {
        content: "";
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: rgba(0,0,0,0.55);
        z-index: 0;
    }

    /* Content above overlay */
    .block-container {
        position: relative;
        z-index: 1;
        color: white;
    }

    /* ---------------- SIDEBAR ---------------- */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #c7d2fe, #a5b4fc);
        width: 380px !important;
    }

    section[data-testid="stSidebar"] * {
        color: #0f172a !important;
        font-weight: 600;
    }
    /* ---------------- TEXT AREA BLACK ---------------- */
    textarea {
        background-color: #111111 !important;
        color: #ffffff !important;
        border-radius: 12px !important;
        border: 1px solid #333333 !important;
        padding: 12px !important;
        font-size: 16px !important;
    }

/* placeholder text */
    textarea::placeholder {
        color: #aaaaaa !important;
    }

     /* Label */
    label {
        color: white !important;
        font-weight: bold !important;
        font-size: 18px !important;
    }
    /* ---------------- BUTTON ---------------- */
    div.stButton > button {
        width: 100%;
        font-size: 18px;
        font-weight: bold;
        height: 3em;
        border-radius: 10px;
    }

    </style>
    """,
    unsafe_allow_html=True
)

# -------------------------------------------------
# SIDEBAR
# -------------------------------------------------
st.sidebar.title("AICW")
st.sidebar.write("**Artificial Intelligence Career for Women**")
st.sidebar.write("Microsoft & SAP in collaboration with Edunet Foundation")

st.sidebar.divider()

st.sidebar.title("👥 Team Information")
st.sidebar.write("**Team Members:**")
st.sidebar.write("- Venkata Padmavathi Nagulakonda")
st.sidebar.write("- Sushma Madhavarapu")
st.sidebar.write("- Charitha Makana")
st.sidebar.write("- Roopika Yedida")

st.sidebar.divider()

st.sidebar.title("PROJECT GUIDE")
st.sidebar.write("### Abdul Aziz Md")
st.sidebar.write("Master Trainer - Edunet Foundation")

# -------------------------------------------------
# MAIN HEADER
# -------------------------------------------------
st.markdown(
    "<h1 style='text-align:center;'>📰 Fake News Detection Using Machine Learning</h1>",
    unsafe_allow_html=True
)

st.info("Enter a news article or headline below and the model will predict whether it is REAL or FAKE.")

# -------------------------------------------------
# LOAD MODEL
# -------------------------------------------------
@st.cache_resource
def load_model():
    with open("NEW_LRmodel.pkl", "rb") as f:
        model = pickle.load(f)

    with open("NEW_VECTmodel.pkl", "rb") as f:
        vectorizer = pickle.load(f)

    return model, vectorizer


model, vectorizer = load_model()

# -------------------------------------------------
# INPUT
# -------------------------------------------------
news_text = st.text_area(
    "✍️ Enter News Text Here:",
    height=220,
    placeholder="Paste the news content..."
)

# -------------------------------------------------
# PREDICT
# -------------------------------------------------
if st.button("🔍 Predict"):

    if news_text.strip() == "":
        st.warning("⚠️ Please enter some news text.")
    else:
        transformed_input = vectorizer.transform([news_text])
        prediction = model.predict(transformed_input)

        st.divider()

        if prediction[0] == "FAKE":
            st.error("🚨 Prediction: This News is FAKE 🚨")
        else:
            st.success("✅ Prediction: This News is REAL ✅")

# -------------------------------------------------
# FOOTER
# -------------------------------------------------
st.divider()
st.caption("Built under AICW initiative | Microsoft • SAP • Edunet Foundation")
