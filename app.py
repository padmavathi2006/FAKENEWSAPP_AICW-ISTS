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
# CUSTOM STYLE
# -------------------------------------------------
st.markdown(
    """
    <style>
    /* Sidebar width */
    section[data-testid="stSidebar"] {
        width: 380px !important;
    }

    /* Sidebar headings */
    section[data-testid="stSidebar"] h1, 
    section[data-testid="stSidebar"] h2, 
    section[data-testid="stSidebar"] h3 {
        font-size: 24px;
        font-weight: bold;
    }

    /* Sidebar text */
    section[data-testid="stSidebar"] p, 
    section[data-testid="stSidebar"] li {
        font-size: 17px;
    }

    /* Main button */
    div.stButton > button {
        width: 100%;
        font-size: 18px;
        font-weight: bold;
        height: 3em;
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
# INPUT AREA
# -------------------------------------------------
news_text = st.text_area("✍️ Enter News Text Here:", height=220, placeholder="Paste the news content...")

# -------------------------------------------------
# PREDICT BUTTON
# -------------------------------------------------
if st.button("🔍 Predict"):

    if news_text.strip() == "":
        st.warning("⚠️ Please enter some news text.")
    else:
        transformed_input = vectorizer.transform([news_text])
        prediction = model.predict(transformed_input)

        st.divider()

        if prediction[0] == "FAKE":
            st.error("🚨 Prediction: This News is FAKE🚨")
        else:
            st.success("✅ Prediction: This News is REAL✅")

# -------------------------------------------------
# FOOTER
# -------------------------------------------------
st.divider()
st.caption("Built under AICW initiative | Microsoft • SAP • Edunet Foundation")
