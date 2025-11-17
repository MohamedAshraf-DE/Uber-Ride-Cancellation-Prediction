import streamlit as st
import pandas as pd
import joblib
import numpy as np
from pathlib import Path
import base64
from fpdf import FPDF
import requests
from streamlit_lottie import st_lottie
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------
# Config + paths
# -----------------------
st.set_page_config(page_title="Ride Cancellation Predictor", page_icon="🚖", layout="wide")

BASE_DIR = Path(__file__).parent

MODEL_PATH = BASE_DIR / "ride_cancel_model.pkl"
LOGO_PATH = BASE_DIR / "uber-icon.png"
IMAGE_PATH = BASE_DIR / "Car.jpg"

SUCCESS_SOUND = BASE_DIR / "great-success-384935.mp3"
FAIL_SOUND = BASE_DIR / "cartoon-fail-trumpet-278822.mp3"

# -----------------------
# Load and encode image
# -----------------------
@st.cache_data
def get_image_as_base64(file_path):
    try:
        with open(file_path, "rb") as f:
            data = f.read()
        return base64.b64encode(data).decode()
    except FileNotFoundError:
        return None

img_base64 = get_image_as_base64(IMAGE_PATH)

# -----------------------
# CSS Styling
# -----------------------
if img_base64:
    background_css = f"""
    [data-testid="stAppViewContainer"] {{
        background-image: url("data:image/jpeg;base64,{img_base64}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
        color: #FAFAFA;
        position: relative;
    }}
    [data-testid="stAppViewContainer"]::before {{
        content: "";
        position: fixed;
        top: 0; left: 0;
        width: 100%; height: 100%;
        background: rgba(10, 5, 30, 0.7);
        backdrop-filter: blur(4px);
        -webkit-backdrop-filter: blur(4px);
        z-index: -1;
    }}
    """
else:
    background_css = """
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        color: #FAFAFA;
    }
    """

st.markdown(f"""
    <style>
    {background_css}
    ::-webkit-scrollbar {{ width: 10px; }}
    ::-webkit-scrollbar-track {{ background: rgba(0,0,0,0.2); }}
    ::-webkit-scrollbar-thumb {{ background-color: #2ECC71; border-radius: 10px; border: 2px solid rgba(0,0,0,0.2); }}
    ::-webkit-scrollbar-thumb:hover {{ background-color: #27AE60; }}
    [data-testid="stSidebar"] {{ 
        background: rgba(0, 0, 0, 0.4); 
        backdrop-filter: blur(15px); 
        -webkit-backdrop-filter: blur(15px); 
        border-right: 1px solid rgba(255, 255, 255, 0.1); 
        padding: 1.5rem; 
        transition: all 0.3s ease;
        max-height: 100vh; 
        overflow-y: auto;  
        overflow-x: hidden; 
        scrollbar-width: thin;
        scrollbar-color: #2ECC71 rgba(0, 0, 0, 0.2);
    }}
    .results-container {{ text-align: center; margin-top: 20px; }}
    .success-box {{
        background-color: rgba(46, 204, 113, 0.2);
        border: 2px solid #2ECC71;
        color: #2ECC71;
        padding: 20px;
        border-radius: 10px;
        font-size: 24px;
    }}
    .error-box {{
        background-color: rgba(231, 76, 60, 0.2);
        border: 2px solid #E74C3C;
        color: #E74C3C;
        padding: 20px;
        border-radius: 10px;
        font-size: 24px;
    }}
    .footer-text {{ text-align: center; color: #bdc3c7; font-size: 0.9em; margin-top: 50px; }}
    </style>
""", unsafe_allow_html=True)

# -----------------------
# Load model safely
# -----------------------
@st.cache_resource(show_spinner=False)
def load_model_safe(path):
    if path.exists():
        st.write(f"Loading model from: {path}")  # Debug info
        return joblib.load(path)
    else:
        st.error(f"Model file not found at: {path}")
        return None

MODEL_PATHS = [BASE_DIR / "ride_cancel_model.pkl", Path("ride_cancel_model.pkl")]
model = None
for p in MODEL_PATHS:
    model = load_model_safe(p)
    if model is not None:
        break

expected_features = getattr(model, "feature_names_in_", None)

# -----------------------
# Sounds
# -----------------------
SOUNDS_AVAILABLE = SUCCESS_SOUND.exists() and FAIL_SOUND.exists()
def play_sound_simple(file_path: Path):
    if file_path.exists():
        st.audio(str(file_path), format="audio/mp3", autoplay=True)
    else:
        st.warning(f"🔇 Sound file not found: {file_path.name}")

# -----------------------
# Session state
# -----------------------
if "last_result" not in st.session_state: st.session_state.last_result = None
if "user_interacted" not in st.session_state: st.session_state.user_interacted = False

# -----------------------
# Sidebar
# -----------------------
with st.sidebar:
    st.header("⚙️ Controls")
    enable_sounds = st.checkbox("🔊 Enable Sounds", value=SOUNDS_AVAILABLE)
    sound_method = st.radio("Sound Playback:", ["Auto-Play", "Manual Play"], index=0)
    show_charts = st.checkbox("📊 Show detailed charts", value=False)
    
    with st.expander("🔊 Sound Files Status"):
        st.write(f"**Success sound:** {'✅' if SUCCESS_SOUND.exists() else '❌'}") 
        st.write(f"**Fail sound:** {'✅' if FAIL_SOUND.exists() else '❌'}")
        if not SOUNDS_AVAILABLE: st.error("❌ Some sound files are missing from the root folder!")
    
    st.markdown("---")
    
    st.header("📝 Enter Ride Details")
    with st.form("input_form"):
        rd = st.number_input("📍 Ride Distance (km)", 0.0, 500.0, 10.0, step=0.5)
        vt = st.selectbox("🚘 Vehicle Type", ["Auto","Bike","Go Mini","Go Sedan","Premier Sedan","Uber XL","eBike"])
        bv = st.number_input("💰 Booking Value ($)", 0.0, 10000.0, 50.0, step=1.0)
        pm = st.selectbox("💳 Payment Method", ["Cash","Credit Card","Debit Card","UPI","Uber Wallet","Unknown"])
        hr = st.slider("⏰ Hour of Day", 0, 23, 12)
        dr = st.slider("🚗 Driver Rating", 1.0, 5.0, 4.2, 0.1)
        cr = st.slider("⭐ Customer Rating", 1.0, 5.0, 4.5, 0.1)
        wd = st.selectbox("📅 Day of Week", ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"])
        weekend = st.checkbox("🎉 Weekend?")
        predict_button = st.form_submit_button("Predict Cancellation →")

# -----------------------
# Header
# -----------------------
cols = st.columns([0.15, 0.7, 0.15])
with cols[0]:
    if LOGO_PATH.exists(): st.image(str(LOGO_PATH), width=100)
    else: st.warning(f"Logo not found at {LOGO_PATH}")
with cols[1]:
    st.markdown("<h1 style='text-align: center;'>🚖 Ride Cancellation Predictor</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center;'>Will your ride be completed or cancelled?</h3>", unsafe_allow_html=True)
with cols[2]:
    try: anim = st_lottie(requests.get("https://assets7.lottiefiles.com/packages/lf20_iwmd6pyr.json").json(), height=100, key="header_lottie")
    except: pass

st.markdown("---")

# -----------------------
# Prediction Logic
# -----------------------
if predict_button:
    st.session_state.user_interacted = True
    st.subheader("📊 Prediction Result")
    
    if model is None:
        st.error("Cannot make prediction: Model file not loaded.")
    else:
        df = pd.DataFrame({
            "Customer Rating": [cr],
            "Driver Ratings": [dr],
            "Booking Value": [min(bv, 1500)],
            "Ride Distance": [rd],
            "hour": [hr],
            "Avg VTAT": [8.3],
            "Avg CTAT": [28.8],
            "Vehicle Type": [vt],
            "Payment Method": [pm]
        })
        df_enc = pd.get_dummies(df, columns=['Vehicle Type','Payment Method'], drop_first=False)
        if expected_features is not None:
            for f in expected_features:
                if f not in df_enc.columns: df_enc[f] = 0
            df_enc = df_enc[expected_features]

        override = None
        override_reason = None
        if rd > 100 and bv < 0.2 * rd:
            override = 1
            override_reason = "Long distance with too low booking value"
        elif cr < 2.0 or dr < 2.0:
            override = 1
            override_reason = "Very low customer or driver rating (<2.0)"

        if override is not None:
            pred = int(override)
            proba = np.array([0.0, 1.0]) if pred==1 else np.array([1.0, 0.0])
            st.warning(f"⚠️ **Rule-based override triggered:** {override_reason}")
        else:
            proba = model.predict_proba(df_enc)[0]
            pred = model.predict(df_enc)[0]

        st.session_state.last_result = {"Prediction": int(pred), "Prob_Completed": float(proba[0]), "Prob_Cancelled": float(proba[1])}

        with st.container():
            st.markdown('<div class="results-container">', unsafe_allow_html=True)
            if pred==1:
                st.markdown(f'<div class="error-box"><span>CANCELLED</span><br><strong>{proba[1]:.1%} Likelihood</strong></div>', unsafe_allow_html=True)
                if enable_sounds and SOUNDS_AVAILABLE and st.session_state.user_interacted: play_sound_simple(FAIL_SOUND)
            else:
                st.markdown(f'<div class="success-box"><span>COMPLETED</span><br><strong>{proba[0]:.1%} Likelihood</strong></div>', unsafe_allow_html=True)
                if enable_sounds and SOUNDS_AVAILABLE and st.session_state.user_interacted: play_sound_simple(SUCCESS_SOUND)
            st.markdown("</div>", unsafe_allow_html=True)

# -----------------------
# Footer
# -----------------------
st.markdown("---")
st.markdown("<div class='footer-text'>🚖 Ride Cancellation Predictor • Powered by Machine Learning</div>", unsafe_allow_html=True)
