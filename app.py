import streamlit as st
import pandas as pd
import joblib
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import base64
from fpdf import FPDF

# Lottie
try:
    from streamlit_lottie import st_lottie
    import requests
except Exception:
    st.error("Install: pip install streamlit-lottie requests fpdf2 seaborn matplotlib")
    st.stop()

# -----------------------
# Config + paths
# -----------------------
st.set_page_config(page_title="Ride Cancellation Predictor", page_icon="🚖", layout="wide")

# --- PATHS ---
# !! IMPORTANT: This is the path to your main project folder !!
BASE_DIR = Path(r"C:\Users\PCCV\OneDrive - Alexandria National University\Desktop\streamlit")

MODEL_PATH = BASE_DIR / "ride_cancel_model.pkl"
LOGO_PATH = BASE_DIR / "logos" / "uber-icon.png"
SOUNDS_PATH = BASE_DIR / "sounds"
IMAGE_PATH = BASE_DIR / "Car.jpg" # Your new background image

SUCCESS_SOUND = SOUNDS_PATH / "great-success-384935.mp3"
FAIL_SOUND = SOUNDS_PATH / "cartoon-fail-trumpet-278822.mp3"


# --- Function to load and encode image ---
@st.cache_data
def get_image_as_base64(file_path):
    try:
        with open(file_path, "rb") as f:
            data = f.read()
        return base64.b64encode(data).decode()
    except FileNotFoundError:
        # NOTE: Using st.error here will cause the app to fail if files aren't present.
        # This is expected behavior for local running.
        return None

# --- Get encoded files ---
img_base64 = get_image_as_base64(IMAGE_PATH)

# --- CSS Styling (Adapted from Movie App) ---
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
        position: absolute;
        top: 0; left: 0; right: 0; bottom: 0;
        background: rgba(10, 5, 30, 0.7); 
        backdrop-filter: blur(4px);
        -webkit-backdrop-filter: blur(4px);
        z-index: -1; 
    }}
    """
else:
    background_css = """
    [data-testid="stAppViewContainer"] {{
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        color: #FAFAFA;
    }}
    """

st.markdown(f"""
    <style>
    /* --- Apply background style --- */
    {background_css}
    
    /* --- Adjust main content area padding/position --- */
    section[data-testid="stVerticalBlock"] {{
        padding-top: 1rem; 
        z-index: 1;
    }}

    /* --- Main Page Titles (Top-Center) --- */
    h1 {{
        color: #FFFFFF;
        /* NEW: Added strong dark shadow for readability on light backgrounds */
        text-shadow: 0 0 15px rgba(46, 204, 113, 0.8), 0 2px 5px rgba(0, 0, 0, 0.7);
        font-size: 3rem; 
        font-weight: 700;
        text-align: center;
        transition: color 0.3s ease;
        margin: 0;
    }}
    
    /* --- Main Page Subtitle --- */
    h3 {{ 
        font-weight: 600; 
        color: #E0E0E0; 
        /* NEW: Added strong dark shadow */
        text-shadow: 0 0 10px rgba(46, 204, 113, 0.5), 0 2px 4px rgba(0, 0, 0, 0.5);
        transition: color 0.3s ease;
        text-align: center; 
        font-size: 1.5rem;
        margin-top: 0.5rem;
    }}
    
    /* --- Main Page Subheaders (e.g., "Prediction Result") --- */
    /* FONT FIX: Added stronger dark shadow */
    .stSubheader {{ 
        font-weight: 600; 
        color: #E0E0E0; 
        text-shadow: 0 0 10px rgba(46, 204, 113, 0.5), 0 2px 5px rgba(0, 0, 0, 0.9);
        transition: color 0.3s ease;
    }}

    /* --- Sidebar --- */
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

    /* --- Webkit (Chrome/Safari) Scrollbar Styling --- */
    [data-testid="stSidebar"]::-webkit-scrollbar {{ width: 8px; }}
    [data-testid="stSidebar"]::-webkit-scrollbar-track {{
        background: rgba(0, 0, 0, 0.2); 
        border-radius: 10px;
    }}
    [data-testid="stSidebar"]::-webkit-scrollbar-thumb {{
        background-color: #2ECC71; 
        border-radius: 10px;
        border: 2px solid rgba(0, 0, 0, 0.2);
    }}
    [data-testid="stSidebar"]::-webkit-scrollbar-thumb:hover {{
        background-color: #27AE60; 
    }}

    /* --- Sidebar Titles --- */
    [data-testid="stSidebar"] h3 {{ 
        color: #FFFFFF; 
        text-shadow: 0 0 5px rgba(46, 204, 113, 0.5); 
        text-align: left; 
        font-size: 1.25rem; 
        font-weight: 600; 
        margin-top: 0;
        margin-bottom: 1rem;
    }}

    /* --- Frosted glass for result areas --- */
    /* FONT FIX: Made background darker and blurrier */
    .results-container, .st-expander {{
        background: rgba(0, 0, 0, 0.85); /* DARKER */
        backdrop-filter: blur(12px); /* BLURRIER */
        -webkit-backdrop-filter: blur(12px); 
        border: 1px solid rgba(255, 255, 255, 0.1); 
        border-radius: 15px;
        padding: 1.5rem 2rem;
        margin-bottom: 1.5rem;
    }}
    
    /* --- FONT FIX: Add dark shadow to text inside frosted glass --- */
    .results-container, .results-container p, .results-container li,
    .st-expander div[data-baseweb="expander"], 
    .st-expander div[data-baseweb="expander"] p, 
    .st-expander div[data-baseweb="expander"] li {{
        color: #FAFAFA !important; /* Ensure text is light */
        /* Stronger dark shadow */
        text-shadow: 0 1px 3px rgba(0,0,0,0.9), 0 1px 2px rgba(0,0,0,1); 
    }}
    /* Apply to streamlit messages inside containers */
    .results-container .stSuccess,
    .results-container .stInfo,
    .results-container .stWarning {{
        color: #FAFAFA !important;
        text-shadow: 0 1px 3px rgba(0,0,0,0.9);
    }}

    
    /* --- Sidebar Form --- */
    [data-testid="stSidebar"] .stForm {{
        background: rgba(0, 0, 0, 0.2); 
        border: 1px solid rgba(255, 255, 255, 0.1); 
        border-radius: 15px;
        padding: 1.5rem;
    }}

    /* --- Input Widgets (Number, Selectbox, Slider, Checkbox) IN SIDEBAR --- */
    [data-testid="stSidebar"] .stNumberInput label, 
    [data-testid="stSidebar"] .stSelectbox label, 
    [data-testid="stSidebar"] .stSlider label, 
    [data-testid="stSidebar"] .stCheckbox label,
    [data-testid="stSidebar"] .stRadio label {{ 
        color: #E0E0E0 !important; 
        font-weight: 500; 
        text-shadow: none; 
        transition: color 0.3s ease;
    }}
    
    /* Text/Number Inputs */
    [data-testid="stSidebar"] .stNumberInput > div > div > input, 
    [data-testid="stSidebar"] .stSelectbox > div > div {{
        background-color: rgba(255, 255, 255, 0.15); 
        color: #FAFAFA; 
        border-radius: 10px; 
        border: 1px solid rgba(255, 255, 255, 0.3); 
        transition: all 0.3s ease;
    }}
    [data-testid="stSidebar"] .stNumberInput > div > div > input:hover, 
    [data-testid="stSidebar"] .stSelectbox > div > div:hover {{
        background-color: rgba(255, 255, 255, 0.25); 
        border-color: rgba(46, 204, 113, 0.5);
    }}
    [data-testid="stSidebar"] .stNumberInput > div > div > input:focus, 
    [data-testid="stSidebar"] .stSelectbox > div > div:focus-within {{
        border-color: #2ECC71;
        box-shadow: 0 0 0 2px rgba(46, 204, 113, 0.5);
        outline: none;
    }}
    [data-testid="stSidebar"] .stSelectbox > div > div > div {{ color: #FAFAFA !important; }}
    
    /* Slider */
    [data-testid="stSidebar"] [data-testid="stSlider"] {{
        padding-top: 10px;
    }}
    [data-testid="stSidebar"] [data-testid="stSlider"] div[data-baseweb="slider"] > div {{
        background: rgba(255, 255, 255, 0.2); /* Track color */
    }}
    [data-testid="stSidebar"] [data-testid="stSlider"] div[data-baseweb="slider"] > div:nth-child(2) {{
        background: linear-gradient(90deg, #27AE60 0%, #2ECC71 100%); /* Filled part color */
    }}
    [data-testid="stSidebar"] [data-testid="stSlider"] div[data-baseweb="slider"] > div:nth-child(3) {{
        background: #FFFFFF; /* Thumb color */
        border: 2px solid #2ECC71;
        box-shadow: 0 0 5px rgba(46, 204, 113, 0.8);
    }}

    /* Checkbox */
    [data-testid="stSidebar"] .stCheckbox {{
        border: 1px solid rgba(255, 255, 255, 0.3);
        background-color: rgba(255, 255, 255, 0.1);
        padding: 10px 15px;
        border-radius: 10px;
        transition: all 0.3s ease;
    }}
    [data-testid="stSidebar"] .stCheckbox:hover {{
        background-color: rgba(255, 255, 255, 0.2);
    }}
    [data-testid="stSidebar"] .stCheckbox label {{
        padding-top: 3px;
    }}
    
    /* --- Main Buttons (Form Submit, Next, Back) --- */
    [data-testid="stSidebar"] .stButton > button {{ 
        width: 100%; 
        margin: 0.5rem auto;
        padding: 12px 30px; 
        font-size: 1.1rem; 
        font-weight: 700; 
        color: #FFFFFF; 
        background: linear-gradient(90deg, #27AE60 0%, #2ECC71 100%); 
        border: none; 
        border-radius: 12px; 
        box-shadow: 0 4px 15px rgba(46, 204, 113, 0.4); 
        transition: all 0.3s ease; 
    }}
    [data-testid="stSidebar"] .stButton > button:hover {{ 
        transform: translateY(-5px) scale(1.02); 
        box-shadow: 0 8px 25px rgba(46, 204, 113, 0.7); 
        background: linear-gradient(90deg, #2ECC71 0%, #27AE60 100%);
    }}

    /* --- Result Boxes (Success & Error) --- */
    .success-box, .error-box {{
        border-radius: 10px;
        text-align: center;
        padding: 1.5rem 1rem;
        margin-bottom: 1.5rem;
        transition: all 0.3s ease;
        backdrop-filter: blur(5px);
        -webkit-backdrop-filter: blur(5px);
    }}
    /* Success (Green) */
    .success-box {{
        background-color: rgba(46, 204, 113, 0.2);
        border: 1px solid #2ECC71;
    }}
    .success-box:hover {{
        background-color: rgba(46, 204, 113, 0.3);
        box-shadow: 0 5px 20px rgba(46, 204, 113, 0.5);
    }}
    .success-box span {{
        font-size: 2.25rem;
        font-weight: 700;
        color: #FFFFFF;
        text-shadow: 0 0 10px rgba(46, 204, 113, 0.8);
    }}
    
    /* Error/Cancelled (Red) */
    .error-box {{
        background-color: rgba(231, 76, 60, 0.2);
        border: 1px solid #E74C3C;
    }}
    .error-box:hover {{
        background-color: rgba(231, 76, 60, 0.3);
        box-shadow: 0 5px 20px rgba(231, 76, 60, 0.5);
    }}
    .error-box span {{
        font-size: 2.25rem;
        font-weight: 700;
        color: #FFFFFF;
        text-shadow: 0 0 10px rgba(231, 76, 60, 0.8);
    }}

    /* --- Lottie Animation Styling --- */
    div[data-testid="stLottie"] {{
        margin-top: 1rem; 
        margin-right: 1rem; 
        position: relative;
        z-index: 10; 
    }}

    /* --- Progress Bar --- */
    [data-testid="stProgressBar"] > div {{
        background: linear-gradient(90deg, #E74C3C 0%, #F39C12 100%);
    }}
    [data-testid="stProgressBar"] {{
        background-color: rgba(255, 255, 255, 0.1);
    }}

    /* --- Expander (for explanations) --- */
    .st-expander {{
        background: none;
        border: none;
        padding: 0;
    }}
    /* FONT FIX: Make expander header darker */
    .st-expander header {{
        background: rgba(0, 0, 0, 0.7); 
        color: #E0E0E0;
        border-radius: 10px;
        transition: all 0.3s ease;
        text-shadow: 0 2px 4px rgba(0,0,0,0.9); 
    }}
    .st-expander header:hover {{
        background: rgba(0, 0, 0, 0.8);
        color: #2ECC71;
    }}
    /* FONT FIX: Make expander body darker */
    .st-expander div[data-baseweb="expander"] {{
        background: rgba(0, 0, 0, 0.85); 
        border-radius: 0 0 10px 10px;
        padding: 1rem;
    }}

    /* --- Footer --- */
    footer {{
        visibility: hidden; 
    }}
    .footer-text {{
        text-align: center; 
        color: #999;
        font-size: 0.9rem;
        padding-top: 2rem;
        transition: color 0.3s ease;
    }}
    .footer-text:hover {{
        color: #FAFAFA;
    }}
    
    </style>
""", unsafe_allow_html=True)


# -----------------------
# RESTORED: Your original complex auto-play sound helper
# -----------------------
def play_sound_autoplay(file_path: Path, sound_type: str):
    """Professional auto-play sound using JavaScript with user interaction tracking"""
    if file_path.exists():
        # Convert sound to base64 for embedding
        sound_bytes = open(file_path, "rb").read()
        b64_sound = base64.b64encode(sound_bytes).decode()
        
        # JavaScript for auto-play with user interaction fallback
        audio_html = f"""
        <audio id="audio_{sound_type}" controls style="display: none;">
            <source src="data:audio/mp3;base64,{b64_sound}" type="audio/mp3">
        </audio>
        <script>
            // Function to play sound
            function play{sound_type.capitalize()}Sound() {{
                var audio = document.getElementById('audio_{sound_type}');
                var playPromise = audio.play();
                
                if (playPromise !== undefined) {{
                    playPromise.then(function() {{
                        console.log('Sound auto-played successfully');
                    }}).catch(function(error) {{
                        console.log('Auto-play blocked, showing manual controls');
                        audio.style.display = 'block';
                        // Create a play button if auto-play fails
                        createPlayButton('audio_{sound_type}');
                    }});
                }}
            }}
            
            // Create manual play button if auto-play fails
            function createPlayButton(audioId) {{
                var audioDiv = document.getElementById(audioId).parentNode;
                // Check if button already exists
                if (audioDiv.querySelector('button')) return;
                
                var playBtn = document.createElement('button');
                playBtn.innerHTML = '▶️ Play Sound';
                playBtn.style.cssText = 'background: #FF4B4B; color: white; border: none; padding: 10px 20px; border-radius: 5px; cursor: pointer; margin: 10px 0;';
                playBtn.onclick = function() {{ document.getElementById(audioId).play(); }};
                audioDiv.appendChild(playBtn);
            }}
            
            // Try to play immediately (will work if user already interacted)
            play{sound_type.capitalize()}Sound();
            
            // Also try after a short delay (for async loading)
            setTimeout(play{sound_type.capitalize()}Sound, 500);
        </script>
        """
        
        # Display the audio HTML
        st.components.v1.html(audio_html, height=0)
        
        # Show a subtle status message
        if sound_type == "fail":
            st.markdown("🔊 *Playing cancellation sound...*")
        else:
            st.markdown("🔊 *Playing success sound...*")
            
    else:
        st.warning(f"🔇 Sound file not found: {file_path.name}")

# -----------------------
# Cache helpers
# -----------------------
@st.cache_resource(show_spinner=False)
def load_model(path):
    return joblib.load(path)

@st.cache_data(show_spinner=False)
def fetch_lottie(url):
    try:
        r = requests.get(url, timeout=6)
        return r.json() if r.status_code == 200 else None
    except:
        return None

# -----------------------
# Load model
# -----------------------
if not MODEL_PATH.exists():
    # If file not found, we don't stop, but we set model to None so we can continue the app
    model = None
else:
    model = load_model(MODEL_PATH)
    expected_features = getattr(model, "feature_names_in_", None)

# Check if sound files exist
SOUNDS_AVAILABLE = SOUNDS_PATH.exists() and SUCCESS_SOUND.exists() and FAIL_SOUND.exists()

# -----------------------
# Simple auto-play sound
# -----------------------
def play_sound_simple(file_path: Path):
    """Simple auto-play that works reliably with user interaction"""
    if file_path.exists():
        st.audio(str(file_path), format="audio/mp3", autoplay=True)
    else:
        st.warning(f"🔇 Sound file not found: {file_path.name}")

# -----------------------
# Session State
# -----------------------
if "last_result" not in st.session_state:
    st.session_state.last_result = None
if "user_interacted" not in st.session_state:
    st.session_state.user_interacted = False

# -----------------------
# Sidebar Layout
# -----------------------
with st.sidebar:
    st.header("⚙️ Controls")
    enable_sounds = st.checkbox("🔊 Enable Sounds", value=SOUNDS_AVAILABLE)
    sound_method = st.radio("Sound Playback:", ["Auto-Play", "Manual Play"], index=0)
    show_charts = st.checkbox("📊 Show detailed charts", value=False)
    
    with st.expander("🔊 Sound Files Status"):
        if SOUNDS_PATH.exists():
            st.success("✅ Sounds folder found")
            st.write(f"**Success sound:** {'✅' if SUCCESS_SOUND.exists() else '❌'}") 
            st.write(f"**Fail sound:** {'✅' if FAIL_SOUND.exists() else '❌'}")
            if not SOUNDS_AVAILABLE:
                st.error("❌ Some sound files are missing!")
        else:
            st.error(f"❌ Sounds folder not found at {SOUNDS_PATH}")
    
    st.markdown("---")
    
    # --- Form is now in the sidebar ---
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
# Main Page Layout (Header + Results)
# -----------------------

# --- Header ---
# Using columns for the top-center title and top-right Lottie
cols = st.columns([0.15, 0.7, 0.15]) 
with cols[0]:
    if LOGO_PATH.exists():
        st.image(str(LOGO_PATH), width=100, use_container_width=False, output_format="PNG")
    else:
        st.warning(f"Logo not found at {LOGO_PATH}")

with cols[1]:
    st.markdown("<h1 class='main-title' style='text-align: center;'>🚖 Ride Cancellation Predictor</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center;'>Will your ride be completed or cancelled?</h3>", unsafe_allow_html=True)


with cols[2]:
    # Adjusted Lottie animation for top-right placement
    anim = fetch_lottie("https://assets7.lottiefiles.com/packages/lf20_iwmd6pyr.json")
    if anim:
        try:
            st_lottie(anim, height=100, key="header_lottie")
        except:
            pass

st.markdown("---") # Separator below the header

# --- Prediction Logic ---
if predict_button:
    st.session_state.user_interacted = True
    
    st.subheader("📊 Prediction Result")
    
    if model is None:
        st.error("Cannot make prediction: Model file (`ride_cancel_model.pkl`) not loaded. Please ensure it is in the correct directory.")
    else:
        # 1. Create DataFrame
        df = pd.DataFrame({
            "Customer Rating": [cr],
            "Driver Ratings": [dr],
            "Booking Value": [min(bv, 1500)], # Cap
            "Ride Distance": [rd],
            "hour": [hr],
            "Avg VTAT": [8.3],  # Default average
            "Avg CTAT": [28.8], # Default average
            "Vehicle Type": [vt],
            "Payment Method": [pm]
        })
        
        # 2. One-hot encode and align
        df_enc = pd.get_dummies(df, columns=['Vehicle Type','Payment Method'], drop_first=False)
        if expected_features is not None:
            for f in expected_features:
                if f not in df_enc.columns:
                    df_enc[f] = 0
            df_enc = df_enc[expected_features]
        
        # 3. --- Rule-based override ---
        override = None
        override_reason = None

        if rd > 100 and bv < 0.2 * rd:
            override = 1 
            override_reason = "Long distance with too low booking value"
        elif cr < 2.0 or dr < 2.0:
            override = 1
            override_reason = "Very low customer or driver rating (<2.0)"

        # 4. --- Make prediction ---
        if override is not None:
            pred = int(override)
            proba = np.array([0.0, 1.0]) if pred == 1 else np.array([1.0, 0.0])
            st.warning(f"⚠️ **Rule-based override triggered:** {override_reason}")
        else:
            proba = model.predict_proba(df_enc)[0]
            pred = model.predict(df_enc)[0]

        # Store last result
        st.session_state.last_result = {"Prediction": int(pred), "Prob_Completed": float(proba[0]), "Prob_Cancelled": float(proba[1])}

        # 5. --- Display Results with new style ---
        with st.container():
            st.markdown(f'<div class="results-container">', unsafe_allow_html=True)
            if pred==1:
                st.markdown(f'<div class="error-box"><span>CANCELLED</span><br><strong style="font-size: 1.5rem;">{proba[1]:.1%} Likelihood</strong></div>', unsafe_allow_html=True)
                
                if enable_sounds and SOUNDS_AVAILABLE:
                    if sound_method == "Auto-Play" and st.session_state.user_interacted:
                        play_sound_simple(FAIL_SOUND)
                    else:
                        st.audio(str(FAIL_SOUND), format="audio/mp3")
                        st.info("🔊 Click play button to hear cancellation sound")

                try:
                    fail_anim = fetch_lottie("https.assets2.lottiefiles.com/packages/lf20_jtbfg2nb.json")
                    if fail_anim: st_lottie(fail_anim, height=150, key="fail_anim")
                except: pass
            else:
                st.markdown(f'<div class="success-box"><span>COMPLETED</span><br><strong style="font-size: 1.5rem;">{proba[0]:.1%} Likelihood</strong></div>', unsafe_allow_html=True)
                
                if enable_sounds and SOUNDS_AVAILABLE:
                    if sound_method == "Auto-Play" and st.session_state.user_interacted:
                        play_sound_simple(SUCCESS_SOUND)
                    else:
                        st.audio(str(SUCCESS_SOUND), format="audio/mp3")
                        st.info("🔊 Click play button to hear success sound")

                try:
                    success_anim = fetch_lottie("https://assets2.lottiefiles.com/packages/lf20_jbrw3hcz.json")
                    if success_anim: st_lottie(success_anim, height=150, key="success_anim")
                except: pass
            
            st.markdown(f"**Cancellation Probability:**")
            st.progress(int(proba[1]*100))
            st.markdown("</div>", unsafe_allow_html=True)


        # 6. --- RESTORED: Explanations ---
        st.subheader("🧠 Why did the model predict this?")

        # Local explanation
        notes = []
        if override_reason:
            notes.append(f"⚠️ **Rule Override:** {override_reason}")
        if bv > 1000: notes.append("💰 **Very High Booking Value** → Higher cancellation risk")
        elif bv < 100: notes.append("💰 **Low Booking Value** → Higher cancellation risk")
        if rd > 30: notes.append("📍 **Long Distance** → Higher cancellation risk")
        if hr in [0, 1, 2, 3, 4, 5]: notes.append("⏰ **Late Night (12AM-6AM)** → Higher risk")
        if vt in ["Bike", "eBike"]: notes.append("🚲 **Two-Wheeler** → Higher risk")
        if pm == "Cash": notes.append("💵 **Cash Payment** → Higher risk")
        if cr < 3.0: notes.append("⭐ **Low Customer Rating** → Higher risk")
        if dr < 3.0: notes.append("🚗 **Low Driver Rating** → Higher risk")
        
        if not notes:
            notes.append("✅ **Standard Ride:** No obvious high-risk factors detected.")
            
        with st.container():
            st.markdown(f'<div class="results-container" style="padding: 1rem 2rem;">', unsafe_allow_html=True)
            for note in notes:
                st.write(f"• {note}")

            confidence = max(proba[0], proba[1])
            if confidence > 0.8: st.success(f"🎯 **High Confidence**: {confidence:.1%}")
            elif confidence > 0.6: st.info(f"🎯 **Medium Confidence**: {confidence:.1%}")
            else: st.warning(f"🎯 **Low Confidence**: {confidence:.1%} - Close call!")
            st.markdown("</div>", unsafe_allow_html=True)


        # 7. Global charts (if toggled)
        if show_charts:
            st.subheader("📊 Global Feature Importance")
            with st.expander("Click to see the most important features for the model"):
                if hasattr(model,"feature_importances_") and expected_features is not None:
                    fi = pd.DataFrame({"feature":expected_features,"importance":model.feature_importances_})
                    top = fi.sort_values("importance",ascending=False).head(10)
                    
                    # Create styled chart
                    plt.style.use('dark_background')
                    fig,ax=plt.subplots(figsize=(10,5))
                    sns.barplot(data=top,x="importance",y="feature",ax=ax, palette="viridis")
                    ax.set_title("Top Important Features (Global)", color="white")
                    ax.tick_params(colors='white')
                    fig.patch.set_alpha(0.0)
                    ax.patch.set_alpha(0.0)
                    st.pyplot(fig)
                else:
                    st.info("Model does not provide feature importances.")
        
        # 8. Download buttons
        if st.session_state.last_result:
            st.subheader("📋 Download Report")
            res = st.session_state.last_result
            csv = pd.DataFrame([res]).to_csv(index=False).encode("utf-8")
            
            pdf_file = BASE_DIR / "prediction_report.pdf"
            try:
                pdf = FPDF()
                pdf.add_page()
                pdf.set_font("Arial", size=14)
                pdf.cell(200,10,"Ride Cancellation Prediction Report",ln=True,align="C")
                for k,v in res.items(): 
                    pdf.cell(200,10,f"{k}: {v}",ln=True)
                pdf.output(str(pdf_file))
                with open(pdf_file,"rb") as f:
                    st.download_button("📥 Download PDF Report", f.read(), "prediction_report.pdf", key="pdf_dl")
            except Exception as e:
                st.error(f"Failed to generate PDF: {e}")
                
            st.download_button("📥 Download CSV Report", csv, "prediction_report.csv","text/csv", key="csv_dl")

# Add professional footer
st.markdown("---")
st.markdown("<div class='footer-text'>🚖 Ride Cancellation Predictor • Powered by Machine Learning</div>", 
            unsafe_allow_html=True)
