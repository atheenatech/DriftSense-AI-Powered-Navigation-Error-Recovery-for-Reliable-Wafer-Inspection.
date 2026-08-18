import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image


# =========================================================
# PROJECT ROOT
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# =========================================================
# PROJECT PATHS
# =========================================================

MODEL_PATH = (
    PROJECT_ROOT
    / "Results"
    / "Training"
    / "best_model.pth"
)

TRAINING_DIR = (
    PROJECT_ROOT
    / "AI"
    / "Training"
)

SEVERITY_CSV = (
    PROJECT_ROOT
    / "Dataset"
    / "severity_rules.csv"
)


# =========================================================
# IMPORT RESNET18 MODEL
# =========================================================

sys.path.insert(
    0,
    str(TRAINING_DIR)
)

try:

    from model import create_model

except Exception as e:

    st.error(
        "❌ Could not import model.py"
    )

    st.write(
        "Expected model location:"
    )

    st.code(
        str(TRAINING_DIR / "model.py")
    )

    st.exception(e)

    st.stop()


# =========================================================
# IMPORT GRAD-CAM
# =========================================================

GRADCAM_DIR = (
    PROJECT_ROOT
    / "Dashboard"
)

sys.path.insert(
    0,
    str(GRADCAM_DIR)
)

try:

    from components.gradcam import generate_gradcam

except Exception as e:

    st.error(
        "❌ Could not import Grad-CAM."
    )

    st.exception(e)

    st.stop()


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Semiconductor Wafer AI Inspection",
    page_icon="🔬",
    layout="wide"
)


# =========================================================
# CLASS NAMES
# =========================================================

CLASSES = [
    "Center",
    "Donut",
    "Edge-Loc",
    "Edge-Ring",
    "Loc",
    "Near-full",
    "Random",
    "Scratch"
]


# =========================================================
# DECISION THRESHOLDS
# =========================================================

LOW_THRESHOLD = 30.0
HIGH_THRESHOLD = 70.0


# =========================================================
# PREMIUM FUTURISTIC UI THEME
# =========================================================
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Space+Grotesk:wght@400;500;600;700&display=swap');

    :root {
        --bg: #050b10;
        --bg2: #08131a;
        --panel: rgba(11, 24, 32, .92);
        --panel2: rgba(15, 31, 40, .96);
        --line: rgba(110, 231, 205, .16);
        --line2: rgba(110, 231, 205, .34);
        --ink: #edfdf9;
        --muted: #88a4a5;
        --mint: #69e7ca;
        --cyan: #61cfff;
        --green: #63e6a8;
        --amber: #ffc66d;
        --red: #ff7782;
        --shadow: 0 18px 55px rgba(0,0,0,.28);
    }

    html, body, .stApp,
    [data-testid="stAppViewContainer"],
    [data-testid="stAppViewContainer"] > .main,
    section[data-testid="stMain"],
    .main {
        background:
            radial-gradient(circle at 80% -10%, rgba(39,101,108,.30), transparent 34%),
            radial-gradient(circle at 0% 35%, rgba(24,73,82,.13), transparent 28%),
            linear-gradient(180deg, var(--bg2) 0%, var(--bg) 72%, #03070a 100%) !important;
        color: var(--ink) !important;
        font-family: 'Space Grotesk', sans-serif;
    }

    .stApp:before {
        content:'';
        position:fixed;
        inset:0;
        pointer-events:none;
        z-index:0;
        opacity:.10;
        background-image:
            linear-gradient(rgba(105,231,202,.09) 1px, transparent 1px),
            linear-gradient(90deg, rgba(105,231,202,.09) 1px, transparent 1px);
        background-size: 48px 48px;
        mask-image: linear-gradient(to bottom, black, transparent 75%);
    }

    .block-container,
    [data-testid="stMainBlockContainer"] {
        max-width: 1480px;
        padding: 1.1rem 3.2rem 4.5rem;
        position:relative;
        z-index:1;
    }

    header[data-testid="stHeader"],
    [data-testid="stToolbar"] {
        background:transparent !important;
    }

    h1,h2,h3,h4 {
        font-family:'Space Grotesk',sans-serif !important;
        color:var(--ink) !important;
        letter-spacing:-.035em;
    }
    h2 { font-size:1.55rem !important; margin-top:1.35rem !important; }
    h3 { font-size:1.02rem !important; }
    p,li,label,[data-testid="stMarkdownContainer"] { color:var(--ink); }
    [data-testid="stCaptionContainer"], .stCaption { color:var(--muted) !important; }
    code,pre { font-family:'DM Mono',monospace; }

    /* Header */
    .topline {
        display:flex;
        justify-content:space-between;
        align-items:center;
        padding:.65rem 0 1rem;
        margin-bottom:2.4rem;
        border-bottom:1px solid var(--line);
    }
    .brand {
        font-family:'DM Mono',monospace;
        font-size:.72rem;
        font-weight:500;
        letter-spacing:.18em;
        color:var(--mint);
    }
    .system {
        font-family:'DM Mono',monospace;
        font-size:.68rem;
        letter-spacing:.1em;
        color:#a8c4c3;
    }
    .system-dot {
        display:inline-block;
        width:7px;height:7px;
        margin-right:8px;
        border-radius:50%;
        background:var(--green);
        box-shadow:0 0 16px rgba(99,230,168,.85);
        animation:pulse 2.1s ease-in-out infinite;
    }
    @keyframes pulse {
        0%,100% { opacity:.5; transform:scale(.88); }
        50% { opacity:1; transform:scale(1.15); }
    }

    .hero { max-width:920px; padding:.2rem 0 1.7rem; }
    .eyebrow {
        color:var(--cyan);
        font-family:'DM Mono',monospace;
        font-size:.66rem;
        letter-spacing:.16em;
        text-transform:uppercase;
        margin-bottom:.85rem;
    }
    .hero-title {
        font-size:clamp(2.5rem,5.7vw,5rem) !important;
        line-height:.98 !important;
        font-weight:600 !important;
        margin:0 !important;
        background:linear-gradient(105deg,#f3fffc 5%,#9ef2df 52%,#66d5ff 100%);
        -webkit-background-clip:text;
        background-clip:text;
        color:transparent !important;
    }
    .hero-lead {
        color:var(--muted) !important;
        font-size:1rem;
        line-height:1.65;
        max-width:720px;
        margin-top:1.15rem;
    }

    .section-kicker {
        color:var(--cyan);
        font-family:'DM Mono',monospace;
        font-size:.62rem;
        font-weight:500;
        letter-spacing:.17em;
        text-transform:uppercase;
        margin:2.1rem 0 .5rem;
    }

    .pipeline {
        display:flex;
        flex-wrap:wrap;
        align-items:center;
        gap:.45rem;
        padding:.8rem 1rem;
        margin:1rem 0 1.7rem;
        border:1px solid var(--line);
        border-radius:12px;
        background:rgba(7,20,27,.72);
        box-shadow:0 10px 35px rgba(0,0,0,.13);
    }
    .pipe-step {
        color:#ccece7;
        font-family:'DM Mono',monospace;
        font-size:.62rem;
        letter-spacing:.05em;
        padding:.42rem .62rem;
        border:1px solid var(--line);
        border-radius:7px;
        background:rgba(105,231,202,.045);
    }
    .pipe-arrow { color:var(--mint); font-size:.78rem; }

    /* Sidebar */
    [data-testid="stSidebar"],
    [data-testid="stSidebarContent"] {
        background:
            linear-gradient(180deg,#07151c 0%,#061017 100%) !important;
        border-right:1px solid var(--line);
    }
    [data-testid="stSidebar"] * { color:var(--ink) !important; }
    [data-testid="stSidebar"] .stMarkdown { line-height:1.65; }
    [data-testid="stSidebar"] hr { border-color:var(--line) !important; }

    /* Upload area */
    [data-testid="stFileUploader"] {
        background:linear-gradient(145deg,rgba(14,31,39,.95),rgba(7,18,24,.95));
        border:1px solid var(--line2);
        border-radius:16px;
        padding:1rem;
        box-shadow:var(--shadow);
    }
    [data-testid="stFileUploaderDropzone"] {
        border:1px dashed rgba(105,231,202,.32) !important;
        border-radius:12px !important;
        background:rgba(105,231,202,.025) !important;
    }

    /* Metrics */
    [data-testid="stMetric"] {
        min-height:102px;
        background:linear-gradient(145deg,rgba(16,37,46,.94),rgba(8,21,28,.96));
        border:1px solid var(--line);
        border-radius:15px;
        padding:1rem 1.05rem;
        box-shadow:0 12px 34px rgba(0,0,0,.17);
    }
    [data-testid="stMetricLabel"] {
        color:var(--muted) !important;
        font-family:'DM Mono',monospace;
        font-size:.61rem;
        letter-spacing:.09em;
        text-transform:uppercase;
    }
    [data-testid="stMetricValue"] {
        color:var(--mint) !important;
        font-family:'DM Mono',monospace;
        font-size:1.28rem;
        margin-top:.25rem;
    }

    /* Images */
    [data-testid="stImage"] {
        background:linear-gradient(145deg,rgba(14,29,36,.9),rgba(6,14,19,.9));
        border:1px solid var(--line);
        border-radius:16px;
        padding:.55rem;
        box-shadow:var(--shadow);
    }
    [data-testid="stImage"] img {
        border-radius:11px;
    }

    /* Progress */
    .stProgress > div > div {
        background:rgba(105,231,202,.09) !important;
        border-radius:99px;
    }
    .stProgress > div > div > div {
        background:linear-gradient(90deg,var(--cyan),var(--mint)) !important;
        border-radius:99px;
    }

    /* Alerts */
    [data-testid="stAlert"] {
        border-radius:12px !important;
        border:1px solid var(--line) !important;
        background:rgba(13,31,39,.92) !important;
        color:var(--ink) !important;
        box-shadow:0 8px 25px rgba(0,0,0,.12);
    }

    /* Result cards */
    .result-box {
        background:
            radial-gradient(circle at 100% 0%,rgba(105,231,202,.11),transparent 42%),
            linear-gradient(145deg,rgba(17,48,54,.96),rgba(8,24,31,.98));
        border:1px solid var(--line2);
        border-radius:17px;
        padding:1.45rem 1.55rem;
        min-height:180px;
        box-shadow:var(--shadow);
    }
    .result-box h2 {
        color:var(--mint) !important;
        font-size:2rem !important;
        margin:.1rem 0 1rem !important;
    }
    .result-box p { color:#d8efeb !important; margin:.55rem 0; }

    .final-box {
        background:
            radial-gradient(circle at 90% 10%,rgba(99,230,168,.12),transparent 35%),
            linear-gradient(145deg,#10363a,#091f27 65%,#0c2935);
        border:1px solid var(--line2);
        border-radius:20px;
        padding:1.8rem 2rem;
        margin-top:1rem;
        box-shadow:0 0 50px rgba(65,220,190,.09),var(--shadow);
    }
    .final-box h2 {
        color:var(--mint) !important;
        font-size:1.8rem !important;
        margin-top:0 !important;
        margin-bottom:1.1rem !important;
    }
    .final-box p { color:#e1f5f1 !important; line-height:1.65; }

    hr { border-color:var(--line) !important; margin:1.8rem 0 !important; }

    /* Buttons */
    .stButton > button {
        background:rgba(105,231,202,.035);
        border:1px solid var(--line2);
        color:var(--mint);
        border-radius:9px;
        transition:.2s ease;
    }
    .stButton > button:hover {
        background:rgba(105,231,202,.10);
        border-color:var(--mint);
        color:#fff;
        transform:translateY(-1px);
    }

    /* Cleaner Streamlit chrome */
    [data-testid="stDecoration"] { display:none; }
    div[data-testid="stStatusWidget"] { visibility:hidden; }

    @media(max-width:900px) {
        .block-container { padding:1rem 1rem 3rem; }
        .topline { flex-direction:column; align-items:flex-start; gap:.6rem; }
        .hero-title { font-size:2.8rem !important; }
        .pipeline { gap:.3rem; }
    }
    </style>
    """,
    unsafe_allow_html=True
)


st.markdown(
    """<style>
.section-kicker {
    position:relative;
    display:inline-flex;
    align-items:center;
    gap:8px;
    padding:.32rem .65rem;
    border:1px solid rgba(97,207,255,.18);
    border-radius:999px;
    background:rgba(97,207,255,.035);
    box-shadow:0 5px 18px rgba(0,0,0,.10);
}
.section-kicker:before {
    content:'';
    width:5px;
    height:5px;
    border-radius:50%;
    background:#69e7ca;
    box-shadow:0 0 10px rgba(105,231,202,.8);
}
h2 {
    padding-bottom:.45rem !important;
    border-bottom:1px solid rgba(110,231,205,.08);
}
[data-testid="stFileUploader"] {
    position:relative;
    overflow:hidden;
}
[data-testid="stFileUploader"]:before {
    content:'WAFER MAP INPUT';
    display:block;
    color:#69e7ca;
    font-family:'DM Mono',monospace;
    font-size:.57rem;
    letter-spacing:.16em;
    margin-bottom:.65rem;
}
[data-testid="stMetric"] {
    position:relative;
    overflow:hidden;
}
[data-testid="stMetric"]:before {
    content:'';
    position:absolute;
    left:0;
    right:0;
    top:0;
    height:2px;
    background:linear-gradient(90deg,transparent,#69e7ca,transparent);
    opacity:.65;
}
[data-testid="stImage"] {
    position:relative;
}
.final-box {
    position:relative;
    overflow:hidden;
}
.final-box:after {
    content:'DECISION SUPPORT';
    position:absolute;
    top:1rem;
    right:1.25rem;
    font-family:'DM Mono',monospace;
    font-size:.52rem;
    letter-spacing:.16em;
    color:rgba(105,231,202,.48);
}
[data-testid="stSidebar"] h2 {
    border-bottom:0 !important;
    padding-bottom:0 !important;
}
[data-testid="stCaptionContainer"] {
    font-family:'DM Mono',monospace !important;
    font-size:.58rem !important;
    letter-spacing:.04em;
}
[data-testid="stMetric"],
.result-box,
.final-box,
[data-testid="stFileUploader"] {
    transition:transform .18s ease, border-color .18s ease, box-shadow .18s ease;
}
[data-testid="stMetric"]:hover,
.result-box:hover,
.final-box:hover,
[data-testid="stFileUploader"]:hover {
    transform:translateY(-2px);
    border-color:rgba(105,231,202,.30);
    box-shadow:0 20px 55px rgba(0,0,0,.28);
}
@media(max-width:700px) {
    .final-box:after { display:none; }
    [data-testid="stMetric"] { min-height:88px; }
}
</style>""",
    unsafe_allow_html=True
)

# =========================================================
# LOAD SEVERITY CSV
# =========================================================

@st.cache_data
def load_severity_rules():

    if not SEVERITY_CSV.exists():

        raise FileNotFoundError(
            "Severity CSV not found:\n"
            f"{SEVERITY_CSV}"
        )

    df = pd.read_csv(
        SEVERITY_CSV
    )

    # Remove spaces from column names
    df.columns = (
        df.columns
        .astype(str)
        .str.strip()
    )

    # -----------------------------------------------------
    # NORMALIZE COLUMN NAMES
    # -----------------------------------------------------

    rename_map = {}

    for column in df.columns:

        normalized = (
            str(column)
            .lower()
            .strip()
            .replace(" ", "_")
            .replace("-", "_")
        )

        if normalized in [
            "defect_class",
            "class",
            "defect",
            "failure_type",
            "failuretype"
        ]:

            rename_map[
                column
            ] = "defect_class"

        elif normalized in [
            "class_risk",
            "risk",
            "risk_score",
            "severity",
            "severity_score"
        ]:

            rename_map[
                column
            ] = "class_risk"

    df = df.rename(
        columns=rename_map
    )

    # -----------------------------------------------------
    # CHECK COLUMNS
    # -----------------------------------------------------

    if "defect_class" not in df.columns:

        raise ValueError(
            "Your CSV needs a defect class column.\n\n"
            "Example:\n"
            "defect_class"
        )

    if "class_risk" not in df.columns:

        raise ValueError(
            "Your CSV needs a class risk column.\n\n"
            "Example:\n"
            "class_risk"
        )

    # Convert risk to numbers
    df["class_risk"] = pd.to_numeric(
        df["class_risk"],
        errors="coerce"
    )

    df["class_risk"] = (
        df["class_risk"]
        .fillna(50.0)
        .clip(0, 100)
    )

    return df


# =========================================================
# LOAD AI MODEL
# =========================================================

@st.cache_resource
def load_ai_model():

    device = torch.device(
        "cpu"
    )

    # Create ResNet18 architecture
    model = create_model()

    # Check model
    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            "Trained model not found:\n"
            f"{MODEL_PATH}"
        )

    # Load checkpoint
    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=False
    )

    # -----------------------------------------------------
    # SUPPORT DIFFERENT CHECKPOINT FORMATS
    # -----------------------------------------------------

    if isinstance(
        checkpoint,
        dict
    ):

        if "model_state_dict" in checkpoint:

            state_dict = (
                checkpoint[
                    "model_state_dict"
                ]
            )

        elif "state_dict" in checkpoint:

            state_dict = (
                checkpoint[
                    "state_dict"
                ]
            )

        else:

            state_dict = checkpoint

    else:

        state_dict = checkpoint

    # Load trained weights
    model.load_state_dict(
        state_dict
    )

    model.to(
        device
    )

    model.eval()

    return model


# =========================================================
# LOAD MODEL
# =========================================================

try:

    model = load_ai_model()

except Exception as e:

    st.error(
        "❌ Unable to load the trained AI model."
    )

    st.exception(e)

    st.stop()


# =========================================================
# LOAD CSV
# =========================================================

try:

    severity_rules = (
        load_severity_rules()
    )

except Exception as e:

    st.error(
        "❌ Unable to load severity_rules.csv"
    )

    st.write(
        "Expected location:"
    )

    st.code(
        str(SEVERITY_CSV)
    )

    st.exception(e)

    st.stop()


# =========================================================
# DEVICE
# =========================================================

device = torch.device(
    "cpu"
)


# =========================================================
# DASHBOARD HEADER
# =========================================================

st.markdown(
    """
    <div class="topline">
        <div class="brand">SEMI-INSPECT AI // FAB ANALYTICS</div>
        <div class="system"><span class="system-dot"></span>SYSTEM ONLINE · CPU INFERENCE</div>
    </div>
    <div class="hero">
        <div class="eyebrow">Defect8 · WM-811K · ResNet18 · Explainable Inspection</div>
        <h1 class="hero-title">Semiconductor wafer intelligence.</h1>
        <p class="hero-lead">A production-oriented inspection console for defect classification, spatial analysis, Grad-CAM explainability, and manufacturing decision support.</p>
    </div>
    <div class="pipeline">
        <span class="pipe-step">01 · WAFER INPUT</span><span class="pipe-arrow">→</span>
        <span class="pipe-step">02 · RESNET18</span><span class="pipe-arrow">→</span>
        <span class="pipe-step">03 · GRAD-CAM</span><span class="pipe-arrow">→</span>
        <span class="pipe-step">04 · SPATIAL RISK</span><span class="pipe-arrow">→</span>
        <span class="pipe-step">05 · FAB ACTION</span>
    </div>
    """,
    unsafe_allow_html=True
)

# =========================================================
# INSPECTION CONTROL SIDEBAR
# =========================================================

st.sidebar.markdown("## INSPECTION CONTROL")
st.sidebar.caption("Upload a wafer map to begin a live analysis run.")
st.sidebar.divider()
st.sidebar.markdown("**WORKFLOW**")
st.sidebar.markdown("""
<div style="font-family:'DM Mono',monospace;font-size:.72rem;line-height:2;color:#8eaaa9">
<span style="color:#7cefd0">●</span> INPUT / .NPY UPLOAD<br>
<span style="color:#7cefd0">●</span> DEFECT8 CLASSIFICATION<br>
<span style="color:#7cefd0">●</span> GRAD-CAM ATTENTION<br>
<span style="color:#7cefd0">●</span> LOCATION & SEVERITY<br>
<span style="color:#7cefd0">●</span> MANUFACTURING ACTION
</div>
""", unsafe_allow_html=True)
st.sidebar.divider()
st.sidebar.caption("The inference engine, preprocessing, checkpoint, and decision logic remain unchanged.")

# =========================================================
# FILE UPLOAD
# =========================================================

st.markdown('<div class="section-kicker">Inspection input</div>', unsafe_allow_html=True)
st.markdown("## Wafer intake", unsafe_allow_html=True)
uploaded_file = st.file_uploader(
    "Upload wafer map (.npy)",
    type=["npy"],
    help="Upload a NumPy wafer map.",
)


# =========================================================
# WAIT FOR FILE
# =========================================================

if uploaded_file is None:

    st.info("Upload a `.npy` wafer map to activate the inspection pipeline.")

    st.stop()


# =========================================================
# LOAD NPY
# =========================================================

try:

    uploaded_file.seek(0)

    wafer = np.load(
        uploaded_file,
        allow_pickle=False
    )

except Exception as e:

    st.error(
        "❌ The uploaded `.npy` file could not be read."
    )

    st.exception(e)

    st.stop()


# =========================================================
# CONVERT
# =========================================================

wafer = np.asarray(
    wafer
)


# =========================================================
# CHECK DIMENSION
# =========================================================

if wafer.ndim != 2:

    st.error(
        f"❌ Invalid wafer map shape: {wafer.shape}"
    )

    st.warning(
        "The AI expects a 2D wafer map."
    )

    st.stop()


# =========================================================
# CLEAN DATA
# =========================================================

wafer = np.nan_to_num(
    wafer,
    nan=0.0,
    posinf=0.0,
    neginf=0.0
)


# =========================================================
# FILE INFORMATION
# =========================================================

st.markdown("<div class=\"section-kicker\">Live sample</div>", unsafe_allow_html=True)
st.markdown("## Wafer telemetry", unsafe_allow_html=True)

file_col1, file_col2, file_col3 = (
    st.columns(3)
)

with file_col1:

    st.metric(
        "File",
        uploaded_file.name
    )

with file_col2:

    st.metric(
        "Shape",
        f"{wafer.shape[0]} × "
        f"{wafer.shape[1]}"
    )

with file_col3:

    st.metric(
        "Data Type",
        str(wafer.dtype)
    )


# =========================================================
# CREATE DISPLAY IMAGE
# =========================================================

display_image = wafer.astype(
    np.float32
)

if display_image.max() > 0:

    display_image = (
        display_image
        / display_image.max()
        * 255
    )

display_image = np.clip(
    display_image,
    0,
    255
).astype(
    np.uint8
)

original_image = Image.fromarray(
    display_image,
    mode="L"
)


# =========================================================
# PREPROCESS FUNCTION
# =========================================================

def preprocess_wafer(
    wafer_array
):

    image = wafer_array.astype(
        np.float32
    )

    # Normalize
    if image.max() > 0:

        image = (
            image
            / image.max()
        )

    tensor = torch.from_numpy(
        image
    )

    # H,W → 1,H,W
    if tensor.ndim == 2:

        tensor = tensor.unsqueeze(
            0
        )

    # 1,H,W → 1,1,H,W
    tensor = tensor.unsqueeze(
        0
    )

    # Resize to ResNet18 input
    tensor = F.interpolate(
        tensor,
        size=(224, 224),
        mode="nearest"
    )

    # Grayscale → RGB
    tensor = tensor.repeat(
        1,
        3,
        1,
        1
    )

    return tensor


# =========================================================
# PREPROCESS
# =========================================================

input_tensor = preprocess_wafer(
    wafer
)


# =========================================================
# AI PREDICTION
# =========================================================

with st.spinner(
    "🤖 AI is analyzing the wafer..."
):

    with torch.no_grad():

        output = model(
            input_tensor.to(
                device
            )
        )

        probabilities = torch.softmax(
            output,
            dim=1
        )

        confidence, prediction = (
            torch.max(
                probabilities,
                dim=1
            )
        )


# =========================================================
# PREDICTION RESULTS
# =========================================================

predicted_index = (
    prediction.item()
)

predicted_class = (
    CLASSES[
        predicted_index
    ]
)

confidence_value = (
    confidence.item()
    * 100
)


# =========================================================
# AI INSPECTION
# =========================================================

st.divider()

st.markdown("<div class=\"section-kicker\">Inference output</div>", unsafe_allow_html=True)
st.markdown("## AI defect detection", unsafe_allow_html=True)

left, right = (
    st.columns(2)
)


# =========================================================
# ORIGINAL WAFER
# =========================================================

with left:

    st.markdown(
        "### 🧿 Uploaded Wafer"
    )

    st.image(
        original_image,
        caption=uploaded_file.name,
        width="stretch"
    )


# =========================================================
# AI RESULT
# =========================================================

with right:

    st.markdown(
        "### 🤖 AI Detection"
    )

    st.markdown(
        f"""
        <div class="result-box">

        <h2>{predicted_class}</h2>

        <p>
        <b>Detected Defect:</b>
        {predicted_class}
        </p>

        <p>
        <b>AI Confidence:</b>
        {confidence_value:.2f}%
        </p>

        </div>
        """,
        unsafe_allow_html=True
    )

    st.progress(
        min(
            confidence_value / 100,
            1.0
        )
    )


# =========================================================
# CONFIDENCE STATUS
# =========================================================

if confidence_value >= 80:

    st.success(
        f"⚠️ AI strongly detects "
        f"**{predicted_class}** "
        f"({confidence_value:.2f}% confidence)"
    )

elif confidence_value >= 50:

    st.warning(
        f"⚠️ Possible defect: "
        f"**{predicted_class}** "
        f"({confidence_value:.2f}% confidence)"
    )

else:

    st.warning(
        f"⚠️ Low-confidence prediction: "
        f"**{predicted_class}** "
        f"({confidence_value:.2f}% confidence)"
    )


# =========================================================
# CLASS PROBABILITIES
# =========================================================

st.divider()

st.markdown("<div class=\"section-kicker\">Model distribution</div>", unsafe_allow_html=True)
st.markdown("## Eight-class probability field", unsafe_allow_html=True)

probabilities_np = (
    probabilities[
        0
    ]
    .cpu()
    .numpy()
)

for i, class_name in enumerate(
    CLASSES
):

    probability = (
        probabilities_np[i]
        * 100
    )

    st.write(
        f"**{class_name}** — "
        f"{probability:.2f}%"
    )

    st.progress(
        float(
            probabilities_np[i]
        )
    )


# =========================================================
# GRAD-CAM
# =========================================================

st.divider()

st.markdown("<div class=\"section-kicker\">Explainability</div>", unsafe_allow_html=True)
st.markdown("## Where the model looked", unsafe_allow_html=True)

st.write(
    """
    Grad-CAM highlights the regions of the
    wafer that contributed most strongly to
    the AI prediction.
    """
)

heatmap_image = None
cam_array = None

try:

    with st.spinner(
        "Generating AI attention heatmap..."
    ):

        heatmap_image, cam_array = (
            generate_gradcam(
                model,
                input_tensor,
                predicted_index
            )
        )

except Exception as e:

    st.error(
        "❌ Grad-CAM generation failed."
    )

    st.exception(e)


# =========================================================
# GRAD-CAM DISPLAY
# =========================================================

if heatmap_image is not None:

    original_rgb = (
        original_image
        .convert("RGB")
        .resize(
            (224, 224)
        )
    )

    original_rgb_array = (
        np.array(
            original_rgb
        ).astype(
            np.float32
        )
    )

    heatmap_rgb = (
        np.array(
            heatmap_image
            .convert("RGB")
        ).astype(
            np.float32
        )
    )

    overlay = (
        0.55
        * original_rgb_array
        +
        0.45
        * heatmap_rgb
    )

    overlay = np.clip(
        overlay,
        0,
        255
    ).astype(
        np.uint8
    )

    overlay_image = (
        Image.fromarray(
            overlay
        )
    )

    cam_col1, cam_col2 = (
        st.columns(2)
    )

    with cam_col1:

        st.markdown(
            "**🔥 AI Attention Heatmap**"
        )

        st.image(
            heatmap_image,
            caption="AI attention regions",
            width="stretch"
        )

    with cam_col2:

        st.markdown(
            "**🔥 Wafer + AI Attention**"
        )

        st.image(
            overlay_image,
            caption="Grad-CAM overlay",
            width="stretch"
        )


# =========================================================
# DEFECT LOCATION ANALYSIS
# =========================================================

st.divider()

st.markdown("<div class=\"section-kicker\">Spatial analysis</div>", unsafe_allow_html=True)
st.markdown("## Defect concentration map", unsafe_allow_html=True)

st.write(
    """
    This section estimates where defect pixels
    are concentrated on the wafer.
    """
)


# =========================================================
# DETERMINE DEFECT MAP
# =========================================================

unique_values = np.unique(
    wafer
)

# WM-811K style:
# 0 = background
# 1 = good die
# 2 = defective die

if set(
    unique_values
).issubset(
    {0, 1, 2}
):

    binary_map = (
        wafer == 2
    ).astype(
        np.uint8
    )

# Binary:
# 0 = good
# 1 = defect

elif set(
    unique_values
).issubset(
    {0, 1}
):

    binary_map = (
        wafer > 0
    ).astype(
        np.uint8
    )

else:

    binary_map = (
        wafer > 0
    ).astype(
        np.uint8
    )


# =========================================================
# DIMENSIONS
# =========================================================

height, width = (
    binary_map.shape
)


# =========================================================
# WAFER CENTER
# =========================================================

center_y = (
    height / 2
)

center_x = (
    width / 2
)


# =========================================================
# DISTANCE MAP
# =========================================================

yy, xx = np.indices(
    binary_map.shape
)

distance = np.sqrt(
    (
        xx - center_x
    ) ** 2
    +
    (
        yy - center_y
    ) ** 2
)

max_distance = (
    distance.max()
)

if max_distance > 0:

    normalized_distance = (
        distance
        / max_distance
    )

else:

    normalized_distance = (
        np.zeros_like(
            distance
        )
    )


# =========================================================
# REGION MASKS
# =========================================================

center_mask = (
    normalized_distance <= 0.33
)

middle_mask = (
    (
        normalized_distance > 0.33
    )
    &
    (
        normalized_distance <= 0.66
    )
)

edge_mask = (
    normalized_distance > 0.66
)


# =========================================================
# DEFECT COUNTS
# =========================================================

total_defects = int(
    binary_map.sum()
)

center_defects = int(
    (
        binary_map
        * center_mask
    ).sum()
)

middle_defects = int(
    (
        binary_map
        * middle_mask
    ).sum()
)

edge_defects = int(
    (
        binary_map
        * edge_mask
    ).sum()
)


# =========================================================
# DEFECT AREA
# =========================================================

total_pixels = (
    binary_map.size
)

if total_pixels > 0:

    defect_area_percentage = (
        total_defects
        / total_pixels
        * 100
    )

else:

    defect_area_percentage = 0.0


# =========================================================
# LOCATION PERCENTAGES
# =========================================================

if total_defects > 0:

    center_percentage = (
        center_defects
        / total_defects
        * 100
    )

    middle_percentage = (
        middle_defects
        / total_defects
        * 100
    )

    edge_percentage = (
        edge_defects
        / total_defects
        * 100
    )

else:

    center_percentage = 0.0
    middle_percentage = 0.0
    edge_percentage = 0.0


# =========================================================
# PRIMARY LOCATION
# =========================================================

location_values = {

    "Center":
        center_percentage,

    "Middle":
        middle_percentage,

    "Edge":
        edge_percentage
}

primary_location = max(
    location_values,
    key=location_values.get
)

primary_location_percentage = (
    location_values[
        primary_location
    ]
)


# =========================================================
# SPATIAL SPREAD
# =========================================================

if total_defects > 0:

    occupied_regions = 0

    if center_defects > 0:
        occupied_regions += 1

    if middle_defects > 0:
        occupied_regions += 1

    if edge_defects > 0:
        occupied_regions += 1

    spatial_spread = (
        occupied_regions
        / 3
        * 100
    )

else:

    spatial_spread = 0.0


# =========================================================
# LOCATION RISK
# =========================================================

if predicted_class in [
    "Edge-Ring",
    "Edge-Loc"
]:

    location_risk = (
        edge_percentage
    )

elif predicted_class == "Center":

    location_risk = (
        center_percentage
    )

elif predicted_class == "Donut":

    location_risk = max(
        middle_percentage,
        edge_percentage
    )

elif predicted_class == "Near-full":

    location_risk = 100.0

else:

    location_risk = (
        primary_location_percentage
    )


# =========================================================
# LOCATION METRICS
# =========================================================

loc1, loc2, loc3, loc4 = (
    st.columns(4)
)

with loc1:

    st.metric(
        "Defect Pixels",
        f"{total_defects:,}"
    )

with loc2:

    st.metric(
        "Defect Area",
        f"{defect_area_percentage:.2f}%"
    )

with loc3:

    st.metric(
        "Primary Location",
        primary_location
    )

with loc4:

    st.metric(
        "Spatial Spread",
        f"{spatial_spread:.1f}%"
    )


# =========================================================
# PRIMARY LOCATION MESSAGE
# =========================================================

st.info(
    f"📍 **Primary defect concentration:** "
    f"{primary_location} "
    f"({primary_location_percentage:.1f}% "
    f"of detected defect pixels)"
)


# =========================================================
# INTERPRETATION
# =========================================================

if predicted_class == "Center":

    interpretation = (
        "The AI classified the pattern as "
        "Center. The defect pattern is "
        "associated with the central "
        "wafer region."
    )

elif predicted_class == "Edge-Ring":

    interpretation = (
        "The AI classified the pattern as "
        "Edge-Ring. The predicted defect "
        "pattern is associated with the "
        "wafer edge/ring."
    )

elif predicted_class == "Edge-Loc":

    interpretation = (
        "The AI classified the pattern as "
        "Edge-Loc. A localized defect "
        "pattern is associated with the "
        "wafer edge."
    )

elif predicted_class == "Scratch":

    interpretation = (
        "The AI classified the pattern as "
        "Scratch. Inspect the spatial "
        "pattern for a scratch-like defect."
    )

else:

    interpretation = (
        f"The AI classified the pattern as "
        f"{predicted_class}. Spatial analysis "
        f"shows the highest concentration in "
        f"the {primary_location.lower()} region."
    )


st.write(
    interpretation
)


# =========================================================
# IMPORTANT LIMITATION
# =========================================================

st.warning(
    """
    **Important:** The location analysis identifies
    the spatial concentration of defect pixels.
    It does not determine the manufacturing process
    that caused the defect.
    """
)


# =========================================================
# FINAL MANUFACTURING RECOMMENDATION
# =========================================================

st.divider()

st.markdown("<div class=\"section-kicker\">Decision support</div>", unsafe_allow_html=True)
st.markdown("## Manufacturing disposition", unsafe_allow_html=True)


# =========================================================
# GET CLASS RISK FROM CSV
# =========================================================

rule_row = severity_rules[
    severity_rules[
        "defect_class"
    ]
    .astype(str)
    .str.strip()
    .str.lower()
    ==
    predicted_class
    .strip()
    .lower()
]


if rule_row.empty:

    class_risk = 50.0

else:

    class_risk = float(
        rule_row.iloc[0][
            "class_risk"
        ]
    )


# =========================================================
# INTERNAL AREA SCORE
# =========================================================

area_score = min(
    (
        defect_area_percentage
        / 20.0
    )
    * 100.0,
    100.0
)


# =========================================================
# INTERNAL SEVERITY CALCULATION
#
# This is NOT displayed to the user.
# =========================================================

severity_score = (

    0.40
    * class_risk

    +

    0.25
    * area_score

    +

    0.20
    * spatial_spread

    +

    0.15
    * location_risk
)


severity_score = float(
    np.clip(
        severity_score,
        0.0,
        100.0
    )
)


# =========================================================
# SPECIAL HIGH-RISK DEFECT
# =========================================================

if predicted_class == "Near-full":

    severity_score = max(
        severity_score,
        71.0
    )


# =========================================================
# FINAL DECISION
# =========================================================

if severity_score <= LOW_THRESHOLD:

    decision = (
        "CONTINUE PRODUCTION"
    )

    recommendation = (
        "The detected defect severity is low. "
        "Production can continue. Assign an "
        "engineer to inspect and monitor the "
        "affected wafer."
    )

    status_type = "success"


elif severity_score <= HIGH_THRESHOLD:

    decision = (
        "HOLD LOT"
    )

    recommendation = (
        "The detected defect severity is moderate. "
        "Hold the affected lot and assign an "
        "engineer for detailed inspection before "
        "releasing the lot."
    )

    status_type = "warning"


else:

    decision = (
        "STOP PRODUCTION LINE"
    )

    recommendation = (
        "The detected defect severity is high. "
        "Stop the affected production line and "
        "immediately escalate the issue to the "
        "engineering team."
    )

    status_type = "error"


# =========================================================
# FINAL RESULT
# =========================================================

if status_type == "success":

    st.success(
        "🟢 CONTINUE PRODUCTION"
    )

elif status_type == "warning":

    st.warning(
        "🟠 HOLD LOT"
    )

else:

    st.error(
        "🔴 STOP PRODUCTION LINE"
    )


# =========================================================
# FINAL RECOMMENDATION BOX
# =========================================================

st.markdown(
    f"""
    <div class="final-box">

        <h2>
        🤖 AI Recommendation
        </h2>

        <p>
        <b>Detected Defect:</b>
        {predicted_class}
        </p>

        <p>
        <b>AI Confidence:</b>
        {confidence_value:.2f}%
        </p>

        <p>
        <b>Recommended Action:</b>
        {decision}
        </p>

        <p>
        <b>Engineering Recommendation:</b>
        {recommendation}
        </p>

    </div>
    """,
    unsafe_allow_html=True
)


# =========================================================
# EXTRA ACTION MESSAGE
# =========================================================

if decision == "CONTINUE PRODUCTION":

    st.info(
        "👨‍🔧 Engineer inspection and monitoring recommended."
    )

elif decision == "HOLD LOT":

    st.warning(
        "🔎 Engineering approval is required before lot release."
    )

else:

    st.error(
        "🛑 Production intervention required."
    )


# =========================================================
# MODEL INFORMATION
# =========================================================

st.divider()

st.markdown("<div class=\"section-kicker\">System specification</div>", unsafe_allow_html=True)
st.markdown("## Inspection engine", unsafe_allow_html=True)

model_col1, model_col2 = (
    st.columns(2)
)

with model_col1:

    st.write(
        "**Architecture:** ResNet18"
    )

    st.write(
        "**Classes:** 8"
    )

    st.write(
        "**Input:** 224 × 224"
    )

    st.write(
        "**Training Dataset:** WM-811K / Defect8"
    )


with model_col2:

    st.write(
        "**Validation Accuracy:** 90.93%"
    )

    st.write(
        "**Macro F1:** 86.38%"
    )

    st.write(
        "**Inference Device:** CPU"
    )

    st.write(
        "**Explainability:** Grad-CAM"
    )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "Semiconductor Wafer AI Inspection System | "
    "ResNet18 + WM-811K + Grad-CAM + "
    "Manufacturing Recommendation Engine"
)