from __future__ import annotations

import tempfile
from pathlib import Path

import streamlit as st

# This file is located at: <project_root>/Dashboard/1_DriftSense_Navigation.py
PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOGO_PATH = PROJECT_ROOT / "Dashboard" / "assets" / "athena_tech_logo.jpeg"
DEFAULT_RESNET = PROJECT_ROOT / "Results" / "Training" / "best_model.pth"

st.set_page_config(page_title="ATHENA TECH | DriftSense Navigation", page_icon="🧭", layout="wide")

st.markdown(
    """
    <style>
    :root { color-scheme: dark; }
    html, body, .stApp, [data-testid="stAppViewContainer"] { background: #060b12 !important; color: #ffffff !important; }
    [data-testid="stHeader"] { background: #060b12 !important; }
    [data-testid="stSidebar"] { background: #09121d !important; border-right: 1px solid #213347 !important; }
    [data-testid="stSidebar"] * { color: #ffffff !important; }
    h1, h2, h3, h4, p, label, span, div, small, li, [data-testid="stMarkdownContainer"], [data-testid="stWidgetLabel"] { color: #ffffff !important; }
    [data-testid="stCaptionContainer"] p { color: #ffffff !important; opacity: .78; }
    input, textarea { background: #0d1825 !important; color: #ffffff !important; border: 1px solid #2c435a !important; }
    /* Upload dropzone only: black text on the white drag-and-drop surface. */
    [data-testid="stFileUploader"] {
        background: #ffffff !important;
        border: 1px solid #9aa9b5 !important;
        border-radius: 12px !important;
        color: #000000 !important;
    }
    [data-testid="stFileUploader"] section,
    [data-testid="stFileUploader"] section > div {
        background: #ffffff !important;
    }
    [data-testid="stFileUploader"] label,
    [data-testid="stFileUploader"] small,
    [data-testid="stFileUploader"] span,
    [data-testid="stFileUploader"] p,
    [data-testid="stFileUploader"] div {
        color: #000000 !important;
    }
    [data-testid="stFileUploader"] svg {
        color: #000000 !important;
        fill: #000000 !important;
    }
    [data-testid="stFileUploader"] button {
        background: #10263a !important;
        color: #ffffff !important;
        border: 1px solid #10263a !important;
    }
    [data-testid="stFileUploader"] button * {
        color: #ffffff !important;
    }
    button { background: #10263a !important; color: #ffffff !important; border: 1px solid #27b7ff !important; }
    button:hover { background: #164361 !important; }
    [data-testid="stMetric"] { background: #0d1825 !important; border: 1px solid #2c435a !important; border-radius: 12px !important; }
    [data-testid="stMetricLabel"], [data-testid="stMetricValue"] { color: #ffffff !important; }
    [data-testid="stAlert"] { color: #ffffff !important; background: #10263a !important; }
    hr { border-color: #213347 !important; }
    .athena-header { border-bottom: 1px solid #213347; padding: 6px 0 24px; margin-bottom: 28px; }
    .eyebrow { color: #27b7ff !important; letter-spacing: 2px; font-size: 12px; font-weight: 700; }
    .athena-title { color: #ffffff !important; font-size: 42px; font-weight: 700; margin: 8px 0; }
    .athena-subtitle { color: #ffffff !important; opacity: .82; font-size: 16px; }
    </style>
    """,
    unsafe_allow_html=True,
)

if LOGO_PATH.exists():
    st.sidebar.image(str(LOGO_PATH), width=190)
st.sidebar.markdown("### ATHENA TECH")
st.sidebar.caption("Vision, Value, Victory · Semiconductor Intelligence")
st.sidebar.divider()
st.sidebar.markdown("**01 · DriftSense Navigation**")
st.sidebar.markdown("02 · Wafer Defect Detection")

st.markdown(
    """
    <div class="athena-header">
      <div class="eyebrow">ATHENA TECH · NAVIGATION INTELLIGENCE · DRIFTSENSE</div>
      <div class="athena-title">DriftSense Navigation</div>
      <div class="athena-subtitle">Sensorless wafer-site recovery, correction estimation, and confidence-aware inspection handoff.</div>
    </div>
    """,
    unsafe_allow_html=True,
)
st.write(
    "Use the hybrid DriftSense pipeline to locate the reference site, estimate a navigation correction, "
    "and classify the recovered crop with the existing ResNet18 model when localization is actionable."
)

ranker_path = st.text_input(
    "HYBRID RANKER CHECKPOINT",
    value="",
    help="Path to DriftSense hybrid_v5/best_ranker.pth. This may be in the separate DriftSense project.",
)
resnet_path = st.text_input(
    "RESNET18 CHECKPOINT",
    value=str(DEFAULT_RESNET),
)

ref_file = st.file_uploader("REFERENCE IMAGE", type=["png", "jpg", "jpeg"])
search_file = st.file_uploader("SEARCH IMAGE", type=["png", "jpg", "jpeg"])

if ref_file is not None:
    st.image(ref_file, caption="Reference image", width="stretch")
if search_file is not None:
    st.image(search_file, caption="Search image", width="stretch")

run = st.button("Run DriftSense + ResNet18", type="primary", disabled=not (ref_file and search_file and ranker_path))

if run:
    ranker = Path(ranker_path).expanduser()
    resnet = Path(resnet_path).expanduser()
    if not ranker.exists():
        st.error(f"Ranker checkpoint not found: {ranker}")
        st.stop()
    if not resnet.exists():
        st.error(f"ResNet18 checkpoint not found: {resnet}")
        st.stop()

    with tempfile.TemporaryDirectory(prefix="driftsense_case_") as temp_dir:
        case_dir = Path(temp_dir) / "uploaded_case"
        case_dir.mkdir()
        (case_dir / "reference.png").write_bytes(ref_file.getvalue())
        (case_dir / "search.png").write_bytes(search_file.getvalue())

        try:
            from AI.Integration.driftsense_resnet_pipeline import run_pipeline
            result = run_pipeline(case_dir, ranker, resnet, Path(temp_dir) / "output")
        except Exception as exc:
            st.error("The integrated DriftSense pipeline failed.")
            st.exception(exc)
            st.stop()

    localization = result["localization"]
    selected = localization["selected"]
    correction = result["correction"]

    st.divider()
    st.subheader("NAVIGATION RESULT")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Status", localization["status"])
    with col2:
        st.metric("Confidence margin", f"{localization['confidence']:.4f}")
    with col3:
        st.metric("Selected X", f"{selected['x']:.1f} px")
    with col4:
        st.metric("Selected Y", f"{selected['y']:.1f} px")

    st.write(
        f"Recommended image-space correction: **dx = {correction['dx_px']:.1f} px**, "
        f"**dy = {correction['dy_px']:.1f} px**, magnitude **{correction['magnitude_px']:.1f} px**."
    )

    if localization["status"] == "actionable":
        st.success("Localization is actionable; the recovered crop was passed to ResNet18.")
        defect = result.get("defect_classification", {})
        st.subheader("RESNET18 DEFECT RESULT")
        d1, d2 = st.columns(2)
        with d1:
            st.metric("Defect class", defect.get("class_name", "Unknown"))
        with d2:
            st.metric("Classifier confidence", f"{100 * defect.get('confidence', 0.0):.2f}%")
    else:
        st.warning("Localization is uncertain; ResNet18 classification was intentionally withheld.")

    st.subheader("CANDIDATE RANKING")
    candidates = localization.get("ranked_candidates", localization.get("candidates", []))
    st.dataframe(candidates, use_container_width=True)

    with st.expander("RAW INTEGRATED RESULT"):
        st.json(result)

st.caption("ATHENA TECH · Correction values are image-space recommendations only; they are not direct hardware commands.")
