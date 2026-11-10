"""Streamlit demo:  streamlit run app.py -- --run outputs/simple_cnn_s42"""
import argparse, sys
import streamlit as st
from PIL import Image
from knowledge import DISCLAIMER
from inference import Predictor

ap = argparse.ArgumentParser()
ap.add_argument("--run", default="outputs/simple_cnn_s42")
args, _ = ap.parse_known_args(sys.argv[1:])

st.set_page_config(page_title="AI Skin Lesion Screening", page_icon="🩺", layout="wide")
st.title("🩺 AI-Assisted Skin Disease Classification")
st.warning(DISCLAIMER)


@st.cache_resource
def load(run):
    return Predictor(run)


pred = load(args.run)
file = st.file_uploader("Upload a skin lesion image (JPG/PNG)", type=["jpg", "jpeg", "png"])
if file:
    img = Image.open(file)
    r = pred.predict(img)
    c1, c2, c3 = st.columns(3)
    c1.image(img, caption="Input", use_container_width=True)
    c2.image(r["heatmap"], caption="Grad-CAM: regions that influenced the prediction", use_container_width=True)
    with c3:
        st.metric("Top prediction", r["code"].upper(), f"{r['confidence']:.0%} confidence")
        st.write(f"**Risk level of predicted class:** {r['risk']}")
        top3 = sorted(r["probs"].items(), key=lambda kv: -kv[1])[:3]
        for k, v in top3:
            st.progress(v, text=f"{k}: {v:.1%}")
    if r["quality_warning"]:
        st.info("The image looks blurry/low-detail; results may be unreliable. Retake in good light, in focus.")
    st.subheader("Explanation"); st.markdown(r["explanation"])
    st.subheader("Recommended next step"); st.error(r["next_step"]) if r["risk"] in ("High", "Very High") else st.success(r["next_step"])
