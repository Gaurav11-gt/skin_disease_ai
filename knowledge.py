"""Plain-language explanations + recommended next steps (general information, NOT medical advice)."""
from config import CLASS_NAMES

INFO = {
    "akiec": dict(risk="Moderate-High", desc="Rough, scaly patches caused by long-term sun damage. Some can progress to squamous cell carcinoma, so they are treated as pre-cancerous.",
                  step="Book a dermatologist appointment in the coming weeks for examination; treatment decisions need a clinician."),
    "bcc": dict(risk="High", desc="The most common skin cancer. It grows slowly and rarely spreads, but it can damage surrounding skin if untreated.",
                step="See a dermatologist soon (within a few weeks) for examination and possible biopsy."),
    "bkl": dict(risk="Low", desc="A group of non-cancerous 'age spots' / waxy growths such as seborrheic keratosis and solar lentigo.",
                step="Usually harmless. Monitor it; see a doctor if it changes, bleeds, itches or you are unsure."),
    "df": dict(risk="Low", desc="A firm, benign skin nodule, often on the legs, commonly after a minor injury.",
               step="Typically needs no treatment. See a doctor if it grows, becomes painful or changes."),
    "mel": dict(risk="Very High", desc="The most dangerous skin cancer. It can spread if not caught early; early detection greatly improves outcomes.",
                step="Seek a dermatologist evaluation as soon as possible (urgent). Do not wait for it to change further."),
    "nv": dict(risk="Low", desc="A common mole made of pigment cells. Most are harmless.",
               step="Monitor with the ABCDE rule (Asymmetry, Border, Colour, Diameter, Evolution) and have routine skin checks."),
    "vasc": dict(risk="Low", desc="Lesions made of blood vessels (e.g. cherry angioma). Usually benign.",
                 step="Usually harmless. See a doctor if it bleeds repeatedly, grows quickly or you are unsure."),
}
HIGH_CONCERN = {"mel", "bcc", "akiec"}
DISCLAIMER = ("This is a research prototype for preliminary screening support only. It is NOT a medical diagnosis and does "
              "not replace a qualified dermatologist. It was trained on dermoscopic images and may be unreliable on "
              "ordinary phone photos.")


def explain(code, confidence, probs_by_class, low_conf=0.50):
    info = INFO[code]
    text = f"**Predicted: {CLASS_NAMES[code]}** (confidence {confidence:.0%}). {info['desc']}"
    step = info["step"]
    # safety net: if the model is unsure but any concerning class has non-trivial probability, escalate
    worry = {c: p for c, p in probs_by_class.items() if c in HIGH_CONCERN and p >= 0.15}
    if confidence < low_conf:
        step = ("The model is NOT confident about this image. Please do not rely on the result; consult a dermatologist, "
                "especially if the lesion is new, changing, bleeding or itching.")
    elif worry and code not in HIGH_CONCERN:
        step = (f"Although the top prediction looks benign, a non-trivial chance ({', '.join(f'{c} {p:.0%}' for c, p in worry.items())}) "
                "was given to a concerning class. A dermatologist check is advised.")
    return text, step, info["risk"]
