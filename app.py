import time

import requests
import streamlit as st

DEFAULT_ENDPOINT = "http://127.0.0.1:8085/serve/sms_spam_clean_v3"

SAMPLES = [
    "Congratulations! You've won a $1000 gift card. Click here to claim now!",
    "Hey, are we still meeting for lunch tomorrow?",
    "URGENT! Your account has been compromised. Send your password immediately to verify.",
]

# ── page config ───────────────────────────────────────────────────────│

st.set_page_config(page_title="SMS Spam Classifier", page_icon="📩")
st.title("📩 SMS Spam Classifier")
st.markdown("Enter an SMS message and get a prediction from the deployed model.")

# ── sidebar — endpoint config ─────────────────────────────────────────│

endpoint = st.sidebar.text_input(
    "ClearML Serving endpoint",
    value=DEFAULT_ENDPOINT,
    help="URL of the deployed ClearML Serving model endpoint",
)

# ── main — input ──────────────────────────────────────────────────────│

text = st.text_area(
    "SMS message",
    value="",
    height=120,
    placeholder="Type or paste an SMS message here...",
)

predict_btn = st.button("Predict", type="primary")

# ── prediction ────────────────────────────────────────────────────────│

if predict_btn:
    if not text.strip():
        st.warning("Please enter a text message.")
    else:
        start = time.perf_counter()
        try:
            response = requests.post(
                endpoint,
                json={"text": text.strip()},
                timeout=10,
            )
            response.raise_for_status()
            result = response.json()
            latency_ms = (time.perf_counter() - start) * 1000

            label = result.get("label", "unknown")
            color = "red" if label == "spam" else "green"

            st.markdown(f"### Prediction: :{color}[**{label}**]")
            st.caption(f"Latency: **{latency_ms:.1f} ms**")

        except requests.exceptions.Timeout:
            st.error("Request timed out. The server may be overloaded or unreachable.")
        except requests.exceptions.ConnectionError:
            st.error(
                f"Cannot connect to the endpoint: `{endpoint}`. "
                "Make sure ClearML Serving is running."
            )
        except requests.exceptions.HTTPError as e:
            st.error(f"HTTP error: {e}")
        except Exception as e:
            st.error(f"Unexpected error: {e}")

# ── examples ──────────────────────────────────────────────────────────│

st.divider()
st.subheader("🧪 Try these examples")
for i, sample in enumerate(SAMPLES, 1):
    st.code(sample, language=None)
