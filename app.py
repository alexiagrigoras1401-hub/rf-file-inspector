import os
import tempfile
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

from rf_core import analyze_path

st.set_page_config(page_title="Signal Intelligence", layout="wide")
import base64

def set_background(image_file):
    with open(image_file, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()

    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

set_background("alexiateo.jpeg")



st.title("Aplicatie Signal Intelligence")
st.caption("Analizează fișiere autorizate: tip semnal (heuristic) + bruiaj + tip bruiaj.")

colL, colR = st.columns([1, 1])

with colL:
    up = st.file_uploader("Încarcă fișier (.dat/.iq/.bin sau .wav)", type=None)

    kind = st.selectbox("Tip fișier", ["IQ (interleaved I/Q)", "WAV"], index=0)
    kind_key = "iq" if kind.startswith("IQ") else "wav"

    fs = None
    dtype = "int16"
    if kind_key == "iq":
        fs = st.number_input("Sample rate (Hz) – obligatoriu pentru IQ", min_value=1.0, value=2400000.0, step=1000.0)
        dtype = st.selectbox("IQ dtype", ["int16", "int8", "float32"], index=0)

    center_hz = st.number_input("Center frequency (Hz) – opțional (doar raportare)", min_value=0.0, value=0.0, step=1.0)
    center_hz = None if center_hz == 0.0 else float(center_hz)

    run = st.button("Analizează", type="primary", disabled=(up is None))

with colR:
    st.subheader("Rezultate")

    if run and up is not None:
        # save upload to temp
        suffix = os.path.splitext(up.name)[1] or ".bin"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tf:
            tf.write(up.getbuffer())
            tmp_path = tf.name

        try:
            rep, (freqs, psd_db) = analyze_path(
                path=tmp_path,
                kind=kind_key,
                fs=float(fs) if fs is not None else None,
                dtype=dtype,
                center_hz=center_hz,
            )

            # Pretty summary
            a, b, c = st.columns(3)
            a.metric("Tip semnal (heuristic)", rep.signal_type)
            b.metric("Bruiaj", "DA" if rep.jammed else "NU")
            c.metric("Tip bruiaj", rep.jamming_type)

            st.write(f"**Noise floor (median PSD):** `{rep.noise_floor_db:.2f} dB/Hz`")
            st.write(f"**Samples analizate:** `{rep.n_samples:,}`   |   **Fs:** `{rep.fs_hz:,.0f} Hz`")

            # PSD plot
            fig = plt.figure(figsize=(10, 4))
            plt.plot(freqs, psd_db, lw=1.0)
            plt.grid(True, alpha=0.3)
            plt.title("PSD")
            plt.xlabel("Frequency (Hz)")
            plt.ylabel("PSD (dB/Hz)")
            st.pyplot(fig, clear_figure=True)

            # Diagnostics
            with st.expander("Diagnostic (features)"):
                st.json(rep.diagnostics)

            # Downloads
            st.download_button(
                label="Descarcă raport JSON",
                data=rep.to_json_bytes(),
                file_name="raport.json",
                mime="application/json",
            )

        except Exception as e:
            st.error(f"Eroare la analiză: {e}")
        finally:
            try:
                os.remove(tmp_path)
            except Exception:
                pass
    else:
        st.info("Încarcă un fișier și apasă **Analizează**.")
