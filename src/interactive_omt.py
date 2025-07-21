import numpy as np
import pandas as pd
import streamlit as st
from plotnine import ggplot, aes, geom_line, theme_bw, labs, ggtitle, theme, theme_matplotlib
from plotnine import element_text


# OMT signal generator
def generate_omt_signal(t, omt_freq_hz, omt_freq_sd, amp_mean, amp_sd, phase_drift_sd, num_motor_units, rng):
    omt = np.zeros_like(t)
    for _ in range(num_motor_units):
        freq = rng.normal(omt_freq_hz, omt_freq_sd)
        amp = rng.normal(amp_mean, amp_sd)
        phase_noise = np.cumsum(rng.normal(0, phase_drift_sd, len(t)))
        omt += amp * np.sin(2 * np.pi * freq * t + phase_noise)
    return omt


# Streamlit UI
st.set_page_config(page_title="OMT Signal Simulator", layout="wide")
st.title("Ocular Microtremor (OMT) Signal Simulator")

# Sidebar parameters
with st.sidebar:
    duration = st.slider("Duration (s)", 0.1, 2.0, 1.0, 0.1)
    fs = st.slider("Sampling Rate (Hz)", 500, 5000, 2400, 100)
    omt_freq_hz = st.slider("Mean Frequency (Hz)", 40.0, 100.0, 80.0, 1.0)
    omt_freq_sd = st.slider("Frequency SD (Hz)", 0.0, 10.0, 5.0, 0.1)
    amp_mean = st.slider("Amplitude Mean (mV)", 0.0, 5.0, 1.0, 0.1) * 1e-3
    amp_sd = st.slider("Amplitude SD (mV)", 0.0, 1.0, 0.2, 0.01) * 1e-3
    phase_drift_sd = st.slider("Phase Drift SD", 0.0, 1.0, 0.1, 0.01)
    num_motor_units = st.slider("Number of Motor Units", 1, 50, 10, 1)

# Generate signal
t = np.linspace(0, duration, int(duration * fs))
rng = np.random.default_rng(seed=42)
omt_signal = generate_omt_signal(t, omt_freq_hz, omt_freq_sd, amp_mean, amp_sd, phase_drift_sd, num_motor_units, rng)

# Prepare DataFrames for Plotnine
df_time = pd.DataFrame({"time": t, "amplitude": omt_signal})
freqs = np.fft.rfftfreq(len(t), d=1 / fs)[20:200]
fft_vals = np.abs(np.fft.rfft(omt_signal))[20:200]
df_fft = pd.DataFrame({"frequency": freqs, "magnitude": fft_vals})

# Create Plotnine plots
p_time = (
    ggplot(df_time, aes(x="time", y="amplitude"))
    + geom_line(color="steelblue")
    + labs(x="Time (s)", y="Amplitude (V)")
    + ggtitle("Time Domain Signal")
    + theme_matplotlib()
    + theme(plot_title=element_text(ha="center"))
)

p_fft = (
    ggplot(df_fft, aes(x="frequency", y="magnitude"))
    + geom_line(color="darkred")
    + labs(x="Frequency (Hz)", y="Magnitude")
    + ggtitle("Frequency Domain (FFT)")
    + theme_matplotlib()
    + theme(plot_title=element_text(ha="center"))
)

# Display side-by-side
col1, col2 = st.columns(2)
with col1:
    st.pyplot(p_time.draw(), clear_figure=True)
with col2:
    st.pyplot(p_fft.draw(), clear_figure=True)


### TO RUN THIS, TYPE:
# streamlit run src/scripts/interactive_omt.py
# in the terminal from the project root directory.
