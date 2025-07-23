
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
from plotnine import geom_rect, ggplot, aes, geom_line, labs, theme, theme_matplotlib, ylim
from scipy.signal import stft

def get_column_figsize(columns=1, total_px=1000, dpi=100, height=3):
    col_px = total_px / columns
    width = col_px / dpi
    return (width, height), dpi

# OMT signal generator
def generate_omt_signal(t, omt_freq_hz, omt_freq_sd, amp_mean, amp_sd, phase_drift_sd, num_motor_units, rng):
    omt = np.zeros_like(t)
    for _ in range(num_motor_units):
        freq = rng.normal(omt_freq_hz, omt_freq_sd)
        amp = rng.normal(amp_mean, amp_sd)
        phase_noise = np.cumsum(rng.normal(0, phase_drift_sd, len(t)))
        omt += amp * np.sin(2 * np.pi * freq * t + phase_noise)
    return omt

def calculate_stft_loss(original_signal, reconstructed_signal, fs, nperseg, noverlap, alpha=1.0, scale_invariant=True):
    """
    Calculates single-resolution STFT loss between two signals, based on DDSP.
    """
    # Guard against empty signals
    if len(original_signal) == 0 or len(reconstructed_signal) == 0:
        return 0.0
        
    # Compute STFT for original signal
    _, _, Zxx_orig = stft(original_signal, fs=fs, nperseg=nperseg, noverlap=noverlap, window='hann')
    mag_orig = np.abs(Zxx_orig)

    # Compute STFT for reconstructed signal
    _, _, Zxx_rec = stft(reconstructed_signal, fs=fs, nperseg=nperseg, noverlap=noverlap, window='hann')
    mag_rec = np.abs(Zxx_rec)

    # Scale invariance (optional, based on DDSP paper)
    if scale_invariant:
        mean_orig = np.mean(mag_orig)
        mag_orig = mag_orig / (mean_orig + 1e-8)
        mag_rec = mag_rec / (mean_orig + 1e-8)  # Normalize by original's mean

    # L1 norm (mean absolute error) for magnitude
    mag_loss = np.mean(np.abs(mag_orig - mag_rec))

    # L1 norm for log-magnitude
    log_orig = np.log1p(mag_orig)
    log_rec = np.log1p(mag_rec)
    log_loss = np.mean(np.abs(log_orig - log_rec))

    total_loss = mag_loss + alpha * log_loss
    return total_loss

# --- Streamlit Layout ---
st.set_page_config(page_title="OMT STFT Visualiser", layout="wide")
st.title("OMT Signal: STFT Sliding Window and Frame FFT")

# --- Sidebar Parameters ---
with st.sidebar:
    duration = st.slider("Signal Duration (s)", 1.0, 5.0, 2.0, 0.5)
    fs = st.slider("Sampling Rate (Hz)", 500, 5000, 4800, 100)
    omt_freq_hz = st.slider("OMT Frequency (Hz)", 40, 100, 80)
    omt_freq_sd = st.slider("Frequency SD", 0.0, 10.0, 5.0)
    amp_mean = st.slider("Amplitude Mean (mV)", 0.1, 5.0, 1.0) * 1e-3
    amp_sd = st.slider("Amplitude SD (mV)", 0.0, 1.0, 0.2) * 1e-3
    phase_drift_sd = st.slider("Phase Drift SD", 0.0, 1.0, 0.1)
    num_motor_units = st.slider("Number of Motor Units", 1, 50, 10)
    nperseg = st.slider(
        "Window Duration (samples)", 100, 2000, 480, 10,
        help="Number of samples per STFT window (segment)."
    )
    hop_samples = st.slider(
        "Hop Duration (samples)", 1, nperseg, 120, 10,
        help="Number of samples to hop between STFT windows. This is the step size."
    )

# --- Signal Generation ---
t = np.linspace(0, duration, int(fs * duration))
rng = np.random.default_rng(seed=42)
signal = generate_omt_signal(t, omt_freq_hz, omt_freq_sd, amp_mean, amp_sd, phase_drift_sd, num_motor_units, rng)
target_signal = amp_mean * np.sin(2 * np.pi * omt_freq_hz * t)

# --- STFT Spectrogram ---
noverlap = nperseg - hop_samples
f_stft, t_stft, Zxx = stft(signal, fs=fs, nperseg=nperseg, noverlap=noverlap)
Zxx_mag = np.abs(Zxx)
Zxx_mag = Zxx_mag[(f_stft >= 20) & (f_stft <= 150), :]
f_crop = f_stft[(f_stft >= 20) & (f_stft <= 150)]

# --- Frame Management ---
if 'frame' not in st.session_state:
    st.session_state.frame = 0
n_frames = (len(signal) - nperseg) // hop_samples + 1  # manual frame count
# Increment early on rerun
if 'advance_frame' in st.session_state and st.session_state.advance_frame:
    st.session_state.frame = (st.session_state.frame + 1) % n_frames
    st.session_state.advance_frame = False  # Reset the flag

if st.session_state.frame >= n_frames:
    st.session_state.frame = 0

# --- Get Current Window ---
start_idx = st.session_state.frame * hop_samples
end_idx = start_idx + nperseg
frame_time = t[start_idx] if start_idx < len(t) else t[-1]

# --- Clip safely
end_idx = min(end_idx, len(signal))
t_window = t[start_idx:end_idx]
signal_window = signal[start_idx:end_idx]
target_window = target_signal[start_idx:end_idx]

# start_idx = int(frame_time * fs)
# end_idx = min(start_idx + nperseg, len(t))
# t_window = t[start_idx:end_idx]
# signal_window = signal[start_idx:end_idx]
# target_window = target_signal[start_idx:end_idx]

# --- Time-Domain Plot of Full Signal ---
st.subheader("Full OMT Signal")
window_s = nperseg / fs
df_time = pd.DataFrame({'time': t, 'amplitude': signal})
figsize, dpi = get_column_figsize(columns=1)
p_time = (
    ggplot(df_time, aes(x='time', y='amplitude')) +
    geom_line(color='steelblue') +
    geom_rect(
        aes(xmin=frame_time, xmax=frame_time + (nperseg / fs), ymin=min(signal), ymax=max(signal)),
        color="red", fill=None, linetype="dashed", size=0.5, inherit_aes=False
    ) +
    labs(x='Time (s)', y='Amplitude (V)') +
    theme_matplotlib() + 
    theme(figure_size=figsize, dpi=dpi)
)
st.pyplot(p_time.draw(), clear_figure=True)

# --- Frame Navigator ---
# if st.button("Next Frame"):
    # st.session_state.frame = (st.session_state.frame + 1) % n_frames

st.button("Next Frame", on_click=lambda: st.session_state.update(advance_frame=True))

st.markdown(f"**Frame {st.session_state.frame + 1} of {n_frames}**")


# --- Column Plots Definition ---
col1, col2 = st.columns(2)

# Plot 1: Current Window Signal
df_window = pd.DataFrame({'time': t_window, 'amplitude': signal_window})
figsize, dpi = get_column_figsize(columns=2)
p_window = (
    ggplot(df_window, aes(x='time', y='amplitude')) +
    geom_line(color='orange') + ylim(min(signal), max(signal)) +
    labs(x='Time (s)', y='Amplitude (V)') +
    theme_matplotlib() +
    theme(figure_size=figsize, dpi=dpi)
)

# Plot 2: FFT of Current Window
if len(signal_window) > 0:
    fft_freqs = np.fft.rfftfreq(len(signal_window), d=1/fs)
    fft_mag = np.abs(np.fft.rfft(signal_window))
    df_fft = pd.DataFrame({'freq': fft_freqs, 'magnitude': fft_mag})
    df_fft = df_fft[(df_fft['freq'] >= 20) & (df_fft['freq'] <= 150)]
    p_fft = (
        ggplot(df_fft, aes(x='freq', y='magnitude')) + geom_line(color='crimson') +
        labs(x='Frequency (Hz)', y='Magnitude') +
        theme_matplotlib() +
        theme(figure_size=figsize, dpi=dpi)
    )
else:
    p_fft = ggplot() + theme(figure_size=figsize, dpi=dpi)

# Plot 3: STFT Spectrogram
figsize, dpi = get_column_figsize(columns=1)
fig_spec, ax_spec = plt.subplots(figsize=figsize, dpi=dpi)
ax_spec.pcolormesh(t_stft, f_crop, Zxx_mag, shading='gouraud', cmap='viridis')
ax_spec.set(xlabel='Time (s)', ylabel='Frequency (Hz)', ylim=(20, 150))
ax_spec.axvline(frame_time, color='cyan', linestyle='--', linewidth=2)


# --- Column Layout ---
with col1:
    st.subheader("Current Window Signal")
    st.pyplot(p_window.draw(), clear_figure=True)
with col2:
    st.subheader("Sliding Window FFT")
    st.pyplot(p_fft.draw(), clear_figure=True)

st.subheader("STFT Spectrogram")
st.pyplot(fig_spec, clear_figure=True)

# --- Loss Calculation Section ---
st.divider()
st.header("📉 Spectral Loss Calculation for Current Frame")

# Calculate loss on the current window
loss_nperseg = len(signal_window)
loss_noverlap = 0 
stft_loss = calculate_stft_loss(target_window, signal_window, fs, loss_nperseg, loss_noverlap)

st.metric(
    label=f"Frame {st.session_state.frame + 1} STFT Loss",
    value=f"{stft_loss:.4f}",
    help="Loss between the OMT signal and a pure sine wave within the current window. Lower is better."
)

# --- Comparison Plot ---
st.subheader("Signal vs. Target in Current Window")
df_compare = pd.DataFrame({
    'time': t_window,
    'OMT Signal (Reconstructed)': signal_window,
    'Target Sine Wave (Original)': target_window
})
df_compare_melted = df_compare.melt('time', var_name='Signal Type', value_name='Amplitude')
figsize, dpi = get_column_figsize(columns=1)
p_compare = (
    ggplot(df_compare_melted, aes(x='time', y='Amplitude', color='Signal Type')) +
    geom_line() +
    labs(x='Time (s)', y='Amplitude (V)') +
    ylim(min(signal), max(signal)) +
    theme_matplotlib() +
    theme(legend_position='top', figure_size=figsize, dpi=dpi)
)
st.pyplot(p_compare.draw(), clear_figure=True)