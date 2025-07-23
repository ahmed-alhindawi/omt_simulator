import numpy as np
import streamlit as st
import polars as pl
import plotnine as p9


def generate_omt_signal(t, omt_freq_hz, omt_freq_sd, amp_mean, amp_sd, phase_drift_sd, num_motor_units, rng):
    omt = np.zeros_like(t)
    for _ in range(num_motor_units):
        freq = rng.normal(omt_freq_hz, omt_freq_sd)
        amp = rng.normal(amp_mean, amp_sd)
        phase_noise = np.cumsum(rng.normal(0, phase_drift_sd, len(t)))
        omt += amp * np.sin(2 * np.pi * freq * t + phase_noise)
    return omt


st.set_page_config(page_title="OMT Metrics Visualiser", layout="wide")
st.title("OMT Frequency and Power Capture Metrics")
# --- UI Sidebar ---
try:
    with st.sidebar:
        st.markdown("### Signal Parameters")
        fs = st.slider("Sampling Frequency (Hz)", 100, 9600, 4800, step=100)
        duration = st.slider("Signal Duration (s)", 0.1, 5.0, 1.0, step=0.1)

        omt_freq_hz = st.slider("OMT Mean Frequency (Hz)", 50, 150, 80)
        omt_freq_sd = st.slider("OMT Frequency SD (Hz)", 0.1, 10.0, 1.0, step=0.1)
        amp_mean = st.slider("Motor Unit Amplitude Mean", 0.5, 5.0, 1.0, step=0.1)
        amp_sd = st.slider("Motor Unit Amplitude SD", 0.0, 2.0, 0.3, step=0.1)
        phase_drift_sd = st.slider("Phase Drift SD", 0.0, 0.1, 0.01, step=0.005)
        num_motor_units = st.slider("Number of Motor Units", 1, 20, 10)

        st.markdown("### Capture Metrics")
        alpha = st.slider("α (Power Threshold)", 0.1, 1.0, 0.5, step=0.01)
        freq_pad = st.slider("±Hz Window for Power Capture", 1.0, 50.0, 5.0, step=1.0)
except Exception as e:
    st.error(f"Sidebar error: {e}")


# --- Generate Signal ---
try:
    rng = np.random.default_rng(42)
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    signal = generate_omt_signal(t, omt_freq_hz, omt_freq_sd, amp_mean, amp_sd, phase_drift_sd, num_motor_units, rng)
except Exception as e:
    st.error(f"Signal generation error: {e}")

# --- FFT and Power Spectrum ---
try:
    N = len(signal)
    fft_vals = np.fft.fft(signal)
    freqs = np.fft.fftfreq(N, d=1 / fs)
    pos_mask = freqs >= 0
    freqs = freqs[pos_mask]
    power = np.abs(fft_vals[pos_mask]) ** 2
    log_power = np.log10(power + 1e-12)
except Exception as e:
    st.error(f"FFT error: {e}")

# --- Capture Metrics ---
try:
    total_power = power.sum()
    sorted_indices = np.argsort(power)[::-1]
    cum_power = 0.0
    selected_freqs = []

    for idx in sorted_indices:
        cum_power += power[idx]
        selected_freqs.append(freqs[idx])
        if cum_power >= alpha * total_power:
            break

    band_min = min(selected_freqs)
    band_max = max(selected_freqs)
    freq_captured = band_min <= omt_freq_hz <= band_max

    band_mask = (freqs >= omt_freq_hz - freq_pad) & (freqs <= omt_freq_hz + freq_pad)
    band_power = power[band_mask].sum()
    power_capture = band_power / total_power
except Exception as e:
    st.error(f"Metric calculation error: {e}")

# --- Log Power Spectrum DataFrame ---
try:
    df = pl.DataFrame({
        "Frequency": freqs,
        "LogPower": log_power
    }).filter((pl.col("Frequency") >= 20) & (pl.col("Frequency") <= 150)).sort("Frequency")

    # Get vertical band height for plotting
    ymin = float(df["LogPower"].min())
    ymax = float(df["LogPower"].max())

    df_bands = pl.DataFrame({
        "xmin": [band_min, omt_freq_hz - freq_pad],
        "xmax": [band_max, omt_freq_hz + freq_pad],
        "ymin": [ymin, ymin],
        "ymax": [ymax, ymax],
        "fill": ["Frequency Capture", "Power Capture"]
    })

    df_plot = df.to_pandas()
    df_bands_plot = df_bands.to_pandas()
except Exception as e:
    st.error(f"Data preparation error: {e}")

# --- Time-Domain Plot ---
try:
    df_time = pl.DataFrame({
        "Time": t,
        "Amplitude": signal
    }).to_pandas()

    p_time = (
        p9.ggplot(df_time, p9.aes(x="Time", y="Amplitude"))
        + p9.geom_line(color="steelblue")
        + p9.labs(
            x="Time (s)",
            y="Amplitude"
        )
        + p9.theme_bw()
        + p9.theme(
            figure_size=(10, 3),
            axis_title=p9.element_text(size=12),
            plot_title=p9.element_text(size=14, weight="bold")
        )
    )
    st.pyplot(p_time.draw())
except Exception as e:
    st.error(f"Plotting time-domain signal error: {e}")

# --- Frequency-Domain Plot ---
try:
    p = (
        p9.ggplot(df_plot, p9.aes(x="Frequency", y="LogPower"))
        + p9.geom_line(color="steelblue")
        + p9.geom_rect(
            df_bands_plot,
            p9.aes(xmin="xmin", xmax="xmax", ymin="ymin", ymax="ymax", fill="fill"),
            alpha=0.3,
            inherit_aes=False
        )
        + p9.geom_vline(xintercept=omt_freq_hz, linetype="dashed", color="red", size=1.0)
        + p9.coord_cartesian(xlim=(20, 150))
        + p9.scale_y_continuous(limits=(ymin, ymax), expand=[0, 0])
        + p9.labs(
            x="Frequency (Hz)",
            y="log₁₀ Power",
            fill="Metric"
        )
        + p9.theme_bw()
        + p9.theme(
            figure_size=(10, 5),
            dpi=100,
            legend_position=(0.99, 0.99),
            legend_background=p9.element_rect(fill="white", alpha=0.8),
            axis_title=p9.element_text(size=12),
            plot_title=p9.element_text(size=14, weight="bold")
        )
    )
    st.pyplot(p.draw())
except Exception as e:
    st.error(f"Plotting spectrum error: {e}")



# --- Display Metrics ---
try:
    st.markdown("### Metrics")
    st.markdown(f"- **FrequencyCaptureMetric**: `{freq_captured}` → Band = [{band_min:.2f}, {band_max:.2f}] Hz")
    st.markdown(f"- **PowerCaptureMetric**: `{power_capture:.4f}` of total power in [{omt_freq_hz - freq_pad:.1f}, {omt_freq_hz + freq_pad:.1f}] Hz")
except Exception as e:
    st.error(f"Metrics display error: {e}")
