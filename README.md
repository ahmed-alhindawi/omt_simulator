# 🧠 OMT STFT Visualiser

An interactive **Streamlit app** for visualising **Ocular Microtremor (OMT)** signals, their **Short-Time Fourier Transforms (STFT)**, and **spectral differences** from a reference sine wave. Designed for clinical or biomedical signal analysis, especially in contexts such as sedation monitoring or brainstem physiology.

---

## 🔧 Features

- 📈 **Synthetic OMT signal generator** with customisable motor unit parameters
- 🎛️ **Adjustable STFT window and hop size**
- 🔁 **Frame-by-frame navigation** through time-frequency windows
- 🟥 **Red rectangle** shows the current analysis window on the full time-domain signal
- 📊 **Current frame FFT**
- 🌈 **STFT spectrogram** with a sliding vertical line
- 📉 **STFT spectral loss metric** compared to a pure sine wave
- 🧪 **Overlay comparison** between true and synthetic signals for each window

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/ahmed-alhindawi/omt-simulator.git
cd omt-stft-visualiser
```

### 2. Install dependenciecs
```
pip install -r requirements.txt
```

### 3. Launch the app
```
streamlit run src/interactive_stft.py`
```

## 🎛️ Controls

### 🔧 Signal Parameters (in sidebar)
- **Signal Duration (s)**: Total duration of the synthetic OMT signal
- **Sampling Rate (Hz)**: Samples per second (affects STFT resolution)
- **OMT Frequency (Hz)**: Central frequency of motor unit oscillations
- **Frequency SD (Hz)**: Variability in frequency across motor units
- **Amplitude Mean (mV)**: Mean amplitude of each motor unit
- **Amplitude SD (mV)**: Amplitude variability across units
- **Phase Drift SD**: Random walk noise applied to the phase
- **Number of Motor Units**: Total motor units contributing to the signal

### 🎚️ STFT Parameters
- **Window Duration (samples)**: Number of samples in each STFT window (`nperseg`)
- **Hop Duration (samples)**: Number of samples between successive windows

### 🔄 Frame Navigation
- **Next Frame button**: Advances to the next windowed segment of the signal
- **Frame counter**: Displays the current frame number out of total frames
- **Red rectangle**: Highlights the current window on the full time-domain signal
- **Cyan vertical line**: Marks current window position on the STFT spectrogram

## 📉 STFT Loss Metric
The app computes a spectral loss between the synthetic OMT signal and a pure sine wave within the current analysis window.

**Components:**
- **Magnitude Loss**: Mean absolute difference of STFT magnitudes
- **Log Loss**: Mean absolute difference of log(1 + magnitude)

**Total Loss = Magnitude Loss + α × Log Loss**

The loss is scale-invariant by default (spectra are normalised to the mean magnitude of the original signal). Lower values indicate spectral similarity to a clean sine wave.

**Live Output**:
- The loss for the current frame is shown as a metric on the page.
- Frame-by-frame comparisons of OMT vs. target signal are visualised below the loss.
