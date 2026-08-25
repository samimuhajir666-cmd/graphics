import io
import numpy as np
import scipy.io.wavfile as wav
import librosa
import plotly.graph_objects as go
import streamlit as st
from streamlit_mic_recorder import mic_recorder

# ==========================================
# ⚙️ CONFIGURATION & PAGE SETUP
# ==========================================
st.set_page_config(
    page_title="Voice Parameter Analyzer", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ==========================================
# 🧠 1. DSP AUDIO ENGINE CLASS
# ==========================================
class VoiceParameterEngine:
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate

    def load_and_normalize(self, audio_bytes):
        """Step 1: Raw Bytes to Normalized Mono Array [-1.0, 1.0]"""
        audio_file = io.BytesIO(audio_bytes)
        sr, audio_data = wav.read(audio_file)
        
        # Stereo to Mono Conversion
        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)
            
        # Float32 Normalization for Librosa DSP
        y_float = audio_data.astype(np.float32) / 32768.0
        return y_float, sr

    def extract_frequency_spectrum(self, y_signal, sr):
        """Step 2: Frequency Analysis via Fast Fourier Transform (FFT)"""
        fft_vals = np.abs(np.fft.rfft(y_signal))
        fft_freqs = np.fft.rfftfreq(len(y_signal), 1.0 / sr)
        peak_freq = float(fft_freqs[np.argmax(fft_vals)])
        
        return {
            "freqs": fft_freqs,
            "magnitudes": fft_vals,
            "peak_freq_hz": peak_freq
        }

    def extract_pitch_contour(self, y_signal, sr):
        """Step 3: Pitch Tracking via YIN Algorithm ($F_0$)"""
        pitch_f0 = librosa.yin(y_signal, fmin=65, fmax=500, sr=sr)
        pitch_f0_clean = np.where(pitch_f0 < 65, np.nan, pitch_f0)
        
        mean_pitch = float(np.nanmean(pitch_f0_clean)) if not np.all(np.isnan(pitch_f0_clean)) else 0.0
        max_pitch = float(np.nanmax(pitch_f0_clean)) if not np.all(np.isnan(pitch_f0_clean)) else 0.0
        
        time_stamps = np.linspace(0, len(y_signal) / sr, len(pitch_f0_clean))
        
        return {
            "time_stamps": time_stamps,
            "pitch_track": pitch_f0_clean,
            "mean_pitch_hz": mean_pitch,
            "max_pitch_hz": max_pitch
        }

    def extract_noise_and_snr(self, y_signal, sr):
        """Step 4: RMS Energy & Noise Floor / SNR Calculation"""
        hop_length = 512
        rms_energy = librosa.feature.rms(y=y_signal, hop_length=hop_length)[0]
        rms_db = 20 * np.log10(rms_energy + 1e-6)
        
        noise_floor_db = float(np.percentile(rms_db, 10))
        peak_signal_db = float(np.max(rms_db))
        snr_db = peak_signal_db - noise_floor_db
        
        time_frames = librosa.frames_to_time(range(len(rms_energy)), sr=sr, hop_length=hop_length)
        
        return {
            "time_frames": time_frames,
            "rms_db": rms_db,
            "noise_floor_db": noise_floor_db,
            "snr_db": snr_db
        }

# ==========================================
# 📈 2. VISUALIZATION FUNCTIONS
# ==========================================
def create_waveform_graph(y_signal, sr):
    """NEW: Raw Audio Waveform (Time Domain)"""
    time_stamps = np.linspace(0, len(y_signal) / sr, len(y_signal))
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=time_stamps,
        y=y_signal,
        mode='lines',
        name='Amplitude',
        line=dict(color='#74c7ec', width=1)
    ))
    fig.update_layout(
        title="🌊 Audio Waveform (Time Domain Signal)",
        xaxis_title="Time (Seconds)",
        yaxis_title="Amplitude",
        template="plotly_dark",
        height=280,
        margin=dict(l=40, r=40, t=40, b=40)
    )
    return fig

def create_frequency_graph(freq_data):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=freq_data["freqs"],
        y=freq_data["magnitudes"],
        mode='lines',
        name='Magnitude',
        line=dict(color='#89b4fa', width=1.5)
    ))
    fig.update_layout(
        title="1️⃣ Frequency Spectrum (FFT)",
        xaxis_title="Frequency (Hz)",
        yaxis_title="Magnitude",
        xaxis=dict(range=[0, 4000]),
        template="plotly_dark",
        height=320
    )
    return fig

def create_pitch_graph(pitch_data):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=pitch_data["time_stamps"],
        y=pitch_data["pitch_track"],
        mode='lines+markers',
        name='Pitch (Hz)',
        line=dict(color='#a6e3a1', width=2),
        marker=dict(size=4)
    ))
    fig.update_layout(
        title="2️⃣ Pitch Variation Over Time (F0 Contour)",
        xaxis_title="Time (Seconds)",
        yaxis_title="Pitch (Hz)",
        template="plotly_dark",
        height=320
    )
    return fig

def create_noise_graph(noise_data):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=noise_data["time_frames"],
        y=noise_data["rms_db"],
        mode='lines',
        name='Signal Power (dB)',
        line=dict(color='#f38ba8', width=2)
    ))
    fig.add_hline(
        y=noise_data["noise_floor_db"],
        line_dash="dash",
        line_color="#f9e2af",
        annotation_text=f"Noise Floor ({noise_data['noise_floor_db']:.1f} dB)",
        annotation_position="bottom right"
    )
    fig.update_layout(
        title="3️⃣ Signal Energy & Background Noise Floor (dB)",
        xaxis_title="Time (Seconds)",
        yaxis_title="Energy Power (dB)",
        template="plotly_dark",
        height=320
    )
    return fig

# ==========================================
# 🖥️ 3. STREAMLIT APPLICATION DASHBOARD
# ==========================================
st.title("🎙️ Voice Diagnostics & Parameter Extraction Engine")
st.caption("Real-Time Signal Processing: Frequency Spectrum, Pitch Contour, and Noise/SNR Analytics")

# Initialize Engine Instance
engine = VoiceParameterEngine()

st.subheader("1. Record Audio Signal")
audio_output = mic_recorder(
    start_prompt="🎤 Start Speaking",
    stop_prompt="⏹️ Stop & Process",
    just_once=True,
    use_container_width=True,
    format="wav",
    key="parameter_recorder"
)

if audio_output and audio_output.get("bytes"):
    audio_bytes = audio_output.get("bytes")
    
    with st.spinner("⚙️ Extracting Frequency, Pitch, and Noise Parameters..."):
        # 1. Preprocess audio
        y_signal, sr = engine.load_and_normalize(audio_bytes)
        
        # 2. Extract core features
        freq_res = engine.extract_frequency_spectrum(y_signal, sr)
        pitch_res = engine.extract_pitch_contour(y_signal, sr)
        noise_res = engine.extract_noise_and_snr(y_signal, sr)

    st.success("✅ Analysis Complete!")
    
    # NEW: Audio Player + Waveform Graph Right Below Recording
    st.audio(audio_bytes, format="audio/wav")
    fig_wave = create_waveform_graph(y_signal, sr)
    st.plotly_chart(fig_wave, use_container_width=True)
    
    st.markdown("---")
    
    # METRICS DISPLAY
    st.subheader("📊 Key Voice Parameters (Extracted)")
    col1, col2, col3, col4 = st.columns(4)
    
    col1.metric("Average Pitch", f"{pitch_res['mean_pitch_hz']:.1f} Hz")
    col2.metric("Peak Pitch", f"{pitch_res['max_pitch_hz']:.1f} Hz")
    col3.metric("Peak Frequency", f"{freq_res['peak_freq_hz']:.1f} Hz")
    col4.metric("Signal-to-Noise Ratio", f"{noise_res['snr_db']:.1f} dB")
    
    st.markdown("---")
    
    # VISUAL GRAPHS DISPLAY
    st.subheader("📈 Signal Visualizations")
    
    fig_freq = create_frequency_graph(freq_res)
    st.plotly_chart(fig_freq, use_container_width=True)
    
    fig_pitch = create_pitch_graph(pitch_res)
    st.plotly_chart(fig_pitch, use_container_width=True)
    
    fig_noise = create_noise_graph(noise_res)
    st.plotly_chart(fig_noise, use_container_width=True)
