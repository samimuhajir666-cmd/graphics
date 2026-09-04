import io
import numpy as np
import scipy.io.wavfile as wav
import librosa
import plotly.graph_objects as go
import streamlit as import io
import numpy as np
import scipy.io.wavfile as wav
import librosa
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

from streamlit_mic_recorder import mic_recorder


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Voice Diagnostics & Parameter Extraction Engine",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
    <style>
        .recorder-box {
            background-color: #11111b;
            border: 2px solid #313244;
            border-radius: 14px;
            padding: 18px;
            margin-bottom: 20px;
        }

        .status-box {
            padding: 12px;
            border-radius: 10px;
            margin-top: 10px;
        }
    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# AUDIO ENGINE
# =========================================================

class VoiceParameterEngine:

    def __init__(self, target_sample_rate=16000):
        self.target_sample_rate = target_sample_rate


    # -----------------------------------------------------
    # LOAD WAV
    # -----------------------------------------------------

    def load_and_normalize(self, audio_bytes):

        if not audio_bytes:
            raise ValueError("Audio data is empty.")

        try:
            audio_file = io.BytesIO(audio_bytes)

            sr, audio_data = wav.read(audio_file)

        except Exception as e:
            raise ValueError(
                f"Could not read WAV audio: {str(e)}"
            )

        if audio_data is None or len(audio_data) == 0:
            raise ValueError("WAV file contains no audio samples.")

        original_dtype = audio_data.dtype

        # -------------------------------------------------
        # Stereo -> Mono
        # -------------------------------------------------

        if audio_data.ndim > 1:
            audio_data = np.mean(
                audio_data.astype(np.float32),
                axis=1
            )

        # -------------------------------------------------
        # Normalize different WAV formats
        # -------------------------------------------------

        if np.issubdtype(original_dtype, np.integer):

            info = np.iinfo(original_dtype)

            max_value = max(
                abs(info.min),
                abs(info.max)
            )

            y = audio_data.astype(np.float32) / max_value

        elif np.issubdtype(original_dtype, np.floating):

            y = audio_data.astype(np.float32)

            max_abs = np.max(np.abs(y))

            if max_abs > 1:
                y = y / max_abs

        else:
            raise ValueError(
                f"Unsupported audio format: {original_dtype}"
            )

        # Remove DC offset
        y = y - np.mean(y)

        # Prevent invalid values
        y = np.nan_to_num(
            y,
            nan=0.0,
            posinf=0.0,
            neginf=0.0
        )

        return y, int(sr)


    # -----------------------------------------------------
    # RESAMPLE
    # -----------------------------------------------------

    def resample_audio(self, y, sr):

        if sr == self.target_sample_rate:
            return y, sr

        y_resampled = librosa.resample(
            y=y,
            orig_sr=sr,
            target_sr=self.target_sample_rate
        )

        return y_resampled.astype(np.float32), self.target_sample_rate


    # -----------------------------------------------------
    # BASIC AUDIO INFORMATION
    # -----------------------------------------------------

    def get_audio_info(self, y, sr):

        duration = len(y) / sr

        peak = float(np.max(np.abs(y)))

        rms = float(
            np.sqrt(np.mean(np.square(y)))
        )

        return {
            "duration": duration,
            "sample_rate": sr,
            "samples": len(y),
            "peak": peak,
            "rms": rms
        }


    # =====================================================
    # FREQUENCY SPECTRUM
    # =====================================================

    def extract_frequency_spectrum(self, y, sr):

        if len(y) < 2:
            raise ValueError("Audio is too short for FFT.")

        # Windowing improves FFT analysis
        window = np.hanning(len(y))

        signal = y * window

        fft_values = np.fft.rfft(signal)

        magnitudes = np.abs(fft_values)

        frequencies = np.fft.rfftfreq(
            len(signal),
            d=1.0 / sr
        )

        # Ignore DC component
        if len(magnitudes) > 1:
            magnitudes[0] = 0

        peak_index = np.argmax(magnitudes)

        peak_frequency = float(
            frequencies[peak_index]
        )

        return {
            "freqs": frequencies,
            "magnitudes": magnitudes,
            "peak_freq_hz": peak_frequency
        }


    # =====================================================
    # PITCH / F0
    # =====================================================

    def extract_pitch_contour(self, y, sr):

        if len(y) < sr * 0.1:
            raise ValueError(
                "Audio is too short for pitch analysis."
            )

        try:

            pitch = librosa.yin(
                y,
                fmin=50,
                fmax=500,
                sr=sr,
                frame_length=2048,
                hop_length=256
            )

        except Exception as e:

            raise ValueError(
                f"Pitch extraction failed: {str(e)}"
            )

        # librosa YIN can produce values outside
        # useful speech range / unvoiced regions.

        pitch_clean = pitch.astype(float)

        pitch_clean[
            (pitch_clean < 50) |
            (pitch_clean > 500)
        ] = np.nan

        # Proper time axis
        time_stamps = librosa.times_like(
            pitch_clean,
            sr=sr,
            hop_length=256
        )

        valid_pitch = pitch_clean[
            ~np.isnan(pitch_clean)
        ]

        if len(valid_pitch) > 0:

            mean_pitch = float(
                np.median(valid_pitch)
            )

            max_pitch = float(
                np.max(valid_pitch)
            )

            min_pitch = float(
                np.min(valid_pitch)
            )

        else:

            mean_pitch = 0.0
            max_pitch = 0.0
            min_pitch = 0.0

        return {
            "time_stamps": time_stamps,
            "pitch_track": pitch_clean,
            "mean_pitch_hz": mean_pitch,
            "max_pitch_hz": max_pitch,
            "min_pitch_hz": min_pitch
        }


    # =====================================================
    # RMS / dBFS / SNR
    # =====================================================

    def extract_noise_and_snr(self, y, sr):

        hop_length = 256

        rms_energy = librosa.feature.rms(
            y=y,
            frame_length=1024,
            hop_length=hop_length
        )[0]

        # RMS -> dBFS
        rms_dbfs = librosa.amplitude_to_db(
            rms_energy,
            ref=1.0,
            top_db=80
        )

        # Estimate quiet/background region
        noise_floor_dbfs = float(
            np.percentile(
                rms_dbfs,
                10
            )
        )

        signal_level_dbfs = float(
            np.percentile(
                rms_dbfs,
                90
            )
        )

        snr_estimate = (
            signal_level_dbfs -
            noise_floor_dbfs
        )

        time_frames = librosa.frames_to_time(
            np.arange(len(rms_energy)),
            sr=sr,
            hop_length=hop_length
        )

        return {

            "time_frames": time_frames,

            "rms_energy": rms_energy,

            "rms_dbfs": rms_dbfs,

            "noise_floor_dbfs":
                noise_floor_dbfs,

            "signal_level_dbfs":
                signal_level_dbfs,

            "snr_db":
                float(snr_estimate)
        }


    # =====================================================
    # SPEECH / SILENCE RATIO
    # =====================================================

    def calculate_voice_ratio(self, y, sr):

        frame_length = 1024
        hop_length = 256

        rms = librosa.feature.rms(
            y=y,
            frame_length=frame_length,
            hop_length=hop_length
        )[0]

        rms_db = librosa.amplitude_to_db(
            rms,
            ref=1.0,
            top_db=80
        )

        # Adaptive threshold
        noise_level = np.percentile(
            rms_db,
            10
        )

        threshold = noise_level + 8

        voice_frames = rms_db > threshold

        voice_ratio = (
            np.sum(voice_frames) /
            len(voice_frames)
        ) * 100

        silence_ratio = 100 - voice_ratio

        return {
            "voice_ratio": float(voice_ratio),
            "silence_ratio": float(silence_ratio),
            "threshold_dbfs": float(threshold)
        }


# =========================================================
# WAVEFORM VISUALIZER
# =========================================================

def render_waveform(y_signal):

    num_bars = 80

    chunks = np.array_split(
        np.abs(y_signal),
        num_bars
    )

    bar_heights = []

    for chunk in chunks:

        if len(chunk) == 0:
            value = 0
        else:
            value = np.sqrt(
                np.mean(
                    np.square(chunk)
                )
            )

        bar_heights.append(value)

    max_height = max(
        bar_heights
    ) if max(bar_heights) > 0 else 1

    bars_html = ""

    for value in bar_heights:

        height = max(
            4,
            int(
                (value / max_height) * 65
            )
        )

        bars_html += f"""
        <div style="
            width:4px;
            height:{height}px;
            background:linear-gradient(
                180deg,
                #89b4fa,
                #f38ba8
            );
            border-radius:3px;
            margin:0 2px;
        "></div>
        """

    html = f"""
    <div style="
        background:#181825;
        padding:20px;
        border-radius:16px;
        border:1px solid #313244;
    ">

        <div style="
            color:#a6adc8;
            font-family:sans-serif;
            font-size:12px;
            font-weight:600;
            margin-bottom:12px;
        ">
            🎙️ AUDIO WAVEFORM
        </div>

        <div style="
            display:flex;
            align-items:center;
            justify-content:center;
            height:90px;
            background:#11111b;
            border-radius:12px;
            padding:0 10px;
            overflow:hidden;
        ">

            {bars_html}

        </div>

    </div>
    """

    components.html(
        html,
        height=155
    )


# =========================================================
# FREQUENCY GRAPH
# =========================================================

def create_frequency_graph(freq_data):

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=freq_data["freqs"],
            y=freq_data["magnitudes"],
            mode="lines",
            name="Magnitude",
            line=dict(
                color="#89b4fa",
                width=1.5
            )
        )
    )

    fig.update_layout(

        title="1️⃣ Frequency Spectrum — FFT",

        xaxis_title="Frequency (Hz)",

        yaxis_title="Magnitude",

        xaxis=dict(
            range=[0, 4000]
        ),

        template="plotly_dark",

        height=350
    )

    return fig


# =========================================================
# PITCH GRAPH
# =========================================================

def create_pitch_graph(pitch_data):

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=pitch_data["time_stamps"],
            y=pitch_data["pitch_track"],
            mode="lines",
            name="F0 Pitch",

            connectgaps=False,

            line=dict(
                color="#a6e3a1",
                width=2
            )
        )
    )

    fig.update_layout(

        title="2️⃣ Pitch / F0 Contour Over Time",

        xaxis_title="Time (Seconds)",

        yaxis_title="Pitch F0 (Hz)",

        yaxis=dict(
            range=[50, 500]
        ),

        template="plotly_dark",

        height=350
    )

    return fig


# =========================================================
# RMS / DBFS GRAPH
# =========================================================

def create_noise_graph(noise_data):

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(

            x=noise_data["time_frames"],

            y=noise_data["rms_dbfs"],

            mode="lines",

            name="RMS dBFS",

            line=dict(
                color="#f38ba8",
                width=2
            )
        )
    )

    fig.add_hline(

        y=noise_data[
            "noise_floor_dbfs"
        ],

        line_dash="dash",

        line_color="#f9e2af",

        annotation_text=(
            f"Estimated Noise Floor "
            f"({noise_data['noise_floor_dbfs']:.1f} dBFS)"
        )
    )

    fig.update_layout(

        title="3️⃣ RMS Energy / dBFS Over Time",

        xaxis_title="Time (Seconds)",

        yaxis_title="Level (dBFS)",

        template="plotly_dark",

        height=350
    )

    return fig


# =========================================================
# MAIN DASHBOARD
# =========================================================

st.title(
    "🎙️ Voice Diagnostics & Parameter Extraction Engine"
)

st.caption(
    "Frequency Spectrum • Pitch/F0 • RMS Energy • "
    "dBFS • SNR • Voice/Silence Analysis"
)


# =========================================================
# ENGINE
# =========================================================

engine = VoiceParameterEngine(
    target_sample_rate=16000
)


# =========================================================
# INPUT SECTION
# =========================================================

st.header("1. 🎤 Record Your Voice")

st.info(
    "Press Start Recording, allow microphone permission, "
    "speak for a few seconds, then press Stop & Process."
)


# =========================================================
# MICROPHONE RECORDER
# =========================================================

try:

    audio_output = mic_recorder(

        start_prompt="🎤 Start Recording",

        stop_prompt="⏹️ Stop & Process",

        just_once=True,

        use_container_width=True,

        format="wav",

        key="voice_parameter_recorder"
    )

except Exception as e:

    audio_output = None

    st.error(
        f"Microphone component error: {e}"
    )


# =========================================================
# GET AUDIO BYTES
# =========================================================

audio_bytes = None


if audio_output:

    if isinstance(audio_output, dict):

        audio_bytes = audio_output.get(
            "bytes"
        )

    else:

        st.warning(
            "Recorder returned an unexpected format."
        )


# =========================================================
# FALLBACK UPLOAD
# =========================================================

st.markdown("---")

st.subheader(
    "📁 Or Upload Audio"
)

uploaded_file = st.file_uploader(
    "Upload a WAV file for testing",
    type=["wav"],
    key="audio_upload"
)


if uploaded_file is not None:

    audio_bytes = uploaded_file.getvalue()

    st.success(
        "Uploaded audio selected for analysis."
    )


# =========================================================
# ANALYSIS
# =========================================================

if audio_bytes:

    try:

        with st.spinner(
            "⚙️ Processing audio signal..."
        ):

            # ---------------------------------------------
            # LOAD
            # ---------------------------------------------

            y_signal, original_sr = (
                engine.load_and_normalize(
                    audio_bytes
                )
            )

            # ---------------------------------------------
            # RESAMPLE
            # ---------------------------------------------

            y_signal, sr = (
                engine.resample_audio(
                    y_signal,
                    original_sr
                )
            )

            # ---------------------------------------------
            # AUDIO INFO
            # ---------------------------------------------

            audio_info = (
                engine.get_audio_info(
                    y_signal,
                    sr
                )
            )

            # ---------------------------------------------
            # FREQUENCY
            # ---------------------------------------------

            freq_res = (
                engine.extract_frequency_spectrum(
                    y_signal,
                    sr
                )
            )

            # ---------------------------------------------
            # PITCH
            # ---------------------------------------------

            pitch_res = (
                engine.extract_pitch_contour(
                    y_signal,
                    sr
                )
            )

            # ---------------------------------------------
            # NOISE / RMS / SNR
            # ---------------------------------------------

            noise_res = (
                engine.extract_noise_and_snr(
                    y_signal,
                    sr
                )
            )

            # ---------------------------------------------
            # VOICE RATIO
            # ---------------------------------------------

            ratio_res = (
                engine.calculate_voice_ratio(
                    y_signal,
                    sr
                )
            )

        st.success(
            "✅ Audio analysis completed successfully!"
        )


        # =================================================
        # AUDIO PLAYER
        # =================================================

        st.subheader(
            "🔊 Recorded Audio"
        )

        st.audio(
            audio_bytes,
            format="audio/wav"
        )


        # =================================================
        # WAVEFORM
        # =================================================

        render_waveform(
            y_signal
        )


        # =================================================
        # BASIC AUDIO INFO
        # =================================================

        st.markdown("---")

        st.subheader(
            "ℹ️ Audio Information"
        )

        info1, info2, info3, info4 = st.columns(4)

        info1.metric(
            "Duration",
            f"{audio_info['duration']:.2f} sec"
        )

        info2.metric(
            "Sample Rate",
            f"{audio_info['sample_rate']} Hz"
        )

        info3.metric(
            "Samples",
            f"{audio_info['samples']:,}"
        )

        info4.metric(
            "Peak Amplitude",
            f"{audio_info['peak']:.3f}"
        )


        # =================================================
        # MAIN PARAMETERS
        # =================================================

        st.markdown("---")

        st.subheader(
            "📊 Key Voice Parameters"
        )

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Average F0",
            f"{pitch_res['mean_pitch_hz']:.1f} Hz"
        )

        col2.metric(
            "Peak F0",
            f"{pitch_res['max_pitch_hz']:.1f} Hz"
        )

        col3.metric(
            "Peak Spectrum Frequency",
            f"{freq_res['peak_freq_hz']:.1f} Hz"
        )

        col4.metric(
            "Estimated SNR",
            f"{noise_res['snr_db']:.1f} dB"
        )


        # =================================================
        # VOICE / SILENCE RATIO
        # =================================================

        st.markdown("---")

        st.subheader(
            "🗣️ Recording Composition"
        )

        r1, r2, r3 = st.columns(3)

        r1.metric(
            "Actual Speech / Voice",
            f"{ratio_res['voice_ratio']:.1f}%"
        )

        r2.metric(
            "Silence / Low Energy",
            f"{ratio_res['silence_ratio']:.1f}%"
        )

        r3.metric(
            "Total Recording",
            "100%"
        )


        # =================================================
        # VISUALIZATIONS
        # =================================================

        st.markdown("---")

        st.header(
            "📈 Signal Visualizations"
        )


        # Frequency
        st.plotly_chart(
            create_frequency_graph(
                freq_res
            ),
            use_container_width=True
        )


        # Pitch
        st.plotly_chart(
            create_pitch_graph(
                pitch_res
            ),
            use_container_width=True
        )


        # RMS
        st.plotly_chart(
            create_noise_graph(
                noise_res
            ),
            use_container_width=True
        )


        # =================================================
        # DETAILED VALUES
        # =================================================

        st.markdown("---")

        st.subheader(
            "🔬 Detailed Measurements"
        )

        d1, d2 = st.columns(2)

        with d1:

            st.write(
                "**Pitch / F0**"
            )

            st.write(
                f"Minimum F0: "
                f"{pitch_res['min_pitch_hz']:.1f} Hz"
            )

            st.write(
                f"Median F0: "
                f"{pitch_res['mean_pitch_hz']:.1f} Hz"
            )

            st.write(
                f"Maximum F0: "
                f"{pitch_res['max_pitch_hz']:.1f} Hz"
            )

        with d2:

            st.write(
                "**Signal Level**"
            )

            st.write(
                f"Signal Level: "
                f"{noise_res['signal_level_dbfs']:.1f} dBFS"
            )

            st.write(
                f"Noise Floor: "
                f"{noise_res['noise_floor_dbfs']:.1f} dBFS"
            )

            st.write(
                f"Estimated SNR: "
                f"{noise_res['snr_db']:.1f} dB"
            )


        # =================================================
        # TECHNICAL NOTE
        # =================================================

        st.markdown("---")

        st.caption(
            "Note: dBFS is a digital audio level relative to "
            "full scale. The SNR shown here is an estimated "
            "relative SNR based on the recording's quiet and "
            "loud RMS regions; it is not a calibrated acoustic "
            "SPL measurement."
        )


    # =====================================================
    # ERROR HANDLING
    # =====================================================

    except Exception as e:

        st.error(
            "❌ Audio processing failed."
        )

        st.exception(e)

        st.info(
            "Try recording again or upload a valid WAV file."
        )


else:

    st.info(
        "🎙️ Waiting for microphone recording..."
    )

    st.write(
        "If the microphone button appears but does not "
        "start recording, check your browser's microphone "
        "permission."
    )st
import streamlit.components.v1 as components
from streamlit_mic_recorder import mic_recorder

# ==========================================
# ⚙️ CONFIGURATION & PAGE SETUP
# ==========================================
st.set_page_config(
    page_title="Voice Parameter Analyzer", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom Styling for Phone Recorder Visualizer Container (FIXED TYPO HERE)
st.markdown("""
<style>
    .recorder-box {
        background-color: #11111b;
        border: 2px solid #313244;
        border-radius: 12px;
        padding: 15px;
        text-align: center;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

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
# 📱 PHONE VOICE RECORDER WAVEFORM COMPONENT
# ==========================================
def render_phone_recorder_waveform(y_signal, num_bars=70):
    """Generates a Smartphone Voice Memos Style Vertical Bar Waveform"""
    # Downsample audio array into fixed number of bar heights (RMS energy per block)
    chunks = np.array_split(np.abs(y_signal), num_bars)
    bar_heights = [float(np.mean(chunk)) * 250 for chunk in chunks]
    
    # Normalize bar heights (min 4px, max 65px)
    max_h = max(bar_heights) if max(bar_heights) > 0 else 1
    normalized_bars = [max(4, int((h / max_h) * 65)) for h in bar_heights]

    # HTML/JS Canvas UI matching Smartphone Recorder Layout
    bars_html = "".join([
        f'<div style="width: 4px; height: {h}px; background: linear-gradient(180deg, #89b4fa, #f38ba8); border-radius: 3px; margin: 0 2px; transition: height 0.2s ease;"></div>'
        for h in normalized_bars
    ])
    
    html_code = f"""
    <div style="background-color: #181825; padding: 20px; border-radius: 16px; border: 1px solid #313244; box-shadow: 0 4px 20px rgba(0,0,0,0.5);">
        <div style="color: #a6adc8; font-family: sans-serif; font-size: 12px; letter-spacing: 1px; margin-bottom: 12px; font-weight: 600;">
            🎙️ SMARTPHONE VOICE RECORDER WAVEFORM
        </div>
        <div style="display: flex; align-items: center; justify-content: center; height: 90px; background-color: #11111b; border-radius: 12px; padding: 0 10px; overflow-x: auto;">
            {bars_html}
        </div>
    </div>
    """
    components.html(html_code, height=155)

# ==========================================
# 📈 2. VISUALIZATION FUNCTIONS (PLOTLY)
# ==========================================
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
        height=300
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
        height=300
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
        height=300
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
    
    # 📱 DISPLAY PHONE RECORDER STYLE WAVEFORM BARS & AUDIO PLAYER
    st.audio(audio_bytes, format="audio/wav")
    render_phone_recorder_waveform(y_signal)
    
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
