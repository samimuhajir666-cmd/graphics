import io
import numpy as np
import scipy.io.wavfile as wav
import librosa
import plotly.graph_objects as go
import streamlit as st
from streamlit_mic_recorder import mic_recorder


# =========================================================
# PAGE SETUP
# =========================================================

st.set_page_config(
    page_title="Voice Parameter Analyzer",
    page_icon="🎙️",
    layout="wide"
)


# =========================================================
# VOICE PARAMETER ENGINE
# =========================================================

class VoiceParameterEngine:

    def load_audio(self, audio_bytes):
        """Load WAV bytes and convert to normalized mono audio."""

        if not audio_bytes:
            raise ValueError("No audio data received.")

        audio_file = io.BytesIO(audio_bytes)

        sr, audio_data = wav.read(audio_file)

        if len(audio_data) == 0:
            raise ValueError("Audio file is empty.")

        original_dtype = audio_data.dtype

        # Stereo -> Mono
        if audio_data.ndim > 1:
            audio_data = np.mean(
                audio_data.astype(np.float32),
                axis=1
            )

        # Integer WAV
        if np.issubdtype(original_dtype, np.integer):

            info = np.iinfo(original_dtype)

            max_value = max(
                abs(info.min),
                abs(info.max)
            )

            y = audio_data.astype(
                np.float32
            ) / max_value

        # Float WAV
        elif np.issubdtype(
            original_dtype,
            np.floating
        ):

            y = audio_data.astype(
                np.float32
            )

            max_value = np.max(
                np.abs(y)
            )

            if max_value > 1:
                y = y / max_value

        else:
            raise ValueError(
                f"Unsupported audio format: {original_dtype}"
            )

        # Remove DC offset
        y = y - np.mean(y)

        y = np.nan_to_num(y)

        return y, int(sr)


    # =====================================================
    # FREQUENCY SPECTRUM
    # =====================================================

    def frequency_spectrum(self, y, sr):

        window = np.hanning(len(y))

        signal = y * window

        fft_values = np.fft.rfft(signal)

        magnitudes = np.abs(fft_values)

        frequencies = np.fft.rfftfreq(
            len(signal),
            1 / sr
        )

        # Ignore DC
        if len(magnitudes) > 1:
            magnitudes[0] = 0

        peak_index = np.argmax(
            magnitudes
        )

        peak_frequency = float(
            frequencies[peak_index]
        )

        return {
            "frequencies": frequencies,
            "magnitudes": magnitudes,
            "peak_frequency": peak_frequency
        }


    # =====================================================
    # PITCH / F0
    # =====================================================

    def pitch_analysis(self, y, sr):

        pitch = librosa.yin(
            y,
            fmin=50,
            fmax=500,
            sr=sr,
            frame_length=2048,
            hop_length=256
        )

        pitch = pitch.astype(float)

        # Remove invalid values
        pitch[
            (pitch < 50) |
            (pitch > 500)
        ] = np.nan

        time = librosa.times_like(
            pitch,
            sr=sr,
            hop_length=256
        )

        valid_pitch = pitch[
            ~np.isnan(pitch)
        ]

        if len(valid_pitch) > 0:

            average_pitch = float(
                np.median(valid_pitch)
            )

            peak_pitch = float(
                np.max(valid_pitch)
            )

        else:

            average_pitch = 0
            peak_pitch = 0

        return {
            "time": time,
            "pitch": pitch,
            "average_pitch": average_pitch,
            "peak_pitch": peak_pitch
        }


    # =====================================================
    # RMS + dBFS + SNR
    # =====================================================

    def loudness_analysis(self, y, sr):

        hop_length = 256

        rms = librosa.feature.rms(
            y=y,
            frame_length=1024,
            hop_length=hop_length
        )[0]

        dbfs = librosa.amplitude_to_db(
            rms,
            ref=1.0,
            top_db=80
        )

        noise_floor = float(
            np.percentile(
                dbfs,
                10
            )
        )

        signal_level = float(
            np.percentile(
                dbfs,
                90
            )
        )

        snr = signal_level - noise_floor

        time = librosa.frames_to_time(
            np.arange(len(rms)),
            sr=sr,
            hop_length=hop_length
        )

        return {
            "time": time,
            "rms": rms,
            "dbfs": dbfs,
            "noise_floor": noise_floor,
            "signal_level": signal_level,
            "snr": float(snr)
        }


    # =====================================================
    # VOICE / SILENCE RATIO
    # =====================================================

    def voice_ratio(self, y, sr):

        rms = librosa.feature.rms(
            y=y,
            frame_length=1024,
            hop_length=256
        )[0]

        dbfs = librosa.amplitude_to_db(
            rms,
            ref=1.0,
            top_db=80
        )

        noise_level = np.percentile(
            dbfs,
            10
        )

        threshold = noise_level + 8

        voice_frames = dbfs > threshold

        voice_percentage = (
            np.sum(voice_frames) /
            len(voice_frames)
        ) * 100

        silence_percentage = (
            100 - voice_percentage
        )

        return (
            float(voice_percentage),
            float(silence_percentage)
        )


# =========================================================
# GRAPH: FREQUENCY
# =========================================================

def frequency_graph(data):

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=data["frequencies"],
            y=data["magnitudes"],
            mode="lines",
            name="Frequency Spectrum"
        )
    )

    fig.update_layout(
        title="Frequency Spectrum (FFT)",
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
# GRAPH: PITCH
# =========================================================

def pitch_graph(data):

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=data["time"],
            y=data["pitch"],
            mode="lines",
            name="F0 Pitch",
            connectgaps=False
        )
    )

    fig.update_layout(
        title="Pitch / F0 Contour",
        xaxis_title="Time (Seconds)",
        yaxis_title="Pitch (Hz)",
        yaxis=dict(
            range=[50, 500]
        ),
        template="plotly_dark",
        height=350
    )

    return fig


# =========================================================
# GRAPH: dBFS
# =========================================================

def loudness_graph(data):

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=data["time"],
            y=data["dbfs"],
            mode="lines",
            name="RMS dBFS"
        )
    )

    fig.add_hline(
        y=data["noise_floor"],
        line_dash="dash",
        annotation_text=(
            f"Noise Floor: "
            f"{data['noise_floor']:.1f} dBFS"
        )
    )

    fig.update_layout(
        title="RMS Energy / dBFS",
        xaxis_title="Time (Seconds)",
        yaxis_title="Level (dBFS)",
        template="plotly_dark",
        height=350
    )

    return fig


# =========================================================
# APP
# =========================================================

st.title(
    "🎙️ Voice Parameter Analyzer"
)

st.write(
    "Record your voice and extract "
    "Pitch, Frequency, RMS, dBFS and SNR."
)


# =========================================================
# ENGINE
# =========================================================

engine = VoiceParameterEngine()


# =========================================================
# MICROPHONE
# =========================================================

st.subheader("🎤 Record Audio")

st.info(
    "Click Start Recording, allow microphone "
    "permission, speak, then click Stop."
)

try:

    audio = mic_recorder(
        start_prompt="🎤 Start Recording",
        stop_prompt="⏹️ Stop Recording",
        just_once=True,
        use_container_width=True,
        format="wav",
        key="voice_recorder"
    )

except Exception as error:

    audio = None

    st.error(
        f"Microphone error: {error}"
    )


# =========================================================
# GET AUDIO
# =========================================================

audio_bytes = None

if audio is not None:

    if isinstance(audio, dict):

        audio_bytes = audio.get("bytes")


# =========================================================
# UPLOAD FALLBACK
# =========================================================

st.subheader("📁 Or Upload WAV")

uploaded = st.file_uploader(
    "Upload a WAV file",
    type=["wav"]
)

if uploaded is not None:

    audio_bytes = uploaded.getvalue()


# =========================================================
# PROCESS AUDIO
# =========================================================

if audio_bytes:

    try:

        with st.spinner(
            "Processing audio..."
        ):

            # Load
            y, sr = engine.load_audio(
                audio_bytes
            )

            # Frequency
            frequency = (
                engine.frequency_spectrum(
                    y,
                    sr
                )
            )

            # Pitch
            pitch = (
                engine.pitch_analysis(
                    y,
                    sr
                )
            )

            # Loudness
            loudness = (
                engine.loudness_analysis(
                    y,
                    sr
                )
            )

            # Voice ratio
            voice, silence = (
                engine.voice_ratio(
                    y,
                    sr
                )
            )

        st.success(
            "✅ Analysis Complete!"
        )


        # -------------------------------------------------
        # AUDIO PLAYER
        # -------------------------------------------------

        st.subheader(
            "🔊 Recorded Audio"
        )

        st.audio(
            audio_bytes,
            format="audio/wav"
        )


        # -------------------------------------------------
        # PARAMETERS
        # -------------------------------------------------

        st.subheader(
            "📊 Voice Parameters"
        )

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "Average Pitch",
            f"{pitch['average_pitch']:.1f} Hz"
        )

        c2.metric(
            "Peak Pitch",
            f"{pitch['peak_pitch']:.1f} Hz"
        )

        c3.metric(
            "Peak Frequency",
            f"{frequency['peak_frequency']:.1f} Hz"
        )

        c4.metric(
            "Estimated SNR",
            f"{loudness['snr']:.1f} dB"
        )


        # -------------------------------------------------
        # VOICE RATIO
        # -------------------------------------------------

        st.subheader(
            "🗣️ Recording Composition"
        )

        r1, r2, r3 = st.columns(3)

        r1.metric(
            "Actual Voice",
            f"{voice:.1f}%"
        )

        r2.metric(
            "Silence / Low Energy",
            f"{silence:.1f}%"
        )

        r3.metric(
            "Total",
            "100%"
        )


        # -------------------------------------------------
        # GRAPHS
        # -------------------------------------------------

        st.subheader(
            "📈 Analysis Graphs"
        )

        st.plotly_chart(
            frequency_graph(frequency),
            use_container_width=True
        )

        st.plotly_chart(
            pitch_graph(pitch),
            use_container_width=True
        )

        st.plotly_chart(
            loudness_graph(loudness),
            use_container_width=True
        )


        # -------------------------------------------------
        # TECHNICAL INFORMATION
        # -------------------------------------------------

        st.subheader(
            "🔬 Technical Information"
        )

        duration = len(y) / sr

        t1, t2, t3 = st.columns(3)

        t1.metric(
            "Duration",
            f"{duration:.2f} sec"
        )

        t2.metric(
            "Sample Rate",
            f"{sr} Hz"
        )

        t3.metric(
            "Samples",
            f"{len(y):,}"
        )

        st.caption(
            "dBFS represents digital signal level relative "
            "to full scale. SNR is an estimated relative "
            "value based on quiet and loud regions."
        )


    except Exception as error:

        st.error(
            "❌ Audio processing failed."
        )

        st.exception(error)


else:

    st.warning(
        "🎙️ No recording received yet. "
        "Start the microphone or upload a WAV file."
    )
