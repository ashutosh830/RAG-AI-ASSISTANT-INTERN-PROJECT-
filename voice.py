from streamlit_mic_recorder import mic_recorder
import speech_recognition as sr
import tempfile
from gtts import gTTS
import streamlit as st


def get_voice_input():

    voice_text = ""

    audio = mic_recorder(
        start_prompt="🎙️ Start Recording",
        stop_prompt="⏹️ Stop Recording",
        key="voice_recorder"
    )

    if audio:
        st.write(audio.keys())
        st.write(audio)

        temp_audio = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".wav"
        )

        temp_audio.write(audio["bytes"])
        temp_audio.close()

        recognizer = sr.Recognizer()

        try:

            with sr.AudioFile(
                temp_audio.name
            ) as source:

                audio_data = recognizer.record(
                    source
                )

            voice_text = recognizer.recognize_google(
                audio_data
            )

            st.success(
                f"Recognized: {voice_text}"
            )

        except Exception as e:

            st.error(
                f"Voice Error: {e}"
            )

    return voice_text


def speak_answer(answer):

    try:

        tts = gTTS(
            text=answer,
            lang="en"
        )

        audio_file = "response.mp3"

        tts.save(audio_file)

        st.audio(
            audio_file,
            format="audio/mp3"
        )

    except Exception as e:

        st.error(
            f"TTS Error: {e}"
        )