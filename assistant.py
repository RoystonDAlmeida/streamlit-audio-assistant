import os
import streamlit as st
import pyaudio
import wave         # For reading and writing audio WAV files
import assemblyai as aai
from langchain_groq import ChatGroq
from gtts import gTTS
import tempfile
from dotenv import load_dotenv

load_dotenv()

# Load AssemblyAI model for transcription
aai.settings.api_key = os.getenv("ASSEMBLY_API_KEY")    # Set your API key in environment variables
transcriber = aai.Transcriber()

# Initialize ChatGroq with your API key
chat_model = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),  # Set your API key in environment variables
    model="mixtral-8x7b-32768"  # Example model name; replace as needed
)

def record_audio(filename):
    """
    @Args:- filename:- str object that contains the name of the file to be recorded in.
    @Return:-
    @Description:-
                This method records input stream in chunks and writes it to filename.
    """

    # Input stream configuration parameters
    chunk = 1024
    sample_format = pyaudio.paInt16
    channels = 1
    fs = 44100
    seconds = 5

    p = pyaudio.PyAudio()
    stream = p.open(format=sample_format, channels=channels,
                    rate=fs, input= True, frames_per_buffer=chunk)  # Input stream

    st.write("Recording...")
    frames = []

    for _ in range(0, int(fs / chunk * seconds)):
        # Reading input stream in chunks
        data = stream.read(chunk)
        frames.append(data)

    st.write("Finished recording.")

    # Closing the input stream
    stream.stop_stream()
    stream.close()
    p.terminate()

    # Writing the 'frames' to the 'filename.wav'
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(p.get_sample_size(sample_format))
        wf.setframerate(fs)
        wf.writeframes(b''.join(frames))


def transcribe_audio(filename):
    """
    @Args:- filename:- str object that contains output filename
    @Return:- result.text:- str object that contains text extracted from 'filename.wav'
    @Description:-
                This method transcribes text from 'filename.wav' and returns the text.
                The extraction of text is performed using AssemblyAI transcriber(defined above)
    """

    result = transcriber.transcribe(filename)
    return result.text


def get_response_from_chatgroq(user_input):
    """
    @Args:- user_input:- str object that contains the text extracted.
    @Return:- response.content:- str object that contains the response from the LLM model.
    @Description:-
                This method invokes the model(configuration defined above) with a prompt(extracted text),
                and returns the response.
    """

    prompt_template = f"You are a helpful assistant. Respond to: {user_input}"
    llm_response = chat_model.invoke(prompt_template)
    return llm_response.content


def text_to_speech(text):
    """
    @Args:- text:- str object that contains the response text of LLM model.
    @Returns:- .mp3 audio file recording that can be played.
    @Description:-
                This method returns the response in an audio .mp3 file
    """

    tts = gTTS(text=text, lang='en')   # An interface to Text-to-Speech API
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tts.save(f"{tmp.name}.mp3")
        return f"{tmp.name}.mp3"


# Streamlit UI setup
st.title("Personalized Voice Assistant")

if st.button("Record Audio", key="record_audio"):
    audio_file_path = "output.wav"      # Sample output file name
    record_audio(audio_file_path)

    # Transcribe(Extract text from speech)
    transcription = transcribe_audio(audio_file_path)
    st.write("Transcription:", transcription)

    if transcription:
        # If transcription is non-empty, pass the extracted text to the LLM model
        response = get_response_from_chatgroq(transcription)
        st.write("Assistant Response:", response)

        # Convert the response text to speech
        audio_response_path = text_to_speech(response)
        st.audio(audio_response_path)  # Play the audio response
