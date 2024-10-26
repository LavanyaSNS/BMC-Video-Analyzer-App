#%%writefile bmc_video_analysis.py
import yt_dlp
import subprocess
import os
import whisper
from pydub import AudioSegment
import librosa
import numpy as np
import webrtcvad
import wave
import contextlib
import cv2
import easyocr
from skimage.metrics import structural_similarity as ssim
import subprocess
import sys
import logging
import requests
import json
import streamlit as st

def get_video_info(video_url):
    ydl_opts = {'skip_download': True}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(video_url, download=False)
        title = info.get('title', 'No title found').lower()
        # Remove the specified substrings
        title = title.replace('sns college of technology', '')
        title = title.replace('bmc video', '')
        title = title.replace('dt project', '')
        title = title.replace('sns institutions', '')
        title = title.replace('sns institution', '')
        title = title.replace('#snsinstitutions', '')
        title = title.replace('#snsdesignthinkers', '')
        title = title.replace('#designthinking', '')
        title = title.replace('-', '')

        # Optionally, strip any extra spaces or separators (e.g., '|')
        title = title.replace('|', '').strip()
        return title

def download_video(video_url, output_path="video.mp4.mkv"):
    ydl_opts = {
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/mp4',
        'outtmpl': output_path,
        'merge_output_format': 'mkv',
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([video_url])
        return output_path
    except Exception as e:
        print(f"Error downloading video: {e}")
        return None

def check_video_quality(video_path):
    command = [
        'ffprobe',
        '-v', 'error',
        '-select_streams', 'v:0',
        '-show_entries', 'stream=width,height',
        '-of', 'csv=p=0',
        video_path
    ]
    try:
        output = subprocess.check_output(command).decode().strip()
        width, height = map(int, output.split(','))

        if height < 480:
            return False, f"Video quality issue: The video resolution is {width}x{height}, but it must be at least 480p."
        return True, f" "

    except subprocess.CalledProcessError as e:
        return False, f"Error checking video quality: {e}"

def extract_audio(video_path, audio_output_path="audio.mp3"):
    # Check if ffmpeg is installed
    if subprocess.call(["which", "ffmpeg"], stdout=subprocess.PIPE, stderr=subprocess.PIPE) != 0:
        print("ffmpeg is not installed. Please install it to extract audio.")
        return None

    # Check if the output file already exists, and if so, remove it to allow overwriting
    if os.path.exists(audio_output_path):
        os.remove(audio_output_path) #already exists. Overwriting the file.")

    # Prepare the ffmpeg command to extract audio
    command = [
        'ffmpeg',
        '-i', video_path,
        '-q:a', '0',
        '-map', 'a',
        audio_output_path
    ]

    # Execute the command
    try:
        print(f"Executing command: {' '.join(command)}")  # Debug: Print the command
        subprocess.run(command, check=True, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
        print(f"Audio extracted successfully: {audio_output_path}")
        return audio_output_path
    except subprocess.CalledProcessError as e:
        print(f"Error extracting audio: {e.stderr.decode()}")  # Print error message from ffmpeg
        return None

# Audio Quality, Tone, Background Noise Checks
def check_audio_quality(audio_path):
    audio = AudioSegment.from_mp3(audio_path)
    bitrate = audio.frame_rate
    channels = audio.channels
    duration = len(audio) / 1000  # Duration in seconds

    if bitrate < 16000:  # Sample rate should be at least 16kHz
        return False, "Audio quality too low (bitrate < 16kHz)."
    if channels != 1:
        # Convert to mono if it has multiple channels
        audio = audio.set_channels(1)
        audio.export(audio_path, format='mp3')  # Overwrite the original audio
        channels = 1  # Update the channels count

    if duration < 180:  # Check if the audio is less than 4 minutes (240 seconds)
        return False, "Video file too short (less than 4 minutes)."

    return True, "Audio quality is acceptable."

def check_audio_tone(audio_path):
    audio_data, sr = librosa.load(audio_path)
    pitches, magnitudes = librosa.piptrack(y=audio_data, sr=sr)

    # Calculate the mean pitch
    pitch_values = [pitches[i][pitches[i] > 0] for i in range(len(pitches))]
    pitch_mean = [np.mean(pitch) if len(pitch) > 0 else 0 for pitch in pitch_values]

    if np.mean(pitch_mean) < 75:  # Too low, possibly not human speech
        return False, "Audio tone may be too low for valid speech."
    return True, "Audio tone is acceptable."


def check_background_noise(audio_path):
    audio_data, sr = librosa.load(audio_path)
    energy = np.sum(audio_data ** 2) / len(audio_data)
    noise_threshold = np.percentile(np.abs(audio_data), 5)
    noise_energy = np.mean(np.abs(audio_data)[np.abs(audio_data) < noise_threshold] ** 2)

    snr = 10 * np.log10(energy / noise_energy)

    if snr < 15:  # Minimum SNR of 15dB
        return False, "Too much background noise."
    return True, "Background noise level is acceptable."


def convert_to_pcm(audio_path):
    """
    Convert audio to 16-bit mono PCM with a sample rate of 16kHz.
    """
    sound = AudioSegment.from_file(audio_path)
    sound = sound.set_channels(1)  # Convert to mono
    sound = sound.set_frame_rate(16000)  # Set sample rate to 16kHz
    pcm_path = "converted_audio.wav"
    sound.export(pcm_path, format="wav")
    return pcm_path


def check_voice_presence(audio_path):
    """
    Check if the audio contains human speech using webrtcvad.
    """
    pcm_audio_path = convert_to_pcm(audio_path)

    vad = webrtcvad.Vad()
    vad.set_mode(1)  # Mode 1: more aggressive detection of speech

    with contextlib.closing(wave.open(pcm_audio_path, 'rb')) as wf:
        sample_rate = wf.getframerate()
        assert sample_rate == 16000, "Sample rate must be 16000 Hz"
        pcm_data = wf.readframes(wf.getnframes())

        frame_duration = 30  # Frame size: 30ms
        frame_size = int(sample_rate * frame_duration / 1000)

        for i in range(0, len(pcm_data), frame_size * 2):  # 2 bytes per sample (16-bit audio)
            frame = pcm_data[i:i + frame_size * 2]
            if vad.is_speech(frame, sample_rate):
                return True, "Voice detected in the audio."

    return False, "No voice detected in the audio."


def transcribe_audio(audio_path):
    model = whisper.load_model("small")
    result = model.transcribe(audio_path)

    transcription_text_file = "transcription.txt"
    transcription_text = result['text']
    with open(transcription_text_file, 'w') as f:
        f.write(transcription_text)

    print("Transcription saved to transcription.txt")
    return transcription_text_file


def extract_images_from_video(video_path, output_dir="frames", interval=10):
    """
    Extract frames from video every 'interval' seconds.
    :param video_path: Path to the video file.
    :param output_dir: Directory to save the extracted images.
    :param interval: Time interval between frames to extract (in seconds).
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)  # Frames per second
    frame_interval = int(fps * interval)

    count = 0
    success = True
    while success:
        success, frame = cap.read()
        if count % frame_interval == 0 and success:
            # Use zero-padding to ensure correct sorting
            img_path = os.path.join(output_dir, f"frame_{count:05d}.jpg")
            cv2.imwrite(img_path, frame)
        count += 1

    cap.release()
    return output_dir


def is_unique_image(img_path, unique_images, similarity_threshold=0.95):
    """
    Check if the image is unique compared to already processed unique images.
    :param img_path: Path to the image file.
    :param unique_images: List of already processed unique images.
    :param similarity_threshold: Similarity threshold for uniqueness.
    :return: True if the image is unique, False otherwise.
    """
    img = cv2.imread(img_path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    for unique_img in unique_images:
        unique_img_gray = cv2.cvtColor(unique_img, cv2.COLOR_BGR2GRAY)
        # Calculate the Structural Similarity Index (SSI)
        score = ssim(img_gray, unique_img_gray)
        if score >= similarity_threshold:
            return False  # Image is similar to an already processed unique image

    return True  # Image is unique


def extract_text_from_image(img_path):
    """
    Extract text from an image using EasyOCR. Skip if no text is detected.
    :param img_path: Path to the image file.
    :return: Extracted text or None if no text is detected.
    """
    reader = easyocr.Reader(['en'])
    results = reader.readtext(img_path)

    if not results:  # No text detected
        print(f"No text detected in image: {img_path}")
        return None

    extracted_text = " ".join([text[1] for text in results])  # Concatenate all detected text
    return extracted_text

def analyze_frames(frames_dir, similarity_threshold=0.95):
    """
    Analyze frames in a directory to extract text from unique images.
    :param frames_dir: Directory containing the frames.
    :param similarity_threshold: Threshold for image similarity.
    :return: Dictionary of extracted texts from unique images.
    """
    unique_images = []
    extracted_texts = {}

    for image_file in os.listdir(frames_dir):
        img_path = os.path.join(frames_dir, image_file)
        if is_unique_image(img_path, unique_images, similarity_threshold):
            # If unique, read the image and extract text
            extracted_text = extract_text_from_image(img_path)
            extracted_texts[image_file] = extracted_text
            # Store the image for future uniqueness checks
            unique_images.append(cv2.imread(img_path))

    return extracted_texts

def transcribe_audio(audio_path):
    """
    Transcribe audio using Whisper model.
    :param audio_path: Path to the audio file.
    :return: Transcribed text.
    """
    model = whisper.load_model("small")
    result = model.transcribe(audio_path)
    return result['text']


import json

def text_file(video_file):
    # Extract text from frames (analyze_frames function extracts text from images)
    extracted_texts = analyze_frames(frames_directory)

    # Transcribe audio (transcribe_audio function processes the audio file and returns the transcription)
    audio_text = transcribe_audio(audio_file)

    # Combine all the extracted text from frames into one block of text
    combined_image_texts = " ".join(text for text in extracted_texts.values())

    # Store the audio text and combined image text in a dictionary
    combined_texts = {
        "Audio Text": audio_text,
        "Extracted Texts from Images": combined_image_texts
    }

    # Save the combined text to a JSON file
    output_file_path = 'extracted_texts_combined.json'
    with open(output_file_path, 'w') as file:
        json.dump(combined_texts, file, indent=4)

    return output_file_path


def verification_report():
    # Set the TogetherAI API key
    api_key = "0b79e2f0ddb16654bc98df9f828e0474d53c7d00eac41328abf06bd4858d14bb"
    #       combined_text = text_file()
    # Load the transcribed text from the file
    with open(combined_text_file, "r") as file:
        transcript = file.read()

    # Create a more detailed and structured prompt for verification task
    prompt = f"""
    You are tasked with analyzing the transcription of a business model canvas (BMC) presentation video.
    Please perform the following checks and provide the results in JSON format:

    1. **Names:** Extract the names of students mentioned when they introduce themselves.
    2. **Introduction:** Check if there is a self introduction with name at the beginning of the transcription. Indicate 'Yes' if an introduction is present and 'No' if not.
    3. **Relevance:** Verify if the content of the transcription is relevant to the video title '{title}'. Answer 'Yes' or 'No' and explain briefly why the content is or is not relevant.
    4. **BMC Topics:** Confirm whether the transcription covers all 9 BMC topics:
    key partners,
    key activities,
    value propositions,
    customer relationships,
    customer segments,
    key resources,
    channels,
    cost structure,
    and revenue streams. Answer 'Yes' or 'Some topics missed' and list the missing topics if any.

    Transcription to analyze:
    {transcript}

    Please return the results in the following format:
    - **Names:** [...]
    - **Introduction:** [Yes/No]
    - **Relevance to Title:** [Yes/No]
    - **BMC Topics Coverage:** [Yes/No]
    - **Overall Analysis:** [Brief assessment]

    Thank you!
    """



    # API endpoint
    url = "https://api.together.xyz/v1/chat/completions"

    # Headers
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    # Request payload
    data = {
        "model": "meta-llama/Llama-Vision-Free",
        "messages": [{"role": "user", "content": prompt}]
    }

    # Send the request to TogetherAI API
    response = requests.post(url, headers=headers, data=json.dumps(data))

    # Check if the request was successful
    if response.status_code == 200:
        # Parse the response
        response_content = response.json()
        # Extract the relevant message from the response
        message_content = response_content["choices"][0]["message"]["content"]

        # Split the response into lines for better formatting
        results = message_content.splitlines()

        # Display the report using st.write for structured formatting
        for line in results:
            st.write(line)  # Display each line as a separate markdown element
    else:
        st.error(f"Request failed with status code {response.status_code}: {response.text}")


# Streamlit app layout
st.title(" :blue[BMC Video Analyzer]")
# Option to download video
video_url = st.text_input(" :green[Enter BMC Video YouTube URL:]")

if st.button("Analyze Video"):
    if video_url:
        # Show spinner while the analysis is in progress
        with st.spinner("Analyzing video..."):
            # Call the functions from your existing code
            title = get_video_info(video_url)
            st.write(f"Video Title: {title}")

            video_file = download_video(video_url)
            if video_file:

                # Spinner for checking video quality
                with st.spinner("Checking video quality..."):
                    quality_check, quality_report = check_video_quality(video_file)
                    st.write(quality_report)

                if quality_check:
                    with st.spinner("Extracting audio..."):
                        audio_file = extract_audio(video_file)

                    if audio_file:
                        audio_checks = []

                        # Spinners for individual audio checks
                        with st.spinner("Checking audio quality..."):
                            audio_checks.append(check_audio_quality(audio_file))
                        with st.spinner("Checking audio tone..."):
                            audio_checks.append(check_audio_tone(audio_file))
                        with st.spinner("Checking background noise..."):
                            audio_checks.append(check_background_noise(audio_file))
                        with st.spinner("Checking voice presence..."):
                            audio_checks.append(check_voice_presence(audio_file))

                        if all(check[0] for check in audio_checks):
                            with st.spinner("Extracting frames from video..."):
                                frames_directory = extract_images_from_video(video_file)

                            with st.spinner("Combining audio transcription and extracted text..."):
                                combined_text_file = text_file(video_file)

                            if combined_text_file:
                                with st.spinner("Generating verification report..."):
                                    verification_report()  # Display the report with structured formatting
                            else:
                                st.error("Failed to combine the text.")
                        else:
                            for check in audio_checks:
                                if not check[0]:
                                    st.error(check[1])
                    else:
                        st.error("Audio extraction failed.")
                else:
                    st.error("Video quality is not sufficient for analysis.")
            else:
                st.error("Failed to download the video.")
    else:
        st.warning("Please enter a valid video URL.")
