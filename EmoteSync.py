import os
import logging
import threading
import tkinter as tk
from tkinter import filedialog, messagebox
import random  # Import random module for displaying support message

from moviepy.editor import VideoFileClip, ImageClip, CompositeVideoClip, AudioFileClip, ColorClip
from pydub import AudioSegment
import whisper
from transformers import pipeline
import soundfile as sf  # Ensure soundfile is imported
import numpy as np  # Import numpy for mathematical functions

# Set up logging for debugging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def select_audio_file():
    audio_file = filedialog.askopenfilename(
        title="Select Audio File",
        filetypes=[("All Files", "*.*")]
    )
    audio_file_var.set(audio_file)

def select_background_video():
    background_file = filedialog.askopenfilename(
        title="Select Background Video",
        filetypes=[("All Files", "*.*")]
    )
    background_file_var.set(background_file)

def select_emotion_images_folder():
    emotion_folder = filedialog.askdirectory(title="Select Folder Containing Emotion Images")
    emotion_folder_var.set(emotion_folder)

def start_processing():
    audio_file = audio_file_var.get()
    use_bg = use_background_var.get()
    background_video = background_file_var.get() if use_bg else ''
    emotion_folder = emotion_folder_var.get()
    if not audio_file or not emotion_folder:
        messagebox.showerror("Error", "Please select an audio file and emotion images folder.")
        return
    if use_bg and not background_video:
        messagebox.showerror("Error", "Please select a background video or uncheck 'Use Background Video'.")
        return
    # Disable the start button to prevent multiple clicks
    start_button.config(state='disabled')
    # Run processing in a separate thread to keep UI responsive
    threading.Thread(target=process_video, args=(audio_file, background_video, emotion_folder, use_bg)).start()

def update_status(message):
    status_var.set(message)

# Function to load emotion images
def load_emotion_images(folder_path):
    emotions = ['happy', 'neutral', 'annoyance', 'anger', 'confusion', 'disbelief']
    emotion_images = {}
    for emotion in emotions:
        img_path = os.path.join(folder_path, f"{emotion}.png")
        if os.path.isfile(img_path):
            emotion_images[emotion] = img_path
        else:
            logger.warning(f"No image found for emotion '{emotion}'. Using default image.")
            default_img_path = os.path.join(folder_path, 'neutral.png')
            if os.path.isfile(default_img_path):
                emotion_images[emotion] = default_img_path
            else:
                emotion_images[emotion] = None  # Handle missing images appropriately
    return emotion_images

# Function to transcribe audio using Whisper
def transcribe_audio(audio_file, model):
    try:
        result = model.transcribe(audio_file)
    except Exception as e:
        logger.error(
            "An error occurred during transcription. Please ensure that 'soundfile' is installed "
            "and that the audio file is in a supported format."
        )
        raise e
    return result['text']

# Function to detect emotions in text
def detect_emotions_in_text(text, emotion_classifier, emotion_mapping):
    # Get emotions from the classifier
    emotions = emotion_classifier(text)
    # Since return_all_scores=True, emotions is a list containing a list of dicts
    # Access the first (and only) element
    emotions = emotions[0]
    # Create a dictionary of emotion scores
    emotion_scores = {e['label']: e['score'] for e in emotions}
    # Determine the emotion with the highest score
    model_emotion = max(emotion_scores, key=emotion_scores.get)
    # Map the model's emotion to your desired emotion labels
    emotion = emotion_mapping.get(model_emotion, 'neutral').lower()
    return emotion

# Function to split audio into chunks
def split_audio(audio_file, chunk_length_ms=5000):
    try:
        audio = AudioSegment.from_file(audio_file)
    except Exception as e:
        logger.error(f"Could not read audio file '{audio_file}'. Error: {e}")
        raise e
    chunks = []
    for i in range(0, len(audio), chunk_length_ms):
        chunk = audio[i:i + chunk_length_ms]
        chunk_file = f"chunk_{i // chunk_length_ms}.wav"
        chunk.export(chunk_file, format="wav")
        start_time = i / 1000.0
        end_time = min((i + chunk_length_ms) / 1000.0, audio.duration_seconds)
        chunks.append((chunk_file, start_time, end_time))
    return chunks

# Function to process audio chunks and build the emotion timeline
def process_audio_chunks(audio_file, emotion_classifier, emotion_mapping, model):
    chunks = split_audio(audio_file)
    emotion_timeline = []
    chunk_count = len(chunks)
    for idx, (chunk_file, start_time, end_time) in enumerate(chunks):
        update_status(f"Processing chunk {idx+1}/{chunk_count}...")
        # Transcribe the audio chunk
        text = transcribe_audio(chunk_file, model)
        if not text.strip():
            emotion = 'neutral'
        else:
            # Detect the dominant emotion in the transcribed text
            emotion = detect_emotions_in_text(text, emotion_classifier, emotion_mapping)
        # Append the emotion data to the timeline
        emotion_timeline.append({
            'start': start_time,
            'end': end_time,
            'emotion': emotion
        })
        # Clean up the temporary chunk file
        os.remove(chunk_file)
    return emotion_timeline

# Function to detect emotions in the audio file
def detect_emotions(audio_file, emotion_classifier, emotion_mapping, model):
    return process_audio_chunks(audio_file, emotion_classifier, emotion_mapping, model)

# Function to merge consecutive segments with the same emotion
def merge_consecutive_segments(emotion_timeline):
    if not emotion_timeline:
        return emotion_timeline
    merged_timeline = [emotion_timeline[0]]
    for segment in emotion_timeline[1:]:
        last_segment = merged_timeline[-1]
        if segment['emotion'] == last_segment['emotion']:
            # Merge the segments by extending the end time
            last_segment['end'] = segment['end']
        else:
            merged_timeline.append(segment)
    return merged_timeline

# The main processing function
def process_video(audio_file, background_video_file, emotion_folder, use_background):
    try:
        update_status("Loading models and files...")
        # Load emotion images
        emotion_images = load_emotion_images(emotion_folder)
        
        # Check if all required emotion images are loaded
        missing_images = [emotion for emotion, img in emotion_images.items() if img is None]
        if missing_images:
            messagebox.showerror("Error", f"Missing images for emotions: {', '.join(missing_images)}")
            update_status("Processing halted due to missing images.")
            start_button.config(state='normal')
            return
        
        # Mapping from model labels to desired emotions
        emotion_mapping = {
            'joy': 'happy',
            'neutral': 'neutral',
            'anger': 'anger',
            'sadness': 'neutral',     # Map 'sadness' to 'neutral' or add 'sadness' if you have an image
            'fear': 'confusion',
            'surprise': 'disbelief',
            'disgust': 'annoyance',
        }
        
        # Initialize the emotion classifier
        emotion_classifier = pipeline(
            'text-classification',
            model='j-hartmann/emotion-english-distilroberta-base',
            return_all_scores=True,  # Use return_all_scores=True to get all scores
            framework='pt'  # Ensure you're using PyTorch
        )
        
        # Load the Whisper model
        model = whisper.load_model("base")  # Options: 'tiny', 'small', 'medium', 'large'
        
        # Load audio
        update_status("Loading audio file...")
        try:
            audio_clip = AudioFileClip(audio_file)
        except Exception as e:
            logger.error(f"Could not load audio file '{audio_file}'. Error: {e}")
            update_status("An error occurred while loading the audio file.")
            start_button.config(state='normal')
            messagebox.showerror("Error", f"Could not load audio file:\n{e}")
            return

        # Use audio duration as total duration
        total_duration = audio_clip.duration

        # Process emotion timeline
        update_status("Detecting emotions in audio...")
        emotion_timeline = detect_emotions(audio_file, emotion_classifier, emotion_mapping, model)

        # Merge consecutive segments if "Avoid Repeating Images" is enabled
        if avoid_repeating_var.get():
            emotion_timeline = merge_consecutive_segments(emotion_timeline)

        # Set default fps
        fps = 24  # You can adjust this value as needed

        # Load background video if provided
        if use_background:
            if not background_video_file:
                messagebox.showerror("Error", "Background video file not specified.")
                update_status("Processing halted due to missing background video.")
                start_button.config(state='normal')
                return
            update_status("Loading background video...")
            try:
                background_video = VideoFileClip(background_video_file)
                fps = background_video.fps  # Use the fps of the background video
            except Exception as e:
                logger.error(f"Could not load video file '{background_video_file}'. Error: {e}")
                update_status("An error occurred while loading the video file.")
                start_button.config(state='normal')
                messagebox.showerror("Error", f"Could not load video file:\n{e}")
                return
            
            # Adjust background video duration
            update_status("Adjusting background video...")
            if background_video.duration < total_duration:
                background_video = background_video.loop(duration=total_duration)
            elif background_video.duration > total_duration:
                background_video = background_video.subclip(0, total_duration)
        else:
            # No background video provided
            # Get dimensions from the first emotion image
            first_image_path = next(iter(emotion_images.values()))
            if first_image_path and os.path.isfile(first_image_path):
                first_image_clip = ImageClip(first_image_path)
                width, height = first_image_clip.size
                first_image_clip.close()
            else:
                width, height = 640, 480  # Default dimensions if image not found

            # Create a transparent background clip
            background_video = ColorClip(size=(width, height), color=(0, 0, 0, 0)).set_duration(total_duration)
            background_video.fps = fps  # Set fps attribute for the background video

        # Create bounce effect function (if bounce is enabled)
        def bounce_effect(t):
            # Bounce effect duration
            d = 0.5  # 0.5 seconds
            if t < d:
                # Bounce scaling factor
                return 1 + 0.05 * np.sin(2 * np.pi * 2 * t / d) * np.exp(-4 * t / d)
            else:
                return 1

        # Create image clips based on emotion timeline
        update_status("Creating emotion overlays...")
        clips = []
        for segment in emotion_timeline:
            emotion = segment['emotion'].lower()
            img_path = emotion_images.get(emotion, emotion_images['neutral'])
            if not os.path.isfile(img_path):
                logger.warning(f"No image found for emotion '{emotion}'. Using neutral image.")
                img_path = emotion_images['neutral']
        
            logger.debug(
                f"Creating clip for emotion '{emotion}' from {segment['start']} to {segment['end']}, "
                f"using image '{img_path}'"
            )

            # Create the image clip
            img_clip = (
                ImageClip(img_path, transparent=True)
                .set_start(segment['start'])
                .set_duration(segment['end'] - segment['start'])
                .set_position(('center', 'center'))
            )

            # Resize image to match background dimensions
            img_clip = img_clip.resize(height=background_video.h)

            # Apply bounce effect if enabled
            if bounce_var.get():
                img_clip = img_clip.resize(lambda t: bounce_effect(t))

            # Set fps for image clip if necessary
            img_clip.fps = fps

            clips.append(img_clip)
        
        # Compose final video
        update_status("Composing final video...")
        final_video = CompositeVideoClip([background_video] + clips)

        # Set fps for the final video
        final_video.fps = fps

        final_video = final_video.set_duration(total_duration)
        final_video = final_video.set_audio(audio_clip)
        
        # Export video
        update_status("Exporting video...")
        # Determine output codec and parameters
        if not use_background:
            # If no background video, we need to preserve transparency
            codec = 'libvpx-vp9'  # VP9 codec supports alpha channel
            output_kwargs = {
                'codec': codec,
                'audio_codec': 'libvorbis',  # Changed from 'aac' to 'libvorbis'
                'preset': 'ultrafast',
                'ffmpeg_params': ['-pix_fmt', 'yuva420p']
            }
            output_filename = 'output_video.webm'  # Use .webm format for VP9 with alpha channel
        else:
            # Use standard codec
            codec = 'libx264'
            output_kwargs = {'codec': codec, 'audio_codec': 'aac'}
            output_filename = 'output_video.mp4'

        final_video.write_videofile(output_filename, fps=fps, **output_kwargs)
        
        update_status("Processing completed!")
        # Enable the start button again
        start_button.config(state='normal')
        # Show message box to inform user
        messagebox.showinfo("Success", f"Video processing completed successfully!\nOutput file: {output_filename}")
    except Exception as e:
        logger.exception("An error occurred during processing.")
        update_status("An error occurred.")
        start_button.config(state='normal')
        messagebox.showerror("Error", f"An error occurred during processing:\n{str(e)}")

# Function to enable or disable background video selection based on the checkbox
def toggle_background_video():
    if use_background_var.get():
        background_file_entry.config(state='normal')
        background_browse_button.config(state='normal')
    else:
        background_file_entry.config(state='disabled')
        background_browse_button.config(state='disabled')
        background_file_var.set('')

# Function to show support message randomly
def show_support_message():
    messagebox.showinfo(
        "Support",
        "If you find this tool useful, please consider supporting me at https://buymeacoffee.com/elodine. Your support enables me to continue the development of FOSS tools like this one!"
    )

# Function to display 'About Us' information
def show_about_us():
    messagebox.showinfo(
        "About This Tool",
        "Welcome to EmoteSync!\n\n"
        "This project is designed to analyze audio files, detect emotions, and generate videos with corresponding emotion images and backgrounds. "
        "It leverages AI models like OpenAI's Whisper for speech recognition and Hugging Face's transformers for emotion detection.\n\n"
    )

# Initialize the main window
root = tk.Tk()
root.title("EmoteSync")

# Variables to hold file paths and options
audio_file_var = tk.StringVar()
background_file_var = tk.StringVar()
emotion_folder_var = tk.StringVar()
status_var = tk.StringVar()
bounce_var = tk.BooleanVar()  # Variable to control bounce effect
use_background_var = tk.BooleanVar(value=True)  # Variable to control background video usage
avoid_repeating_var = tk.BooleanVar()  # Variable to control avoiding repeating images

# Create and place widgets
# Row 0: About Us button
about_button = tk.Button(root, text="About Us", command=show_about_us)
about_button.grid(row=0, column=2, sticky="e", padx=5, pady=5)

# Row 1: Audio File
tk.Label(root, text="Audio File:").grid(row=1, column=0, sticky="e")
tk.Entry(root, textvariable=audio_file_var, width=50).grid(row=1, column=1)
tk.Button(root, text="Browse...", command=select_audio_file).grid(row=1, column=2)

# Row 2: Emotion Images Folder
tk.Label(root, text="Emotion Images Folder:").grid(row=2, column=0, sticky="e")
tk.Entry(root, textvariable=emotion_folder_var, width=50).grid(row=2, column=1)
tk.Button(root, text="Browse...", command=select_emotion_images_folder).grid(row=2, column=2)

# Row 3: Background Video
tk.Label(root, text="Background Video:").grid(row=3, column=0, sticky="e")
background_file_entry = tk.Entry(root, textvariable=background_file_var, width=50)
background_file_entry.grid(row=3, column=1)
background_browse_button = tk.Button(root, text="Browse...", command=select_background_video)
background_browse_button.grid(row=3, column=2)

# Row 4: Use Background Video checkbox
tk.Checkbutton(root, text="Use Background Video", variable=use_background_var, command=toggle_background_video).grid(row=4, column=1, sticky="w")

# Row 5: Enable Bounce Effect checkbox
tk.Checkbutton(root, text="Enable Bounce Effect", variable=bounce_var).grid(row=5, column=1, sticky="w")

# Row 6: Avoid Repeating Images checkbox
tk.Checkbutton(root, text="Avoid Repeating Images", variable=avoid_repeating_var).grid(row=6, column=1, sticky="w")

# Row 7: Start Processing button
start_button = tk.Button(root, text="Start Processing", command=start_processing)
start_button.grid(row=7, column=1, pady=10)

# Row 8: Status Label
tk.Label(root, textvariable=status_var).grid(row=8, column=1)

# Row 9: Sticky AI usage message
tk.Label(root, text="The use of this tool directly supports AI").grid(row=9, column=0, columnspan=3, sticky='we')

# Configure grid weights to make the bottom label sticky
root.grid_rowconfigure(8, weight=1)
root.grid_columnconfigure(1, weight=1)

# Initialize the background video fields based on the checkbox
toggle_background_video()

# Randomly show support message approximately once every six times the program is opened
if random.randint(1, 6) == 1:
    show_support_message()

root.mainloop()
