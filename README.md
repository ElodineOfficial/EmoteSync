# EmoteSync: A PNGTuber’s best friend!

EmoteSync is a Python application that analyzes audio files, detects emotions using AI models, and generates videos with corresponding emotion images overlaid on a background video or transparent background. It leverages AI models like OpenAI's Whisper for speech recognition and Hugging Face's transformers for emotion detection.

## Table of Contents

- [Features](#features)
- [Dependencies](#dependencies)
- [Installation](#installation)
  - [Prerequisites](#prerequisites)
  - [Clone the Repository](#clone-the-repository)
  - [Create a Virtual Environment](#create-a-virtual-environment)
  - [Install Required Packages](#install-required-packages)
  - [Install FFmpeg](#install-ffmpeg)
  - [Verify Installation](#verify-installation)
- [Usage](#usage)
  - [Prepare Emotion Images](#prepare-emotion-images)
  - [Run the Application](#run-the-application)
  - [Using the GUI](#using-the-gui)
- [Examples](#examples)
  - [Example 1: Create a Video with a Background Video](#example-1-create-a-video-with-a-background-video)
  - [Example 2: Create a Video with a Transparent Background](#example-2-create-a-video-with-a-transparent-background)
- [Troubleshooting](#troubleshooting)
- [Known Issues](#known-issues)
- [License](#license)
- [Acknowledgments](#acknowledgments)
- [Support](#support)

## Features

- **Audio Emotion Detection**: Analyze audio files and detect emotions in speech using AI models.
- **Video Generation**: Generate videos with emotion images corresponding to detected emotions overlaid on a background video or transparent background.
- **Background Video Support**: Optionally overlay emotion images on a background video.
- **Bounce Effect**: Enable a bounce animation effect on emotion images.
- **Avoid Repeating Images**: Merge consecutive segments with the same emotion to avoid repeating images.

## Dependencies

EmoteSync requires the following Python packages and tools:

- [Python 3.7 or higher](https://www.python.org/)
- [OpenAI Whisper](https://github.com/openai/whisper)
- [Hugging Face Transformers](https://github.com/huggingface/transformers)
- [PyTorch](https://pytorch.org/)
- [Pydub](https://github.com/jiaaro/pydub)
- [MoviePy](https://zulko.github.io/moviepy/)
- [SoundFile](https://pypi.org/project/SoundFile/)
- [NumPy](https://numpy.org/)
- [tkinter](https://docs.python.org/3/library/tkinter.html)
- [FFmpeg](https://ffmpeg.org/) (required by MoviePy and Pydub)

**Note**: FFmpeg is essential for audio and video processing. Ensure that FFmpeg is installed and accessible in your system's PATH.

## Installation

### Prerequisites

- **Python 3.7 or higher**: [Download Python](https://www.python.org/downloads/)
- **FFmpeg**: Install FFmpeg from [here](https://ffmpeg.org/download.html) or use a package manager like `brew`, `apt`, or `chocolatey`.

### Clone the Repository

```bash
git clone https://github.com/ElodineOfficial/EmoteSync.git
cd EmoteSync
```

### Create a Virtual Environment

It is recommended to use a virtual environment to manage dependencies.

```bash
# On Windows
python -m venv venv
venv\Scripts\activate

# On macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### Install Required Packages

Install the required Python packages using `pip`:

```bash
pip install -r requirements.txt
```

**Contents of `requirements.txt`:**

```
openai-whisper
transformers
pydub
moviepy
soundfile
numpy
torch
```

**Note**: If you have a GPU and want to use it for processing, install the GPU-enabled version of PyTorch. Visit [PyTorch.org](https://pytorch.org/get-started/locally/) for installation instructions.

### Install FFmpeg

#### On Windows:

Download FFmpeg from the [official website](https://ffmpeg.org/download.html#build-windows). Extract the files and add the `bin` directory to your system's PATH environment variable.

#### On macOS:

```bash
brew install ffmpeg
```

#### On Linux (Ubuntu/Debian):

```bash
sudo apt-get update
sudo apt-get install ffmpeg
```

### Verify Installation

Run the following command to verify that all dependencies are installed correctly:

```bash
python -c "import whisper; import transformers; import pydub; import moviepy; import soundfile; import numpy; import torch; import tkinter; print('All dependencies are installed successfully.')"
```

If no errors occur and you see the message "All dependencies are installed successfully.", you're ready to use EmoteSync.

## Usage

### Prepare Emotion Images

Create a folder containing images representing different emotions. The images should be named exactly as follows:

- `happy.png`
- `neutral.png`
- `annoyance.png`
- `anger.png`
- `confusion.png`
- `disbelief.png`

Place the images in a folder and remember the path to this folder.

**Notes:**

- Images should be in PNG format with transparent backgrounds if possible.
- If an image for a specific emotion is missing, the script will attempt to use `neutral.png` as a fallback.
- Ensure that the images are appropriately sized. The script resizes images to match the background video's height, but having images with reasonable dimensions is recommended.
- You can easily edit this portion to add more emotions and variation, these are just a base few to get you started!

### Run the Application

You can run the application by executing the script:

```bash
python EmoteSync.py
```

**Note**: Replace `EmoteSync.py` with the actual filename if you changed it.

### Using the GUI

1. **About Us**: Click on the "About Us" button to learn more about EmoteSync.

2. **Select Audio File**:

   - Click on the "Browse..." button next to the "Audio File" field.
   - Select the audio file you want to process.
   - Supported audio formats include WAV, MP3, M4A, and others supported by Pydub and FFmpeg.

3. **Select Emotion Images Folder**:

   - Click on the "Browse..." button next to the "Emotion Images Folder" field.
   - Select the folder containing your emotion images.

4. **Background Video (Optional)**:

   - If you want to use a background video, ensure the "Use Background Video" checkbox is checked (default).
   - Click on the "Browse..." button next to the "Background Video" field.
   - Select your background video file.
   - Supported video formats include MP4, AVI, MOV, and others supported by MoviePy and FFmpeg.

5. **Use Background Video**:

   - If you do not want to use a background video, uncheck the "Use Background Video" checkbox.
   - A transparent background will be used instead.

6. **Enable Bounce Effect**:

   - Check the "Enable Bounce Effect" checkbox if you want the emotion images to have a bounce animation effect.
   - Don't forget, if you want to changethe bounce effect I've kept things organized so you can easily do that!

7. **Avoid Repeating Images**:

   - Check the "Avoid Repeating Images" checkbox to merge consecutive segments with the same emotion, preventing the same image from appearing back-to-back.

8. **Start Processing**:

   - Click the "Start Processing" button to begin processing the video.
   - The status label at the bottom will update to show the current processing step.
   - A message box will appear upon completion or if an error occurs.

### Output

- The processed video will be saved in the same directory as the script with the filename:
  - `output_video.mp4` if a background video is used.
  - `output_video.webm` if no background video is used (video will have a transparent background).

- The output video will have the same duration as your audio file.

## Examples

### Example 1: Create a Video with a Background Video

1. Prepare your audio file, e.g., `speech.mp3`.
2. Prepare your emotion images in a folder, e.g., `emotion_images/`.
3. Prepare your background video, e.g., `background.mp4`.
4. Run the application and select the above files and folder.
5. Optionally, enable the bounce effect and avoid repeating images.
6. Click "Start Processing".
7. Upon completion, find `output_video.mp4` in the script's directory.

### Example 2: Create a Video with a Transparent Background

1. Prepare your audio file, e.g., `speech.mp3`.
2. Prepare your emotion images in a folder, e.g., `emotion_images/`.
3. Uncheck the "Use Background Video" checkbox.
4. Run the application and select the audio file and emotion images folder.
5. Click "Start Processing".
6. Upon completion, find `output_video.webm` in the script's directory.

**Note**: The output video will have a transparent background and may not play correctly in some media players. Use a compatible player like VLC or embed the video in a web page.

## Troubleshooting

- **Missing Emotion Images**:

  - Ensure that all required images are present in the emotion images folder with the exact filenames.
  - The application will warn you if any images are missing and halt processing.

- **Error During Processing**:

  - If an error occurs, check the console output for error messages.
  - Common issues include missing dependencies, corrupted files, or unsupported formats.

- **Slow Performance**:

  - Processing time depends on the length of the audio file and the performance of your system.
  - Using the `tiny` or `small` versions of the Whisper model can speed up processing at the cost of accuracy.
  - If you have a GPU, ensure that PyTorch is installed with CUDA support.

- **Audio File Not Loading**:

  - Ensure that the audio file is in a supported format and is not corrupted.
  - Install FFmpeg and ensure it's accessible in your system's PATH.

- **SoundFile Error**:

  - If you encounter an error related to `soundfile` or `libsndfile`, install `soundfile` via pip:

    ```bash
    pip install soundfile
    ```

  - Ensure that you have the necessary system libraries. On Ubuntu/Debian, you can install libsndfile:

    ```bash
    sudo apt-get install libsndfile1
    ```

## Known Issues

- **GPU Memory Usage**:

  - Using larger Whisper models (e.g., `medium`, `large`) requires significant GPU memory.
  - If you encounter memory errors, try using a smaller model like `base` or `small`.

- **Video Output Issues**:

  - If the output video does not play correctly, ensure that you have the necessary codecs installed.
  - Transparent videos require specific codecs and may not be supported by all players.

- **Model Download Times**:

  - The first time you run the application, it will download the Whisper model and the emotion detection model, which may take some time depending on your internet speed.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **OpenAI Whisper**: [GitHub Repository](https://github.com/openai/whisper)
- **Hugging Face Transformers**: [GitHub Repository](https://github.com/huggingface/transformers)
- **MoviePy**: [Documentation](https://zulko.github.io/moviepy/)
- **Pydub**: [GitHub Repository](https://github.com/jiaaro/pydub)
- **tkinter**: [Python Documentation](https://docs.python.org/3/library/tkinter.html)

## Support

If you find this tool useful, please consider supporting the developer:

- [Buy Me a Coffee](https://buymeacoffee.com/elodine)

Your support enables the continued development of open-source tools like this one!
