
![Logo](/assets/ASL-cover-image.webp)

![GitHub top language](https://img.shields.io/github/languages/top/rishabhshah13/Real-Time-SLR?style=flat-square)
![GitHub repo size](https://img.shields.io/github/repo-size/rishabhshah13/Real-Time-SLR?color=yellow&style=flat-square)

# Real-time Sign Language Recognition

## Overview
Sign language is an important means of communication for individuals with hearing impairments. This project aims to build a real-time sign language gesture recognition system using deep learning techniques. The system utilizes 1D Convolutional Neural Networks (1DCNN) and Transformers to recognize sign language gestures based on the hand landmarks extracted from MediaPipe.

## Installation
1. Clone this repository to your local machine using either the HTTPS or SSH link provided on the repository's GitHub page. You can use the following command to clone the repository via HTTPS:

```bash
git clone https://github.com/rishabhshah13/Real-Time-SLR.git
```

2. Once the repository is cloned, navigate to the root directory of the project:

```bash
cd Real-Time-SLR
```

3. It is recommended to create a virtual environment to isolate the dependencies of this project. You can create a virtual environment using venv module. Run the following command to create a virtual environment named "venv":

```bash
python3 -m venv Real-Time-SLR
```

4. Activate the virtual environment. The activation steps depend on the operating system you're using:

- For Windows:
```bash
venv\Scripts\activate
```
- For macOS/Linux:
```bash
source venv/bin/activate
```

5. Now, you can install the required dependencies by running the following command:

```bash
pip install -r requirements.txt
```

6. Once the installation is complete, you're ready to use the real-time sign language gesture recognition system.

**Note:** Make sure you have a webcam connected to your machine or available on your device for capturing hand gestures.

You have successfully installed the system and are ready to use it for real-time sign language gesture recognition. Please refer to the [Usage](#usage) section in the `README.md` for instructions on how to run and utilize the system.

## Usage
To use the sign language gesture recognition system, follow these steps:

1. Ensure that you have installed all the required dependencies (see [Installation](#installation)).

2. Choose the method of running the application:
   - **Local OpenCV Module**:
     If you want to use the OpenCV module locally, run `cv2main.py`:

     ```bash
     python cv2main.py
     ```

   - **Streamlit Web UI**:
     If you want to use the Streamlit web UI, run `st_app.py`:

     ```bash
     streamlit run st_app.py
     ```

3. The system will start capturing your hand gestures using the webcam (for `cv2main.py`) or through the Streamlit web interface, and display the recognized gestures in real-time.


3. The system will start capturing your hand gestures using the webcam and display the recognized gestures in real-time.

## Models
The repository includes pre-trained models for sign language gesture recognition. The following models are available in the models directory:

- **fingerspelling**
  - letter_model.p: Trained model for fingerspelling lettersrecognition.
  - number_model.p: Trained model for fingerspelling numbers recognition.
- **gloss**
  - gloss_model.h5: Trained model for gloss recognition.



## Directory Structure
The directory structure of this repository is as follows:

```bash
├── Readme.md
├── assets
├── config
│   ├── config.py         # Configuration file for the project
│   └── config.yaml       # YAML configuration file for the project
├── cv2main.py            # Main script for real-time gesture recognition using OpenCV module
├── data
│   └── sign_to_prediction_index_map.json  # Mapping of sign to prediction index
├── dockerfile            # Dockerfile for containerizing the application
├── mac_requirements.txt  # Requirements file for macOS
├── models
│   ├── fingerspelling
│   │   ├── letter_model.p   # Pre-trained model for fingerspelling recognition (pickle format)
│   │   └── number_model.p   # Pre-trained model for fingerspelling recognition (pickle format)
│   └── gloss
│       └── gloss_model.h5   # Pre-trained model for gloss recognition (HDF5 format)
├── packages.txt           # Text file containing list of required packages
├── requirements.txt       # Requirements file for the project
├── scripts
│   ├── gloss
│   │   ├── backbone.py             # Backbone model for gloss recognition
│   │   ├── gloss_utils.py          # Utility functions for gloss recognition
│   │   ├── landmarks_extraction.py # Extracting landmarks for gloss recognition
│   │   └── my_functions.py         # Custom functions for gloss recognition
│   ├── inference
│   │   ├── fingerspellinginference.py  # Script for fingerspelling inference
│   │   └── glossinference.py           # Script for gloss inference
│   ├── train_classifier.py    # Script for training classifiers
│   ├── turn.py                # Script for TURN server
│   └── utils.py               # Utility functions used across scripts
└── st_app.py                 # Streamlit web UI application for gesture recognition

```
