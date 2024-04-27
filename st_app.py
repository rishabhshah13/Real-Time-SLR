import cv2
import mediapipe as mp
from scripts.gloss.backbone import TFLiteModel, get_model
from scripts.gloss.landmarks_extraction import load_json_file
import streamlit as st
import threading
from scripts.gloss.my_functions import *
from streamlit_webrtc import RTCConfiguration, WebRtcMode, webrtc_streamer
import av

import sys
import argparse

from autocorrect import Speller
from scripts.utils import load_model
from scripts.inference.fingerspellinginference import recognize_fingerpellings
from scripts.inference.glossinference import getglosses
from scripts.turn import get_ice_servers

# from config.config import *
from config.config import (
    model_letter_path,
    model_number_path,
    index_map,
    gloss_models_path,
    min_tracking_confidence,
    min_detection_confidence,
    MAX_HANDS,
    YELLOW,
    FONT,
    BLACK,
)

from streamlit_shortcuts import add_keyboard_shortcuts


# from flask import Flask, request, jsonify
# from flask_cors import CORS
# import cv2
# import numpy as np
# import time
# import mediapipe as mp
# from scripts.gloss.backbone import TFLiteModel, get_model
# from scripts.gloss.landmarks_extraction import mediapipe_detection, draw, extract_coordinates, load_json_file
# from config.config import SEQ_LEN, THRESH_HOLD
# import streamlit as st
# import threading
# import numpy as np
# import mediapipe as mp
# import cv2
# from scripts.gloss.my_functions import *
# from streamlit_webrtc import RTCConfiguration, WebRtcMode, webrtc_streamer
# import av

# import sys
# import cv2
# import argparse
# import numpy as np
# import mediapipe as mp
# from autocorrect import Speller
# from scripts.utils import load_model, save_gif, save_video
# from scripts.gloss.my_functions import *
# from scripts.gloss.landmarks_extraction import load_json_file
# from scripts.gloss.backbone import TFLiteModel, get_model
# from config.config import *

# from scripts.inference.fingerspellinginference import recognize_fingerpellings
# from scripts.inference.glossinference import getglosses
# from scripts.turn import get_ice_servers

# from streamlit_shortcuts import add_keyboard_shortcuts


# Initialize MediaPipe solutions
mp_holistic = mp.solutions.holistic
mp_hands = mp.solutions.hands

# Autocorrect Word
spell = Speller(lang="en")

# Load models
import pickle


@st.cache_resource()
def load_modela(model_path):
    """
    Load the TFLite model from the given path.

    Args:
        model_path (str): Path to the TFLite model file.

    Returns:
        model: Loaded TFLite model.
    """
    with open(model_path, "rb") as model_file:
        model_dict = pickle.load(model_file)
        model = model_dict["model"]
    return model


# Load maps
s2p_map = {k.lower(): v for k, v in load_json_file(index_map).items()}
p2s_map = {v: k for k, v in load_json_file(index_map).items()}
encoder = lambda x: s2p_map.get(x.lower())
decoder = lambda x: p2s_map.get(x)

# Load TFLite model

sequence_data = []


if "kerasmodel" not in st.session_state.keys():
    print("Loading Keras model")

    # models_path = ['sign_language/models/islr-fp16-192-8-seed_all42-foldall-last.h5']
    models = [get_model() for _ in gloss_models_path]
    for model, path in zip(models, gloss_models_path):
        model.load_weights(path)

    tflite_keras_model = TFLiteModel(islr_models=models)

    st.session_state["kerasmodel"] = tflite_keras_model

tflite_keras_model = st.session_state["kerasmodel"]


if "letter_model" not in st.session_state.keys():
    print("Loading Number model")
    print("Loading Letter model")

    letter_model = load_model(model_letter_path)
    number_model = load_model(model_number_path)

    st.session_state["letter_model"] = letter_model
    st.session_state["number_model"] = number_model

letter_model = st.session_state["letter_model"]
number_model = st.session_state["number_model"]


def parse_opt():
    """
    Parse command line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s", "--source", type=str, default=None, help="Video Path/0 for Webcam"
    )
    parser.add_argument(
        "-a", "--autocorrect", action="store_true", help="Autocorrect Misspelled Word"
    )
    parser.add_argument("-g", "--gif", action="store_true", help="Save GIF Result")
    parser.add_argument("-v", "--video", action="store_true", help="Save Video Result")
    parser.add_argument("-t", "--timing", type=int, default=8, help="Timing Threshold")
    parser.add_argument("-wi", "--width", type=int, default=800, help="Webcam Width")
    parser.add_argument("-he", "--height", type=int, default=600, help="Webcam Height")
    parser.add_argument("-f", "--fps", type=int, default=30, help="Webcam FPS")
    opt = parser.parse_args()
    return opt


def process_input(opt):
    """
    Process command line arguments.

    Args:
        opt (argparse.Namespace): Parsed arguments.

    Returns:
        tuple: Tuple containing video_path, fps, webcam_width, webcam_height.
    """
    global saveGIF, TIMING, autocorrect, number_mode, fingerspelling_mode, draw_landmarks_flag

    print(f"Timing Threshold is {TIMING} frames.")
    print(f"Using Autocorrect: {autocorrect}")

    if source == None or source.isnumeric():
        video_path = 0
    else:
        video_path = source

    fps = opt.fps
    webcam_width = opt.width
    webcam_height = opt.height

    return video_path, fps, webcam_width, webcam_height


# Locking frames from multi threading
lock = threading.Lock()

# Global variables
opt = parse_opt()

saveGIF = opt.gif
source = opt.source
TIMING = opt.timing
autocorrect = opt.autocorrect
number_mode = False
fingerspelling_mode = True
draw_landmarks_flag = False
file_path = "config/config.yaml"

video_path, fps, webcam_width, webcam_height = process_input(opt)
_output = [[], []]
output = []
frame_array = []
current_hand = 0
res = []


## CHECK THIS ONE PROPERLY!
import yaml


def edit_yaml_variable(file_path, variable_name, new_value):
    """
    Edit a variable in a YAML file.

    Args:
        file_path (str): Path to the YAML file.
        variable_name (str): Name of the variable to be edited.
        new_value: New value of the variable.
    """
    try:
        # Load the YAML file
        config = None
        while config == None:
            with open(file_path, "r") as f:
                config = yaml.safe_load(f)
    except FileNotFoundError as e:
        print("Error:", e)
        # If file not found, create an empty config
        config = {}

    # Edit the variable
    config[variable_name] = new_value

    # Write the changes back to the YAML file
    with open(file_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)


def read_yaml_variable(file_path, variable_name):
    """
    Read a variable from a YAML file.

    Args:
        file_path (str): Path to the YAML file.
        variable_name (str): Name of the variable to be read.

    Returns:
        Any: Value of the variable if it exists, None otherwise.
    """
    try:
        # Load the YAML file
        config = None
        while config == None:
            with open(file_path, "r") as f:
                config = yaml.safe_load(f)
            print("Read Config ", config)
        # Return the value of the variable if exists, otherwise return None
        return config.get(variable_name)

    except FileNotFoundError:
        # If file not found, return None
        return None


def change_fingerspelling_mode():
    """
    Change the fingerspelling mode.
    """
    global fingerspelling_mode
    print("Current fingerspelling_mode: ", fingerspelling_mode)
    fingerspelling_mode = read_yaml_variable(file_path, "fingerspelling_mode")
    fingerspelling_mode = not fingerspelling_mode
    edit_yaml_variable(file_path, "fingerspelling_mode", fingerspelling_mode)

    if fingerspelling_mode:
        st.write("Fingerspelling mode!")
    else:
        st.write("Gloss mode!")


def change_number_modemode():
    """
    Change the number mode.
    """
    global number_mode

    global number_mode
    print("Current number_mode: ", number_mode)
    number_mode = read_yaml_variable(file_path, "number_mode")
    number_mode = not number_mode
    edit_yaml_variable(file_path, "number_mode", number_mode)


def clearoutput():
    """
    Clear the output.
    """
    edit_yaml_variable(file_path, "output", [])
    edit_yaml_variable(file_path, "_output", [[], []])


def change_drawlandmarks():
    """
    Change the draw landmarks flag.
    """

    global draw_landmarks_flag

    print("Current draw_landmarks_flag: ", draw_landmarks_flag)
    draw_landmarks_flag = read_yaml_variable(file_path, "draw_landmarks_flag")
    draw_landmarks_flag = not draw_landmarks_flag
    edit_yaml_variable(file_path, "draw_landmarks_flag", draw_landmarks_flag)


st.button("fingerspelling", on_click=change_fingerspelling_mode)
st.button("number_mode", on_click=change_number_modemode)
st.button("clearoutput", on_click=clearoutput)
st.button("drawlandmarks", on_click=change_drawlandmarks)


add_keyboard_shortcuts(
    {"k": "fingerspelling", "l": "number_mode", "v": "clearoutput", "d": "drawlandmarks"}
)


def process_frame(
    image,
    fingerspelling_mode,
    number_mode,
    output,
    current_hand,
    TIMING,
    autocorrect,
    holistic,
    hands,
    _output,
    res,
    drawlandmarks,
):
    """
    Process a single frame.

    Args:
        image (ndarray): Input image.
        fingerspelling_mode (bool): Fingerspelling mode flag.
        number_mode (bool): Number mode flag.
        output (list): List containing detected gestures/characters.
        current_hand (int): Current hand index.
        TIMING (int): Timing threshold.
        autocorrect (bool): Autocorrect misspelled word flag.
        holistic (Holistic): MediaPipe Holistic instance.
        hands (Hands): MediaPipe Hands instance.
        _output (list): List containing previous output.
        res (list): List to store recognition results.
        drawlandmarks (bool): Flag to draw landmarks.

    Returns:
        tuple: Processed image, updated output list, updated current hand index, updated _output list.
    """
    global letter_model, number_model, tflite_keras_model, sequence_data

    if fingerspelling_mode:
        try:
            image, current_hand, output, _output = recognize_fingerpellings(
                image,
                number_mode,
                letter_model,
                number_model,
                hands,
                current_hand,
                output,
                _output,
                TIMING,
                autocorrect,
                drawlandmarks,
            )
        except Exception as error:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            print(f"{error}, line {exc_tb.tb_lineno}")
    else:
        try:
            image, sequence_data = getglosses(
                output,
                decoder,
                tflite_keras_model,
                sequence_data,
                holistic,
                image,
                res,
                drawlandmarks,
            )

        except Exception as error:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            print(f"{error}, line {exc_tb.tb_lineno}")

    return image, output, current_hand, _output


# Deployment configuration
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)


def video_frame_callback(frame):
    """
    Video frame callback function for WebRTC streaming.

    Args:
        frame (av.VideoFrame): Video frame.

    Returns:
        av.VideoFrame: Processed video frame.
    """
    global opt, video_path, fps, webcam_width, webcam_height, frame_array, current_hand, res  # ,  _output, output

    fingerspelling_mode = read_yaml_variable(file_path, "fingerspelling_mode")
    number_mode = read_yaml_variable(file_path, "number_mode")
    drawlandmarks = read_yaml_variable(file_path, "draw_landmarks_flag")

    output = read_yaml_variable(file_path, "output")
    _output = read_yaml_variable(file_path, "_output")

    image = frame.to_ndarray(format="bgr24")

    with mp_holistic.Holistic(
        min_detection_confidence=0.5, min_tracking_confidence=0.5
    ) as holistic:
        with mp_hands.Hands(
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            max_num_hands=MAX_HANDS,
        ) as hands:

            if video_path == 0:
                image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
            else:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            image, output, current_hand, _output = process_frame(
                image,
                fingerspelling_mode,
                number_mode,
                output,
                current_hand,
                TIMING,
                autocorrect,
                holistic,
                hands,
                _output,
                res,
                drawlandmarks,
            )

            output_text = str(output)
            output_size = cv2.getTextSize(output_text, FONT, 0.5, 2)[0]
            cv2.rectangle(
                image, (5, 0), (10 + output_size[0], 10 + output_size[1]), YELLOW, -1
            )
            cv2.putText(image, output_text, (10, 15), FONT, 0.5, BLACK, 2)

            mode_text = f"Number: {number_mode}"
            mode_size = cv2.getTextSize(mode_text, FONT, 0.5, 2)[0]
            cv2.rectangle(
                image, (5, 45), (10 + mode_size[0], 10 + mode_size[1]), YELLOW, -1
            )
            cv2.putText(image, mode_text, (10, 40), FONT, 0.5, BLACK, 2)

            frame_array.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    edit_yaml_variable(file_path, "output", output)
    edit_yaml_variable(file_path, "_output", _output)

    return av.VideoFrame.from_ndarray(image, format="bgr24")


def run_sign_detector():
    """
    Run the sign language detector.
    """
    cam = webrtc_streamer(
        key="Sign-Language-Detector",
        mode=WebRtcMode.SENDRECV,
        # rtc_configuration=RTC_CONFIGURATION,
        # rtc_configuration={
        # "iceServers": get_ice_servers(),
        # "iceTransportPolicy": "relay",
        # },
        # video_processor_factory=OpenCVVideoProcessor,
        async_processing=True,
        video_frame_callback=video_frame_callback,
    )


def main():
    """
    Main function.
    """
    st.title("Real Time Sign Language Recognition")
    st.subheader(
        'Tip: Press "k" to enable fingerspelling | "l" for number mode | "v" to clear output | "d" to draw landmarks'
    )
    run_sign_detector()


if __name__ == "__main__":
    main()
