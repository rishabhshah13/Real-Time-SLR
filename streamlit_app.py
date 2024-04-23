# # %%

from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import time
import mediapipe as mp
from sign_language.src.backbone import TFLiteModel, get_model
from sign_language.src.landmarks_extraction import mediapipe_detection, draw, extract_coordinates, load_json_file 
from sign_language.src.config import SEQ_LEN, THRESH_HOLD
import streamlit as st
import threading
import numpy as np
import mediapipe as mp
import cv2
from sign_language.my_functions import *
from streamlit_webrtc import RTCConfiguration, WebRtcMode, webrtc_streamer
import av

import sys
import cv2
import argparse
import numpy as np
import mediapipe as mp
from autocorrect import Speller
from utils import load_model, save_gif, save_video
from sign_language.my_functions import *
from sign_language.src.landmarks_extraction import load_json_file
from sign_language.src.backbone import TFLiteModel, get_model
from sign_language.src.config import SEQ_LEN, THRESH_HOLD
from config import *

mp_holistic = mp.solutions.holistic
mp_hands = mp.solutions.hands

# Autocorrect Word
spell = Speller(lang='en')

# Load models
import pickle
@st.cache_resource()
def load_modela(model_path):
    with open(model_path, 'rb') as model_file:
        model_dict = pickle.load(model_file)
        model = model_dict['model']
    return model

#####################
# letter_model = load_modela(model_letter_path)
# number_model = load_modela(model_number_path)

# # letter_model = load_model(model_letter_path)
# # number_model = load_model(model_number_path)

# # Load maps
# s2p_map = {k.lower(): v for k, v in load_json_file("sign_language/src/sign_to_prediction_index_map.json").items()}
# p2s_map = {v: k for k, v in load_json_file("sign_language/src/sign_to_prediction_index_map.json").items()}
# encoder = lambda x: s2p_map.get(x.lower())
# decoder = lambda x: p2s_map.get(x)

# # Load TFLite model
# models_path = ['sign_language/models/islr-fp16-192-8-seed_all42-foldall-last.h5']
# models = [get_model() for _ in models_path]
# for model, path in zip(models, models_path):
#     model.load_weights(path)


# @st.cache_resource()
# def kmodel():
#     return TFLiteModel(islr_models=models)
# tflite_keras_model = kmodel()


# tflite_keras_model = TFLiteModel(islr_models=models)
# sequence_data = []
#####################




# Load maps
s2p_map = {k.lower(): v for k, v in load_json_file("sign_language/src/sign_to_prediction_index_map.json").items()}
p2s_map = {v: k for k, v in load_json_file("sign_language/src/sign_to_prediction_index_map.json").items()}
encoder = lambda x: s2p_map.get(x.lower())
decoder = lambda x: p2s_map.get(x)

# Load TFLite model

sequence_data = []


if "kerasmodel" not in st.session_state.keys():
    print("Loading Keras model")

    models_path = ['sign_language/models/islr-fp16-192-8-seed_all42-foldall-last.h5']
    models = [get_model() for _ in models_path]
    for model, path in zip(models, models_path):
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
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--source', type=str, default=None, help='Video Path/0 for Webcam')
    parser.add_argument('-a', '--autocorrect', action='store_true', help='Autocorrect Misspelled Word')
    parser.add_argument('-g', '--gif', action='store_true', help='Save GIF Result')
    parser.add_argument('-v', '--video', action='store_true', help='Save Video Result')
    parser.add_argument('-t', '--timing', type=int, default=8, help='Timing Threshold')
    parser.add_argument('-wi', '--width',  type=int, default=800, help='Webcam Width')
    parser.add_argument('-he', '--height', type=int, default=600, help='Webcam Height')
    parser.add_argument('-f', '--fps', type=int, default=30, help='Webcam FPS')
    opt = parser.parse_args()
    return opt



def process_input(opt):

    global saveGIF, saveVDO, TIMING, autocorrect, numberMode, fingerspellingmode

    
    
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
saveVDO = opt.video
source = opt.source
TIMING = opt.timing
autocorrect = opt.autocorrect
numberMode = False
fingerspellingmode = True
file_path = "config.yaml"

video_path, fps, webcam_width, webcam_height = process_input(opt)
_output = [[], []]
output = []
frame_array = []
current_hand = 0
res = []




def handle_key_press(key):
    global output, saveGIF, saveVDO, numberMode, fingerspellingmode

    # Press 'Esc' to quit
    if key == 27:
        return False

    # Press 'Backspace' to delete last word
    elif key == 8:
        output.pop()

    # elif key == ord(' '):
    #     fingerspellingmode = not fingerspellingmode

    # # Press 's' to save result
    # elif key == ord('s'):
    #     saveGIF = True
    #     saveVDO = True
    #     return False

    # Press 'm' to change mode between alphabet and number
    elif key == ord('m'):
        if fingerspellingmode:
            numberMode = not numberMode

    # Press 'c' to clear output
    elif key == ord('c'):
        output.clear()

    return True



import yaml

def edit_yaml_variable(file_path, variable_name, new_value):
    try:
        # Load the YAML file
        with open(file_path, "r") as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        # If file not found, create an empty config
        config = {}

    # Edit the variable
    config[variable_name] = new_value

    # Write the changes back to the YAML file
    with open(file_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)


def read_yaml_variable(file_path, variable_name):
    try:
        # Load the YAML file
        with open(file_path, "r") as f:
            config = yaml.safe_load(f)
            
        # Return the value of the variable if exists, otherwise return None
        return config.get(variable_name)
    
    except FileNotFoundError:
        # If file not found, return None
        return None



from streamlit_shortcuts import add_keyboard_shortcuts

def change_fingerspellingmode():
    global fingerspellingmode
    print("Current fingerspellingmode: ", fingerspellingmode)
    fingerspellingmode = read_yaml_variable(file_path, 'fingerspellingmode')
    fingerspellingmode = not fingerspellingmode
    edit_yaml_variable(file_path, 'fingerspellingmode', fingerspellingmode)

    if fingerspellingmode:
        st.write("Fingerspelling mode!")
    else:
        st.write("Gloss mode!")


def change_numbermodemode():
    global numberMode

    global numberMode
    print("Current numberMode: ", numberMode)
    numberMode = read_yaml_variable(file_path, 'numberMode')
    numberMode = not numberMode
    edit_yaml_variable(file_path, 'numberMode', numberMode)

def clearoutput():
    edit_yaml_variable(file_path, 'output', [])
    edit_yaml_variable(file_path, '_output', [[],[]])
    


# st.button("delete", on_click=delete_callback)
st.button("fingerspelling", on_click=change_fingerspellingmode)
st.button("numbermode", on_click=change_numbermodemode)
st.button("clearoutput", on_click=clearoutput)



# add_keyboard_shortcuts({
#     'Ctrl+Alt+k': 'fingerspelling',
#     'Ctrl+Alt+l': 'numbermode',

# })

add_keyboard_shortcuts({
    'k': 'fingerspelling',
    'l': 'numbermode',
    'v': 'clearoutput'

})


def process_frame(image, fingerspellingmode, numberMode, output, current_hand, TIMING, autocorrect,holistic,hands,_output,res):
    global letter_model, number_model, tflite_keras_model, sequence_data
    
    if fingerspellingmode:
        try:
            from fingerspellinginference import recognize_fingerpellings
            image, current_hand, output, _output = recognize_fingerpellings(image, numberMode, letter_model,
                                                                            number_model, hands, current_hand, output,
                                                                            _output, TIMING, autocorrect) 
        except Exception as error:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            print(f"{error}, line {exc_tb.tb_lineno}")
    else:
        try:
            from glossinference import getglosses
            image, sequence_data = getglosses(output, decoder, tflite_keras_model, sequence_data, holistic, image,res)

        except Exception as error:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            print(f"{error}, line {exc_tb.tb_lineno}")

    return image, output, current_hand, _output


# Deployment configuration
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)


# from twilio.rest import Client
# import os
# # Find your Account SID and Auth Token at twilio.com/console
# # and set the environment variables. See http://twil.io/secure
# account_sid = os.environ['TWILIO_ACCOUNT_SID']
# auth_token = os.environ['TWILIO_AUTH_TOKEN']
# client = Client(account_sid, auth_token)

# token = client.tokens.create()
from turn import get_ice_servers



def video_frame_callback(frame):
    

    global opt, video_path, fps, webcam_width, webcam_height, frame_array, current_hand, res #,  _output, output

    fingerspellingmode = read_yaml_variable(file_path, 'fingerspellingmode')
    numberMode = read_yaml_variable(file_path, 'numberMode')

    output = read_yaml_variable(file_path, 'output')
    _output = read_yaml_variable(file_path, '_output')

    image = frame.to_ndarray(format="bgr24")


    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        with mp_hands.Hands(min_detection_confidence=min_detection_confidence,
                            min_tracking_confidence=min_tracking_confidence, max_num_hands=MAX_HANDS) as hands:


            if video_path == 0:
                image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
            else:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            image, output, current_hand, _output = process_frame(image, fingerspellingmode,numberMode, output, current_hand, TIMING,
                                                        autocorrect,holistic,hands,_output,res)


            output_text = str(output)
            output_size = cv2.getTextSize(output_text, FONT, 0.5, 2)[0]
            cv2.rectangle(image, (5, 0), (10 + output_size[0], 10 + output_size[1]), YELLOW, -1)
            cv2.putText(image, output_text, (10, 15), FONT, 0.5, BLACK, 2)

            mode_text = f"Number: {numberMode}"
            mode_size = cv2.getTextSize(mode_text, FONT, 0.5, 2)[0]
            cv2.rectangle(image, (5, 45), (10 + mode_size[0], 10 + mode_size[1]), YELLOW, -1)
            cv2.putText(image, mode_text, (10, 40), FONT, 0.5, BLACK, 2)


            frame_array.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

            key = cv2.waitKey(5) & 0xFF

            # if not handle_key_press(key):
            #     break

    # print(f"Gesture Recognition:\n{' '.join(output)}")

    edit_yaml_variable(file_path, 'output', output)
    edit_yaml_variable(file_path, '_output', _output)
    
    # print(output)

    # image = frame
    return av.VideoFrame.from_ndarray(image,format="bgr24")
    

def run_sign_detector():

    # global numba

    # print(numba)
    # if "numbb" in st.session_state:
    #     print("Sesstopm stete", st.session_state.numbb)
    # print(numbb)

    cam = webrtc_streamer(
        key="Sign-Language-Detector",
        mode=WebRtcMode.SENDRECV,
        # rtc_configuration=RTC_CONFIGURATION,
        rtc_configuration={
        "iceServers": get_ice_servers(),
        "iceTransportPolicy": "relay",
        },
        # video_processor_factory=OpenCVVideoProcessor,
        async_processing=True,
        video_frame_callback=video_frame_callback
    )

def main():
    st.title("Real Time Sign Language to Text")
    st.header("Description of the app")
    run_sign_detector()


if __name__ == "__main__":
    main()