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
letter_model = load_model(model_letter_path)
number_model = load_model(model_number_path)

# Load maps
s2p_map = {k.lower(): v for k, v in load_json_file("sign_language/src/sign_to_prediction_index_map.json").items()}
p2s_map = {v: k for k, v in load_json_file("sign_language/src/sign_to_prediction_index_map.json").items()}
encoder = lambda x: s2p_map.get(x.lower())
decoder = lambda x: p2s_map.get(x)

# Load TFLite model
models_path = ['sign_language/models/islr-fp16-192-8-seed_all42-foldall-last.h5']
models = [get_model() for _ in models_path]
for model, path in zip(models, models_path):
    model.load_weights(path)
tflite_keras_model = TFLiteModel(islr_models=models)
sequence_data = []


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


def handle_key_press(key):
    global output, saveGIF, saveVDO, numberMode, fingerspellingmode, draw_landmarks_flag

    # Press 'Esc' to quit
    if key == 27:
        return False

    # Press 'Backspace' to delete last word
    elif key == 8:
        output.pop()

    elif key == ord(' '):
        fingerspellingmode = not fingerspellingmode

    elif key == ord('d'):
        draw_landmarks_flag = not draw_landmarks_flag    

    # Press 's' to save result
    elif key == ord('s'):
        saveGIF = True
        saveVDO = True
        return False
        

    # Press 'm' to change mode between alphabet and number
    elif key == ord('m'):
        if fingerspellingmode:
            numberMode = not numberMode

    # Press 'c' to clear output
    elif key == ord('c'):
        output.clear()

    return True


def process_frame(image, fingerspellingmode,numberMode, output, current_hand, TIMING, autocorrect,holistic,hands,_output,res):
    global letter_model, number_model, tflite_keras_model, sequence_data, draw_landmarks_flag

    if fingerspellingmode:
        try:
            from fingerspellinginference import recognize_fingerpellings
            image, current_hand, output, _output = recognize_fingerpellings(image, numberMode, letter_model,
                                                                            number_model, hands, current_hand, output,


        except Exception as error:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            print(f"{error}, line {exc_tb.tb_lineno}")
    else:
        try:
            from glossinference import getglosses
            image, sequence_data = getglosses(output, decoder, tflite_keras_model, sequence_data, holistic, image,res,draw_landmarks_flag)

        except Exception as error:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            print(f"{error}, line {exc_tb.tb_lineno}")

    return image, output, current_hand


def process_input(opt):
    global saveGIF, saveVDO, TIMING, autocorrect, numberMode, fingerspellingmode, output, _output, draw_landmarks_flag

    saveGIF = opt.gif
    saveVDO = opt.video
    source = opt.source
    TIMING = opt.timing
    autocorrect = opt.autocorrect
    numberMode = False
    fingerspellingmode = False
    draw_landmarks_flag = False
    _output = [[], []]
    output = []
    
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


def main():
    opt = parse_opt()
    video_path, fps, webcam_width, webcam_height = process_input(opt)

    global output, _output
    
    _output = [[], []]
    output = []

    frame_array = []
    current_hand = 0
    res = []

    capture = cv2.VideoCapture(video_path, cv2.CAP_DSHOW)
    desired_fps = 24

    # Get the current frame rate
    current_fps = capture.get(cv2.CAP_PROP_FPS)

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        with mp_hands.Hands(min_detection_confidence=min_detection_confidence,
                            min_tracking_confidence=min_tracking_confidence, max_num_hands=MAX_HANDS) as hands:
            while capture.isOpened():
                success, image = capture.read()
                if not success:
                    if video_path == 0:
                        print("Ignoring empty camera frame.")
                        continue
                    else:
                        print("Video ends.")
                        break

                if video_path == 0:
                    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
                else:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    frame_width = int(capture.get(3))
                    frame_height = int(capture.get(4))

                image, output, current_hand = process_frame(image, fingerspellingmode,numberMode, output, current_hand, TIMING,
                                                             autocorrect,holistic,hands,_output,res)

                output_text = str(output)
                output_size = cv2.getTextSize(output_text, FONT, 0.5, 2)[0]
                cv2.rectangle(image, (5, 0), (10 + output_size[0], 10 + output_size[1]), YELLOW, -1)
                cv2.putText(image, output_text, (10, 15), FONT, 0.5, BLACK, 2)

                mode_text = f"Number: {numberMode}"
                mode_size = cv2.getTextSize(mode_text, FONT, 0.5, 2)[0]
                cv2.rectangle(image, (5, 45), (10 + mode_size[0], 10 + mode_size[1]), YELLOW, -1)
                cv2.putText(image, mode_text, (10, 40), FONT, 0.5, BLACK, 2)

                cv2.imshow('American Sign Language', image)
                frame_array.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

                key = cv2.waitKey(5) & 0xFF

                if not handle_key_press(key):
                    break
                
                # Get the current frame rate
                current_fps = capture.get(cv2.CAP_PROP_FPS)

                # Calculate the ratio to adjust frame rate
                ratio = current_fps / desired_fps

                # Delay to match the desired frame rate
                if cv2.waitKey(int(1000 / desired_fps)) & 0xFF == ord('q'):
                    break


    cv2.destroyAllWindows()
    capture.release()

    print(f"Gesture Recognition:\n{' '.join(output)}")

    if saveGIF == True:
        print(f"Saving GIF Result..")
        save_gif(
            frame_array, fps=fps,
            output_dir="./assets/result_ASL.gif"
        )

    if saveVDO == True:
        print(f"Saving Video Result..")
        if video_path == 0:
            width = webcam_width
            height = webcam_height
        else:
            width = frame_width
            height = frame_height
        save_video(
            frame_array, fps=fps,
            width=width, height=height,
            output_dir="./assets/result_ASL.mp4"
        )


if __name__ == '__main__':
    main()
