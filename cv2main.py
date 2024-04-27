import sys
import cv2
import argparse
import mediapipe as mp
from autocorrect import Speller
from scripts.utils import load_model, save_gif
from scripts.gloss.my_functions import *
from scripts.gloss.landmarks_extraction import load_json_file
from scripts.gloss.backbone import TFLiteModel, get_model
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
    help_text,
)


mp_holistic = mp.solutions.holistic
mp_hands = mp.solutions.hands

# Autocorrect Word
spell = Speller(lang="en")

# Load models
letter_model = load_model(model_letter_path)
number_model = load_model(model_number_path)

# Load maps
s2p_map = {k.lower(): v for k, v in load_json_file(index_map).items()}
p2s_map = {v: k for k, v in load_json_file(index_map).items()}
encoder = lambda x: s2p_map.get(x.lower())
decoder = lambda x: p2s_map.get(x)

# Load TFLite model
models = [get_model() for _ in gloss_models_path]
for model, path in zip(models, gloss_models_path):
    model.load_weights(path)
tflite_keras_model = TFLiteModel(islr_models=models)
sequence_data = []


def parse_opt():
    """
    Parse command line arguments.

    Returns:
    - opt (argparse.Namespace): Parsed arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s", "--source", type=str, default=None, help="Video Path/0 for Webcam"
    )
    parser.add_argument(
        "-a", "--autocorrect", action="store_true", help="Autocorrect Misspelled Word"
    )
    parser.add_argument("-g", "--gif", action="store_true", help="Save GIF Result")
    parser.add_argument("-t", "--timing", type=int, default=8, help="Timing Threshold")
    parser.add_argument("-wi", "--width", type=int, default=800, help="Webcam Width")
    parser.add_argument("-he", "--height", type=int, default=600, help="Webcam Height")
    parser.add_argument("-f", "--fps", type=int, default=30, help="Webcam FPS")
    opt = parser.parse_args()
    return opt


def handle_key_press(key):
    """
    Handle key presses.

    Args:
    - key (int): Key code of the pressed key.

    Returns:
    - bool: True if program should continue, False if program should stop.
    """
    global output, saveGIF, number_mode, fingerspelling_mode, draw_landmarks_flag

    # Press 'Esc' to quit
    if key == 27:
        return False

    # Press 'Backspace' to delete last word
    elif key == 8:
        output.pop()

    elif key == ord("k"):
        fingerspelling_mode = not fingerspelling_mode

    elif key == ord("d"):
        draw_landmarks_flag = not draw_landmarks_flag

    # Press 's' to save result
    elif key == ord("s"):
        saveGIF = False
        return False

    # Press 'm' to change mode between alphabet and number
    elif key == ord("l"):
        if fingerspelling_mode:
            number_mode = not number_mode

    # Press 'c' to clear output
    elif key == ord("v"):
        output.clear()

    return True


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
):
    """
    Process each frame of the video or webcam feed.

    Args:
    - image (numpy array): Input image.
    - fingerspelling_mode (bool): Whether the fingerspelling mode is active.
    - number_mode (bool): Whether the number recognition mode is active.
    - output (list): List of recognized words or gestures.
    - current_hand (int): Number of hands detected in the previous frame.
    - TIMING (int): Timing threshold for recognizing gestures.
    - autocorrect (bool): Whether to autocorrect misspelled words.
    - holistic: MediaPipe Holistic model.
    - hands: MediaPipe Hands model.
    - _output (list): List of character sequences predicted from hand gestures.
    - res (list): List of recognized glosses.

    Returns:
    - image (numpy array): Processed image.
    - output (list): Updated list of recognized words or gestures.
    - current_hand (int): Updated number of hands detected in the current frame.
    """
    global letter_model, number_model, tflite_keras_model, sequence_data, draw_landmarks_flag

    if fingerspelling_mode:
        try:
            from scripts.inference.fingerspellinginference import (
                recognize_fingerpellings,
            )

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
                draw_landmarks_flag,
            )
        except Exception as error:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            print(f"{error}, line {exc_tb.tb_lineno}")
    else:
        try:
            from scripts.inference.glossinference import getglosses

            image, sequence_data = getglosses(
                output,
                decoder,
                tflite_keras_model,
                sequence_data,
                holistic,
                image,
                res,
                draw_landmarks_flag,
            )

        except Exception as error:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            print(f"{error}, line {exc_tb.tb_lineno}")

    return image, output, current_hand


def process_input(opt):
    """
    Process input arguments.

    Args:
    - opt (argparse.Namespace): Parsed arguments.

    Returns:
    - video_path: Video path or webcam index.
    - fps (int): Frame rate.
    - webcam_width (int): Webcam width.
    - webcam_height (int): Webcam height.
    """
    global saveGIF, TIMING, autocorrect, number_mode, fingerspelling_mode, output, _output, draw_landmarks_flag

    saveGIF = opt.gif
    source = opt.source
    TIMING = opt.timing
    autocorrect = opt.autocorrect
    number_mode = False
    fingerspelling_mode = False
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
    """
    Main function to run the ASL recognition system.
    """
    opt = parse_opt()
    video_path, fps, _, _ = process_input(opt)

    global output, _output

    _output = [[], []]
    output = []

    frame_array = []
    current_hand = 0
    res = []

    capture = cv2.VideoCapture(video_path, cv2.CAP_DSHOW)
    if capture.read()[0] is False:
        capture = cv2.VideoCapture(video_path)

    desired_fps = 24

    # Get the current frame rate
    current_fps = capture.get(cv2.CAP_PROP_FPS)

    with mp_holistic.Holistic(
        min_detection_confidence=0.5, min_tracking_confidence=0.5
    ) as holistic:
        with mp_hands.Hands(
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            max_num_hands=MAX_HANDS,
        ) as hands:
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

                image, output, current_hand = process_frame(
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
                )

                output_text = str(output)
                output_size = cv2.getTextSize(output_text, FONT, 0.5, 2)[0]
                cv2.rectangle(
                    image,
                    (5, 0),
                    (10 + output_size[0], 10 + output_size[1]),
                    YELLOW,
                    -1,
                )
                cv2.putText(image, output_text, (10, 15), FONT, 0.5, BLACK, 2)

                mode_text = f"Number: {number_mode}"
                mode_size = cv2.getTextSize(mode_text, FONT, 0.5, 2)[0]
                cv2.rectangle(
                    image, (5, 45), (10 + mode_size[0], 10 + mode_size[1]), YELLOW, -1
                )
                cv2.putText(image, mode_text, (10, 40), FONT, 0.5, BLACK, 2)

                cv2.putText(image, help_text, (10, 70), FONT, 0.5, BLACK, 2)

                cv2.imshow("American Sign Language", image)

                frame_array.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

                key = cv2.waitKey(5) & 0xFF

                if not handle_key_press(key):
                    break

                # Delay to match the desired frame rate
                if cv2.waitKey(int(1000 / desired_fps)) & 0xFF == ord("q"):
                    break

    cv2.destroyAllWindows()
    capture.release()

    print(f"Gesture Recognition:\n{' '.join(output)}")

    if saveGIF is True:
        print(f"Saving GIF Result..")
        save_gif(frame_array, fps=fps, output_dir="./assets/result_ASL.gif")


if __name__ == "__main__":
    main()
