import cv2
import numpy as np
from utils import calc_landmark_list, draw_landmarks as draw_landmarks_hands, draw_info_text
from autocorrect import Speller
from config import *

# Autocorrect Word
spell = Speller(lang='en')

def get_output(idx, _output, output, TIMING, autocorrect):
    """
    Get output from predicted characters.
    
    Args:
    - idx (int): Index of the output list.
    - _output (list): List of character sequences predicted from hand gestures.
    - output (list): List of recognized words.
    - TIMING (int): Threshold for timing to consider a character.
    - autocorrect (bool): Whether to autocorrect misspelled words.
    
    Returns:
    - None
    """
    key = []
    for i in range(len(_output[idx])):
        character = _output[idx][i]
        counts = _output[idx].count(character)

        # Add character to key if it exceeds 'TIMING THRESHOLD'
        if (character not in key) or (character != key[-1]):
            if counts > TIMING:
                key.append(character)

    # Add key character to output text
    text = ""
    for character in key:
        if character == "?":
            continue
        text += str(character).lower()

    # Autocorrect Misspelled Word
    text = spell(text) if autocorrect else text

    # Add word to output list
    if text != "":
        _output[idx] = []
        output.append(text.title())
    return None

def recognize_fingerpellings(image, numberMode, letter_model, number_model, hands, current_hand, output, _output, TIMING, autocorrect, draw_landmarks_flag):
    """
    Recognize finger spellings from the given image.

    Args:
    - image (numpy array): Input image.
    - numberMode (bool): Whether to recognize numbers.
    - letter_model: Model for recognizing letters.
    - number_model: Model for recognizing numbers.
    - hands: MediaPipe Hands model.
    - current_hand (int): Number of hands in the previous frame.
    - output (list): List of recognized words.
    - _output (list): List of character sequences predicted from hand gestures.
    - TIMING (int): Threshold for timing to consider a character.
    - autocorrect (bool): Whether to autocorrect misspelled words.
    - draw_landmarks_flag (bool): Whether to draw landmarks on the image.

    Returns:
    - image (numpy array): Image with recognized gestures and landmarks.
    - current_hand (int): Number of hands in the current frame.
    - output (list): Updated list of recognized words.
    - _output (list): Updated list of character sequences predicted from hand gestures.
    """
    results = hands.process(image)

    # Draw the hand annotations on the image
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    multi_hand_landmarks = results.multi_hand_landmarks
    multi_handedness = results.multi_handedness

    _gesture = []
    data_aux = []

    # Number of hands
    isIncreased = False
    isDecreased = False

    if current_hand != 0:
        if results.multi_hand_landmarks is None:
            isDecreased = True
        else:
            if len(multi_hand_landmarks) > current_hand:
                isIncreased = True
            elif len(multi_hand_landmarks) < current_hand:
                isDecreased = True

    if results.multi_hand_landmarks:
        h, w, _ = image.shape
        for idx in reversed(range(len(multi_hand_landmarks))):
            current_select_hand = multi_hand_landmarks[idx]
            handness = multi_handedness[idx].classification[0].label

            # Get (x, y) coordinates of hand landmarks
            x_values = [lm.x for lm in current_select_hand.landmark]
            y_values = [lm.y for lm in current_select_hand.landmark]

            # Get Minimum and Maximum Values
            min_x = int(min(x_values) * w)
            max_x = int(max(x_values) * w)
            min_y = int(min(y_values) * h)
            max_y = int(max(y_values) * h)

            # Flip Left Hand to Right Hand
            if handness == 'Left':
                x_values = list(map(lambda x: 1 - x, x_values))
                min_x -= 10

            # Create Data Augmentation for Corrected Hand
            for i in range(len(current_select_hand.landmark)):
                data_aux.append(x_values[i] - min(x_values))
                data_aux.append(y_values[i] - min(y_values))

            if not numberMode:
                # Alphabets Prediction
                prediction = letter_model.predict([np.asarray(data_aux)])
                gesture = str(prediction[0]).title()
                gesture = gesture if gesture != 'Unknown_Letter' else '?'
            else:
                # Numbers Prediction
                prediction = number_model.predict([np.asarray(data_aux)])
                gesture = str(prediction[0]).title()
                gesture = gesture if gesture != 'Unknown_Number' else '?'

            # Draw Bounding Box
            if draw_landmarks_flag:
                cv2.rectangle(image, (min_x - 20, min_y - 10), (max_x + 20, max_y + 10), BLACK, 4)
                image = draw_info_text(image, [min_x - 20, min_y - 10, max_x + 20, max_y + 10], gesture)

            _gesture.append(gesture)

    # Number of hands is decreasing, create "SPACE"
    if isDecreased == True:
        if current_hand == 1:
            get_output(0, _output, output, TIMING, autocorrect)

    # Number of hands is the same, append gesture
    else:
        if results.multi_hand_landmarks is not None:
            _output[0].append(_gesture[0])

    # Track hand numbers
    if results.multi_hand_landmarks:
        current_hand = len(multi_hand_landmarks)
    else:
        current_hand = 0

    return image, current_hand, output, _output
