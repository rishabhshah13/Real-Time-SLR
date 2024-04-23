import cv2
import numpy as np
from utils import calc_landmark_list, draw_landmarks as draw_landmarks_hands, draw_info_text
from autocorrect import Speller
from config import *


# Autocorrect Word
spell = Speller(lang='en')

def get_output(idx,_output,output,TIMING, autocorrect):

    # global _output, output, autocorrect, TIMING

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


def recognize_fingerpellings(image, numberMode, letter_model, number_model, hands, current_hand, output, _output,TIMING, autocorrect):

    
    # global mp_drawing 
    # global output, _output

    image.flags.writeable = False
    results = hands.process(image)

    # Draw the hand annotations on the image
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


    multi_hand_landmarks = results.multi_hand_landmarks
    multi_handedness = results.multi_handedness

    # Load Classification Model
    # letter_model = load_model(model_letter_path)
    # number_model = load_model(model_number_path)

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

            # mp_drawing.draw_landmarks(image, current_select_hand, mp_hands.HAND_CONNECTIONS)
            landmark_list = calc_landmark_list(image, current_select_hand)
            image = draw_landmarks_hands(image, landmark_list)

            # Get (x, y) coordinates of hand landmarks
            x_values = [lm.x for lm in current_select_hand.landmark]
            y_values = [lm.y for lm in current_select_hand.landmark]

            # Get Minimum and Maximum Values
            min_x = int(min(x_values) * w)
            max_x = int(max(x_values) * w)
            min_y = int(min(y_values) * h)
            max_y = int(max(y_values) * h)

            # Draw Text Information
            cv2.putText(image, f"Hand No. #{idx}", (min_x - 10, max_y + 30), FONT, 0.5, GREEN, 2)
            cv2.putText(image, f"{handness} Hand", (min_x - 10, max_y + 60), FONT, 0.5, GREEN, 2)

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
            cv2.rectangle(image, (min_x - 20, min_y - 10), (max_x + 20, max_y + 10), BLACK, 4)
            image = draw_info_text(image, [min_x - 20, min_y - 10, max_x + 20, max_y + 10], gesture)

            _gesture.append(gesture)

    # Number of hands is decreasing, create "SPACE"
    if isDecreased == True:
        if current_hand == 1:
            get_output(0,_output,output,TIMING, autocorrect)

    # Number of hands is the same, append gesture
    else:
        if results.multi_hand_landmarks is not None:
            _output[0].append(_gesture[0])

    # Track hand numbers
    if results.multi_hand_landmarks:
        current_hand = len(multi_hand_landmarks)
    else:
        current_hand = 0

    return image, current_hand,output, _output