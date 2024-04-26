import cv2
import numpy as np
from scripts.gloss.my_functions import *
from scripts.gloss.landmarks_extraction import mediapipe_detection, draw, extract_coordinates 
from config.config import SEQ_LEN, THRESH_HOLD

def getglosses(output, decoder, tflite_keras_model, sequence_data, holistic, image, res, draw_landmarks_flag):
    """
    Get glosses from the detected landmarks.

    Args:
    - output (list): List of recognized glosses.
    - decoder (function): Decoder function to convert model predictions to glosses.
    - tflite_keras_model: TensorFlow Lite model for gloss recognition.
    - sequence_data (list): List to store sequences of landmarks.
    - holistic: MediaPipe Holistic model.
    - image (numpy array): Input image.
    - res (list): List of recognized glosses.
    - draw_landmarks_flag (bool): Whether to draw landmarks on the image.

    Returns:
    - image (numpy array): Image with drawn landmarks and recognized glosses.
    - sequence_data (list): Updated sequence data.
    """
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

    # Detect landmarks using MediaPipe Holistic
    image, results = mediapipe_detection(image, holistic)

    # Draw landmarks if required
    if draw_landmarks_flag:
        draw(image, results)
                        
    try:
        # Extract landmark coordinates
        landmarks = extract_coordinates(results)
    except:
        landmarks = np.zeros((468 + 21 + 33 + 21, 3))

    sequence_data.append(landmarks)
                        
    sign = ""
                        
    # Generate the prediction for the given sequence data.
    if len(sequence_data) % SEQ_LEN == 0:
        prediction = tflite_keras_model(np.array(sequence_data, dtype=np.float32))["outputs"]

        if np.max(prediction.numpy(), axis=-1) > THRESH_HOLD:
            sign = np.argmax(prediction.numpy(), axis=-1)
                            
        sequence_data = []
                        
    image = cv2.flip(image, 1)
    cv2.putText(image, f"{len(sequence_data)}", (image.shape[1] - cv2.getTextSize(f"{len(sequence_data)}",
                                            cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0][0] - 3, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                        
    image = cv2.flip(image, 1)

    # if sign != "" and decoder(sign) not in res:
    if sign != "" and decoder(sign) not in res:
        # res.insert(0, decoder(sign))
        output.insert(0, decoder(sign))
                        
    image = cv2.flip(image, 1)
    
    return image, sequence_data
