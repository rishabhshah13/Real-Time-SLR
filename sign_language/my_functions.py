import mediapipe as mp
import cv2
import numpy as np


# class CFG:
#     """
#     Configuration class for ASL recognition.

#     Attributes:
#         sequence_length (int): Length of the sequence used for recognition.
#         rows_per_frame (int): Number of rows per frame in the image.
#     """
#     sequence_length = SEQ_LEN
#     rows_per_frame = ROWS_PER_FRAME


mp_holistic = mp.solutions.holistic # holistic model
mp_drawing = mp.solutions.drawing_utils # drawing utilities


def draw_landmarks(image, results):
    """
    Draw the landmarks on the image.

    Args:
        image (numpy.ndarray): The input image.
        results: The landmarks detected by Mediapipe.

    Returns:
        None
    """
    # Draw landmarks for left hand
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION,
                            mp_drawing.DrawingSpec(color=(0,0,0), thickness=1, circle_radius=0),
                            mp_drawing.DrawingSpec(color=(227, 224, 113), thickness=1, circle_radius=0))
    # mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
    #                           mp_drawing.DrawingSpec(color=(0,150,0), thickness=3, circle_radius=3),
    #                           mp_drawing.DrawingSpec(color=(0,0,0), thickness=2, circle_radius=2))
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                            mp_drawing.DrawingSpec(color=(227, 224, 113), thickness=3, circle_radius=3),
                            mp_drawing.DrawingSpec(color=(227, 224, 113), thickness=2, circle_radius=2))
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                            mp_drawing.DrawingSpec(color=(227, 224, 113), thickness=3, circle_radius=3),
                            mp_drawing.DrawingSpec(color=(227, 224, 113), thickness=2, circle_radius=2))

def image_process(image, model):
    """
    Process the image and obtain sign landmarks.

    Args:
        image (numpy.ndarray): The input image.
        model: The Mediapipe holistic object.

    Returns:
        results: The processed results containing sign landmarks.
    """
    # Set the image to read-only mode
    image.flags.writeable = True
    # Convert the image from BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Process the image using the model
    results = model.process(image)
    # Set the image back to writeable mode
    image.flags.writeable = True
    # Convert the image back from RGB to BGR
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return results

def keypoint_extraction(results):
    """
    Extract the keypoints from the sign landmarks.

    Args:
        results: The processed results containing sign landmarks.

    Returns:
        keypoints (numpy.ndarray): The extracted keypoints.
    """
    # Extract the keypoints for the left hand if present, otherwise set to zeros
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]) if results.face_landmarks else np.zeros((468, 3)) * np.nan
    pose = np.array([[res.x, res.y, res.z] for res in results.pose_landmarks.landmark]) if results.pose_landmarks else np.zeros((33, 3)) * np.nan
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]) if results.left_hand_landmarks else np.zeros((21, 3)) * np.nan
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]) if results.right_hand_landmarks else np.zeros((21, 3)) * np.nan
    return np.concatenate([face, lh, pose, rh])