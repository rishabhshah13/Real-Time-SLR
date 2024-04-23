from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import time
import mediapipe as mp
from src.backbone import TFLiteModel, get_model
from src.landmarks_extraction import mediapipe_detection, draw, extract_coordinates, load_json_file 
from src.config import SEQ_LEN, THRESH_HOLD
import base64


app = Flask(__name__)
CORS(app)

mp_holistic = mp.solutions.holistic 
mp_drawing = mp.solutions.drawing_utils

s2p_map = {k.lower():v for k,v in load_json_file("src/sign_to_prediction_index_map.json").items()}
p2s_map = {v:k for k,v in load_json_file("src/sign_to_prediction_index_map.json").items()}
encoder = lambda x: s2p_map.get(x.lower())
decoder = lambda x: p2s_map.get(x)

models_path = [
                './models/islr-fp16-192-8-seed_all42-foldall-last.h5',
]
models = [get_model() for _ in models_path]

# Load weights from the weights file.
for model,path in zip(models,models_path):
    model.load_weights(path)

res = []
sequence_data = []
tflite_keras_model = TFLiteModel(islr_models=models)
cap = cv2.VideoCapture(0)

def process_frame(frame):

    global sequence_data
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:

        image, results = mediapipe_detection(frame, holistic)
        draw(image, results)
        
        try:
            landmarks = extract_coordinates(results)
        except:
            landmarks = np.zeros((468 + 21 + 33 + 21, 3))
        sequence_data.append(landmarks)
        
        sign = ""
        
        # Generate the prediction for the given sequence data.
        if len(sequence_data) % SEQ_LEN == 0:
            prediction = tflite_keras_model(np.array(sequence_data, dtype = np.float32))["outputs"]

            if np.max(prediction.numpy(), axis=-1) > THRESH_HOLD:
                sign = np.argmax(prediction.numpy(), axis=-1)
            
            sequence_data = []
        
        image = cv2.flip(image, 1)
        
        cv2.putText(image, f"{len(sequence_data)}", (3, 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        
        image = cv2.flip(image, 1)
        
        # Insert the sign in the result set if sign is not empty.
        if sign != "" and decoder(sign) not in res:
            res.insert(0, decoder(sign))
        
        # Get the height and width of the image
        height, width = image.shape[0], image.shape[1]

        # Create a white column
        white_column = np.ones((height // 8, width, 3), dtype='uint8') * 255

        # Flip the image vertically
        image = cv2.flip(image, 1)
        
        # Concatenate the white column to the image
        image = np.concatenate((white_column, image), axis=0)
        
        cv2.putText(image, f"{', '.join(str(x) for x in res)}", (3, 65),
                            cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0), 2, cv2.LINE_AA)
                        
        return image

@app.route('/', methods=['GET'])
def check():
    return 'Working'


@app.route('/process_frame_route', methods=['POST'])
def process_frame_route():
    frame_data = request.json['frame']
    print(type(frame_data))
    _, encoded_image = frame_data.split(',', 1)
    frame = base64.b64decode(encoded_image)
    frame = np.frombuffer(frame, dtype=np.uint8)
    frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)
    
    processed_frame = process_frame(frame)
    
    # Encode the processed frame in the same format as the input frame
    _, encoded_processed_frame = cv2.imencode('.jpg', processed_frame)
    processed_frame_data = base64.b64encode(encoded_processed_frame).decode('utf-8')

    return jsonify({'processed_frame': processed_frame_data})



if __name__ == '__main__':
    app.run(debug=True)
