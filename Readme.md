# Real Time Sign Language Recognition

This project is a real-time sign language recognition system that can recognize American Sign Language (ASL) gestures from a live video stream. It utilizes the MediaPipe library for hand and body pose estimation and TensorFlow Lite models for fingerspelling and gesture recognition. The application is built using Python and Streamlit for the user interface.

## Installation

To run this project, you need to follow these steps:

1. Clone the repository:

   ```bash
   git clone <repository_url>
   ```

2. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the application:

   ```bash
   streamlit run app.py
   ```

## Usage

The application allows users to perform real-time sign language recognition using a webcam or a video file. Here are the main features:

- **Fingerspelling Mode:** Toggle between fingerspelling mode and gesture recognition mode by pressing the "fingerspelling" button or using the "k" key.
  
- **Number Mode:** Switch between recognizing letters and numbers by pressing the "number mode" button or using the "l" key.
  
- **Clear Output:** Clear the recognized sign output by pressing the "clear output" button or using the "v" key.
  
- **Draw Landmarks:** Toggle the visibility of hand landmarks by pressing the "draw landmarks" button or using the "d" key.

## Configuration

The application can be configured using a YAML file (`config.yaml`) located in the root directory. You can modify the following parameters:

- `fingerspellingmode`: Set to `True` for fingerspelling mode and `False` for gesture recognition mode.
  
- `numberMode`: Set to `True` for number mode and `False` for letter mode.
  
- `draw_landmarks_flag`: Set to `True` to draw hand landmarks on the video stream.

## Acknowledgments

This project utilizes the following libraries and tools:

- [MediaPipe](https://google.github.io/mediapipe/) for hand and body pose estimation.
  
- [TensorFlow Lite](https://www.tensorflow.org/lite) for running lightweight machine learning models.
  
- [Streamlit](https://streamlit.io/) for building interactive web applications with Python.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.