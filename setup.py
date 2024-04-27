from setuptools import setup, find_packages

setup(
    name="your_project_name",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "autocorrect==2.6.1",
        "av==11.0.0",
        "imageio==2.14.0",
        "matplotlib==3.8.4",
        "mediapipe==0.10.9",
        "numpy==1.26.4",
        "opencv-contrib-python==4.9.0.80",
        "opencv-python==4.9.0.80",
        "opencv-python-headless==4.9.0.80",
        "PyYAML==6.0.1",
        "scikit-learn==1.2.0",
        "streamlit==1.28.2",
        "streamlit-shortcuts==0.1.1",
        "streamlit-webrtc==0.47.6",
        "tensorflow==2.15.0",
        "tensorflow-macos==2.15.0",
        "twilio==9.0.5",
    ],
    entry_points={
        "console_scripts": [
            "your_command_name = your_package_name.your_module_name:main_function_name",
        ],
    },
)
