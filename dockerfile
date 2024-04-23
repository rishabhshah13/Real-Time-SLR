FROM python:3.9
COPY . /app
WORKDIR /app
RUN apt-get update ##[edited]
RUN apt-get install ffmpeg libsm6 libxext6 libhdf5-dev -y
RUN pip install -r requirements.txt
EXPOSE 8501
RUN mkdir ~/.streamlit  
WORKDIR /app
ENTRYPOINT ["streamlit", "run", "app.py"]