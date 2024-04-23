FROM python:3.9
COPY . /app
WORKDIR /app
RUN apt-get update ##[edited]
RUN apt-get install ffmpeg libsm6 libxext6 libhdf5-dev -y
RUN pip install -r requirements.txt
RUN mkdir ~/.streamlit  
WORKDIR /app
EXPOSE 8080
CMD ["streamlit", "run", "--server.port", "8080", "st_app.py"]