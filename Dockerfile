FROM python:3.7-slim
COPY . /root
WORKDIR /root
RUN pip install flask gunicorn numpy sklearn scipy pandas flask-cors
RUN pip install -U flask-cors