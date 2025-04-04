# Use an official Python runtime as a parent image
FROM python:3.11-slim-buster

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y gcc python3-dev && rm -rf /var/lib/apt/lists/*

COPY . /app

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

CMD ["python3", "app.py"]