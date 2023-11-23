# Use an official Python runtime as a parent image
FROM python:3.10

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Run the script to save the model to BentoML storage
RUN python ./savemodeltobento.py

# Change directory to the 'service' folder
WORKDIR /app/service

# Build BentoML service
# RUN bentoml build

# Run BentoML service
CMD bentoml build && bentoml serve service:wave_arrival_detector
