# #FROM python:3.9
# FROM continuumio/miniconda3

# #Directory of your new application in the container
# WORKDIR /home/app 

# RUN apt-get update
# RUN apt-get install nano unzip
# RUN apt install curl -y
# #RUN curl -fsSL https://get.deta.dev/cli.sh | sh

# COPY requirements.txt /dependencies/requirements.txt
# RUN pip install -r /dependencies/requirements.txt

# COPY . /home/app

# CMD streamlit run --server.port 5000 heart_dashboard.py

FROM python:3.9-slim

# Set working directory
WORKDIR /home/app

# Install required packages
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    nano \
    unzip \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt /dependencies/requirements.txt
RUN pip install --no-cache-dir -r /dependencies/requirements.txt

# Copy application files
COPY . /home/app

# Set command to run Streamlit app
CMD streamlit run --server.port 5000 heart_dashboard.py
