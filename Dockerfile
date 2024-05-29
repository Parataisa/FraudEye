# Use the official Python 3.9.12 image as the base image
FROM python:3.9.12

# Set working directory
WORKDIR /app

# Copy requirements files
COPY requirements.build requirements.build

# Install system dependencies if needed
RUN apt-get update && apt-get install -y curl

# Install Node.js and npm
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash -
RUN apt-get install -y nodejs

# Install configurable-http-proxy
RUN npm install -g configurable-http-proxy

# Upgrade pip
RUN pip install --upgrade pip

# Install pip-tools
RUN pip install pip-tools

# Install the dependencies from requirements.txt
RUN pip install -r requirements.build

# Install JupyterHub
RUN pip install jupyterhub

# Configure JupyterHub
RUN jupyterhub --generate-config

# Create a default user
RUN useradd -m user && echo "user:passwd" | chpasswd

# Start the JupyterHub server
CMD ["jupyterhub", "-f", "/app/jupyterhub_config.py"]