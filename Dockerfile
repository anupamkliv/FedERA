# Dockerfile

# Determine the system architecture
ARG ARCHITECTURE

# Use the official Python base image
FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && \
    apt-get install -y \
        git \
        protobuf-compiler

# Install PyTorch
RUN pip install torch torchvision

# Set the working directory
WORKDIR /usr/src/federa

# Copy the repository code into the container
COPY . .

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Expose the port for communication with the server
EXPOSE 8214

# Set the entry point command
ENTRYPOINT []

# Default argument values for the server
#CMD ["python", "-m", "federa.server.start_server"]

# Specify the command arguments as environment variables
ENV SERVER_ARGS=""
ENV CLIENT_ARGS=""

# Start the server and client using the provided arguments
CMD bash -c "python -m federa.server.start_server ${SERVER_ARGS} & \
              python -m federa.client.start_client ${CLIENT_ARGS}"
