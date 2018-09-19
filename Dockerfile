# Environment to run pytorch
#
# Author: John Inacay

# Use base images from nvidia cuda images
FROM nvidia/cuda:9.2-cudnn7-runtime

# Install python3 for virtualenv
RUN apt-get update -qq && apt-get install -y \
    python3-venv \
    python3-pip

# Setup virtualenv
ENV WORKENV_NAME=VideoGAN
RUN python3 -m venv ${WORKENV_NAME}
RUN ${WORKENV_NAME}/bin/pip3 install --upgrade pip

# Install pytorch
RUN ${WORKENV_NAME}/bin/pip3 install http://download.pytorch.org/whl/cu92/torch-0.4.1-cp35-cp35m-linux_x86_64.whl
RUN ${WORKENV_NAME}/bin/pip3 install torchvision

# Install other dependencies.
RUN ${WORKENV_NAME}/bin/pip3 install -y \
    matplotlib \
    scikit-image
