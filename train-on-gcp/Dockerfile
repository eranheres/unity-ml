FROM nvidia/cuda:11.2.0-base
#FROM python:3.7.9

RUN apt-get update -y
RUN apt-get install -y xvfb curl python3 python3-pip

COPY docker_requirements.txt requirements.txt
RUN pip3 install -r requirements.txt
RUN curl -sSL https://sdk.cloud.google.com | bash

RUN mkdir /mnt/pwd
# WORKDIR /mnt/pwd

COPY train_on_docker.sh train_on_docker.sh

# tensorboard
EXPOSE 6006
ENV PATH $PATH:/root/google-cloud-sdk/bin
RUN echo '[GoogleCompute]\nservice_account = default' > /etc/boto.cfg

ENTRYPOINT ["/bin/bash", "train_on_docker.sh"]

