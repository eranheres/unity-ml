FROM python:3.7.9

RUN apt-get update
RUN apt-get install -y xvfb curl
RUN python -m pip install --upgrade pip
RUN curl -sSL https://sdk.cloud.google.com | bash

COPY docker_requirements.txt requirements.txt
RUN pip install -r requirements.txt

RUN mkdir /mnt/pwd
# WORKDIR /mnt/pwd

COPY train_on_docker.sh train_on_docker.sh

# tensorboard
EXPOSE 6006
ENV PATH $PATH:/root/google-cloud-sdk/bin
RUN echo '[GoogleCompute]\nservice_account = default' > /etc/boto.cfg

ENTRYPOINT ["/bin/bash", "train_on_docker.sh"]

