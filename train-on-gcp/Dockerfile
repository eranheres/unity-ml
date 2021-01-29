FROM python:3.7.9

RUN apt-get update
RUN apt-get install -y xvfb
RUN python -m pip install --upgrade pip

COPY docker_requirements.txt requirements.txt
RUN pip install -r requirements.txt

RUN mkdir /mnt/pwd
WORKDIR /mnt/pwd

# tensorboard
EXPOSE 6006

ENTRYPOINT ["/bin/bash"]
# xvfb-run --auto-servernum --server-args='-screen 0 640x480x24' mlagents-learn $(pwd)/Assets/config/rollerball_config.yaml --env=$(pwd)/builds/rollerball_linux.x86_64 --run-id=run1

