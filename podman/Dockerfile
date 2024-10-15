# syntax=docker/dockerfile:1.4
FROM python:3.9-slim

RUN apt-get update
RUN apt-get install git -y
RUN apt-get install ffmpeg libsm6 libxext6 -y

# Info of device run
ENV GIT_BRANCH
ENV MODEL_NAME
ENV COMPUTER_NAME


# Tracking
ENV TELEGRAM_BOT_TOKEN
ENV CHAT_ID
ENV API_WANDB_TOKEN

# Training configs
ENV BATCH_SIZE 1024
ENV LEARNING_RATE 0.000012
ENV GAMMA 0.8
ENV NUM_ENVS 8

WORKDIR /
RUN git clone --depth=1 -b main https://github.com/htrbao/PPO_M3_Simulator
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu118

WORKDIR /PPO_M3_Simulator/gym-match3
RUN pip install --no-cache-dir -e .

WORKDIR /PPO_M3_Simulator
RUN pip install --no-cache-dir -r requirements.txt

WORKDIR /PPO_M3_Simulator
RUN cp info_config.py.example info_config.py

COPY run.sh .
CMD ["sh", "run.sh"]
