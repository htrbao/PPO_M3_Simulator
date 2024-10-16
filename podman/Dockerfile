# syntax=docker/dockerfile:1.4
FROM python:3.9-slim

RUN apt-get update
RUN apt-get install git -y
RUN apt-get install ffmpeg libsm6 libxext6 -y

WORKDIR /
RUN git clone --depth=1 -b main https://github.com/htrbao/PPO_M3_Simulator
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu118

WORKDIR /PPO_M3_Simulator/gym-match3
RUN pip install --no-cache-dir -e .

WORKDIR /PPO_M3_Simulator
RUN pip install --no-cache-dir -r requirements.txt

WORKDIR /PPO_M3_Simulator
RUN mkdir logging

COPY run.sh .
ENTRYPOINT ["sh", "run.sh"]
