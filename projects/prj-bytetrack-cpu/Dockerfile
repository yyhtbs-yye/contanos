
# Base image: Nvidia PyTorch https://ngc.nvidia.com/catalog/containers/nvidia:pytorch
FROM python:3.9-slim

# Environment variables matching the script's argument parser
ENV IN_MQTT="mqtt://192.168.200.206:1883,topic=yolox,qos=2,buffer_threshold=100" 
ENV OUT_MQTT="mqtt://192.168.200.206:1883,topic=bytetrack,qos=2,queue_max_len=100" 
ENV DEVICES="cpu"
    
# Update and install necessary packages
RUN apt update && apt install -y git

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      libgl1-mesa-glx \ 
      libglib2.0-0

# Set the parent working directory
WORKDIR /app

# Clone the repository with submodules into a subdirectory 'boxmot'
RUN git clone https://github.com/yyhtbs-yye/boxmat_lite

RUN cd boxmat_lite && pip install -r requirements.txt \
    && pip install -e .

COPY bytetrack_main.py .
COPY bytetrack_worker.py .

CMD ["bash", "-lc", "\
    python bytetrack_main.py \
      --in_mqtt \"${IN_MQTT}\" \
      --out_mqtt \"${OUT_MQTT}\" \
      --devices \"${DEVICES}\" \
"]