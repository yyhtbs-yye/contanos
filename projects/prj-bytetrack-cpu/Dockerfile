FROM python:3.9-slim

ENV IN_MQTT="mqtt://192.168.200.206:1883,topic=yolox,qos=2,buffer_threshold=100" 
ENV OUT_MQTT="mqtt://192.168.200.206:1883,topic=bytetrack,qos=2,queue_max_len=100" 
ENV DEVICES="cpu"
    
RUN apt update && apt install -y git

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      libgl1-mesa-glx \ 
      libglib2.0-0

WORKDIR /app

RUN pip install git+https://github.com/yyhtbs-yye/boxmat_lite

COPY bytetrack_main.py .
COPY bytetrack_worker.py .

CMD ["bash", "-lc", "\
    python bytetrack_main.py \
      --in_mqtt \"${IN_MQTT}\" \
      --out_mqtt \"${OUT_MQTT}\" \
      --devices \"${DEVICES}\" \
"]