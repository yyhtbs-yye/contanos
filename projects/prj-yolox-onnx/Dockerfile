FROM yyhtbs/contanos:base-onnx-gpu-contanos

ENV IN_RTSP="rtsp://192.168.200.206:8554,topic=mystream"
ENV OUT_MQTT="mqtt://192.168.200.206:1883,topic=yolox,qos=2,queue_max_len=50"
ENV DEVICES="cuda:0"
ENV MODEL_INPUT_SIZE="640 640"
ENV BACKEND="onnxruntime"

SHELL ["/bin/bash", "-lc"]
RUN conda init bash
RUN conda activate onnx

RUN conda activate onnx && pip install --no-cache-dir \
        opencv-contrib-python==4.11.0.86 \
        paho-mqtt==2.1.0 \
        tqdm

WORKDIR /app

RUN conda activate onnx && pip install git+https://github.com/yyhtbs-yye/rtmlib_copy.git

COPY yolox_main.py .
COPY yolox_worker.py .

CMD ["bash", "-lc", "conda activate onnx && \
                     python yolox_main.py \
                         --in_rtsp \"${IN_RTSP}\" \
                         --out_mqtt \"${OUT_MQTT}\" \
                         --devices \"${DEVICES}\" \
                         --model_input_size \"${MODEL_INPUT_SIZE}\" \
                         --backend \"${BACKEND}\" \
"]
