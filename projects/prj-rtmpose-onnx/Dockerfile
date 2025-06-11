FROM yyhtbs/contanos:base-onnx-gpu-contanos

ENV IN_RTSP="rtsp://192.168.200.206:8554,topic=mystream"
ENV IN_MQTT="mqtt://192.168.200.206:1883,topic=yolox,qos=2,queue_max_len=100"
ENV OUT_MQTT="mqtt://192.168.200.206:1883,topic=rtmpose,qos=2,queue_max_len=50"
ENV DEVICES="cuda:3,cuda:4,cuda:5"
ENV MODEL_INPUT_SIZE="192 256"
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

COPY rtmpose_main.py .
COPY rtmpose_worker.py .

CMD ["bash", "-lc", "conda activate onnx && \
                     python rtmpose_main.py \
                         --in_rtsp \"${IN_RTSP}\" \
                         --in_mqtt \"${IN_MQTT}\" \
                         --out_mqtt \"${OUT_MQTT}\" \
                         --devices \"${DEVICES}\" \
                         --model_input_size \"${MODEL_INPUT_SIZE}\" \
                         --backend \"${BACKEND}\" \
"]
