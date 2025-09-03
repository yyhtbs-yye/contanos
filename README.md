**Contanos STRIDE**

This library is a copy of yyhtbs-yye (Yuhang Ye, TUS)'s STRIDE repo. 

We rename the repo to seperate the framework (Contanos Call) and implementation (STRIDE).

License is the same. 

# Acknowledgement
This work has received funding as an Open Call project under the SPARTA project from the European Union's Horizon Europe programme (grant agreement No. 101069732).

The STRIDE **README** is below:

# Sport Real-time TRacking, Identification & DEtection, on top of Contanos [STRIDE]

**STRIDE** is a modular, real‑time sports analytics pipeline for detection, tracking and pose estimation. It is **built on top of the Contanos framework** (base classes, I/O conventions, and runtime helpers) but **this repository does not vendor the Contanos code** anymore.

- **In containers**: the base Docker images used by STRIDE fetch and install `contanos-core` automatically.

- **For local debugging (outside Docker)**: clone [`contanos-core`](https://github.com/yyhtbs-yye/contanos-core) and use only the `contanos` folder (or install it as a package). A minimal approach:

  ```bash
  git clone https://github.com/yyhtbs-yye/contanos-core
  # Option A: make the package importable
  export PYTHONPATH=/path/to/contanos-core:$PYTHONPATH
  # Option B: symlink just the folder next to your scripts
  ln -s /path/to/contanos-core/contanos ./contanos
  ```

---

## Project Description

STRIDE decomposes a video analytics workload into **independent micro‑services** wired together with **MQTT topics** and **RTSP streams**:

- **Input**: an RTSP source service (or your own camera) publishes frames to the pipeline.
- **Perception**:
  - **YOLOX** (ONNX) for person detection
  - **ByteTrack** for multi‑object tracking
  - **RTMPose** (ONNX) for multi‑person 2D pose estimation
  - Optional modules such as **Jersey OCR** (GPU) and **CMC** (CPU)
- **Visualization**: an annotation/overlay service renders results and republishes an **RTSP output** you can watch in VLC/FFplay.
- **Glue**: an **MQTT broker** (Eclipse Mosquitto) and an **RTSP server** (MediaMTX) coordinate message passing and stream fan‑out.

Each service is a small container that follows the **Contanos base classes and I/O conventions** (e.g., `IN_MQTT_URL_*`, `OUT_MQTT_URL`, `IN_RTSP_URL`, `OUT_RTSP_URL`). This keeps components loosely coupled and easy to swap while enabling low‑latency, real‑time operation.

> Note: Contanos is an external dependency. STRIDE's images install `contanos-core` at build time; the repository itself intentionally excludes the Contanos sources.

## Features

- **End‑to‑end real‑time pipeline**
  - Detection → Tracking → Pose → (optional) OCR/CMC → Annotation/RTSP out
- **Service‑per‑task architecture**
  - Every task runs as its **own container**, enabling independent scaling and failure isolation.
- **Standards‑based I/O**
  - **RTSP** for video in/out (MediaMTX), **MQTT** for structured results/events between stages.
- **GPU‑ready base images**
  - Curated **base images** for ONNX Runtime (GPU), PyTorch (GPU), and OpenCV (CPU).
- **Config via environment variables**
  - Uniform env keys across services (e.g., `IN_MQTT_URL_1`, `OUT_MQTT_URL`, `DEVICES`, `MODEL_INPUT_SIZE`).
- **Local dev without vendoring**
  - Keep your repo clean; for local runs, simply clone **`contanos-core`** and use its `contanos` folder.

## Installation and Setup

### Prerequisites

- **Docker** and **Docker Compose v2**
- For GPU workloads: **NVIDIA driver**, **CUDA‑capable GPU**, and **NVIDIA Container Toolkit** on the host
- (Optional, for local debugging) **Python 3.9+** and a Conda/venv with `requirements.txt`

### 1) Clone this repository

```bash
git clone <this-repo-url>
cd sparta-stride-main
```

### 2) Build the base images

The base images are tagged as `contanos:base-opencv-cpu`, `contanos:base-onnx-gpu`, and `contanos:base-pytorch-gpu`. They also **install `contanos-core`** inside the image.

```bash
# from the repository root
bash ./build-base-images.bash
```

> If you prefer manual builds:
>
> ```bash
> docker build stride/base-opencv-cpu   -t contanos:base-opencv-cpu
> docker build stride/base-onnx-gpu     -t contanos:base-onnx-gpu
> docker build stride/base-pytorch-gpu  -t contanos:base-pytorch-gpu
> ```

### 3) (Optional) Local debugging without Docker

If you want to run workers locally, clone Contanos once and make the `contanos` package importable:

```bash
git clone https://github.com/yyhtbs-yye/contanos-core
export PYTHONPATH=/absolute/path/to/contanos-core:$PYTHONPATH
# or place/symlink just the `contanos/` folder next to your scripts
```

## Usage

The ready‑to‑run **Docker Compose** file lives in `stride/docker-compose.yml`.

### Quick start (end‑to‑end demo)

```bash
cd stride
docker compose up --build -d
# or selectively:
# docker compose up --build -d mqtt-broker rtsp-server mp4-rtsp-source yolox-service bytetrack-service rtmpose-service annotator-service
```

What you get by default:

- **MQTT broker** at `localhost:1883`
- **RTSP server** (MediaMTX) at `rtsp://localhost:8554/`
- **Input stream** from the `mp4-rtsp-source` service
- **Annotated RTSP output** at **`rtsp://localhost:5108/annotated_stream`**
- **MQTT topics** produced/consumed by services:
  - `yolox`, `bytetrack`, `rtmpose`, `cmc`, `jerseyocr`
  - The annotator subscribes to these and publishes the RTSP overlay.

**Watching the output**: open the annotated stream in VLC/FFplay, e.g.

```bash
ffplay -rtsp_transport tcp rtsp://localhost:5108/annotated_stream
```

### Configuration via environment variables

The compose file wires services using a small set of shared env vars:

- `IN_RTSP_URL` – RTSP input (e.g., `rtsp://localhost:8554,topic=mystream`)

- `OUT_RTSP_URL` – RTSP output (host:port and topic)

- `IN_MQTT_URL_*` – One or more MQTT subscriptions, e.g.

  ```
  mqtt://localhost:1883,topic=yolox,client_id=yolox,qos=2,queue_max_len=100
  ```

- `OUT_MQTT_URL` – Where a service publishes results, same URI style as above

- `DEVICES` – Compute device(s), e.g. `cuda:0` or `cuda:0,cuda:1` (CPU services ignore this)

- `MODEL_INPUT_SIZE` – Optional model‑specific input resolution (e.g., `640,640`)

You can override these on `docker compose` command lines or by editing `stride/docker-compose.yml`.

### Logs & troubleshooting

```bash
docker compose logs -f yolox-service
docker compose ps
docker stats
```

If the annotated stream is empty, ensure the producer topics (`yolox`, `bytetrack`, `rtmpose`) are flowing and the annotator is subscribed.

## Directory Structure

```
.
├── build-base-images.bash         # helper script to (re)build base images
├── pyproject.toml / requirements.txt
├── stride/
│   ├── docker-compose.yml         # end-to-end pipeline orchestration
│   ├── README_DOCKER_SERVICES.md  # deeper dive into the services
│   ├── mqtt-broker/               # Eclipse Mosquitto broker
│   ├── mediamtx/                  # RTSP server support
│   ├── mp4-rtsp-source-raw/       # simple MP4 → RTSP source
│   ├── mp4-transcoder-sei/        # (optional) RTSP/mp4 transcoder with SEI
│   ├── base-opencv-cpu/           # CPU base (OpenCV + utilities)
│   ├── base-onnx-gpu/             # GPU base (Conda + ONNX Runtime)
│   ├── base-pytorch-gpu/          # GPU base (Conda + PyTorch)
│   ├── prj-yolox-onnx/            # detection service (YOLOX, ONNX)
│   ├── prj-bytetrack-cpu/         # multi-object tracker (ByteTrack)
│   ├── prj-rtmpose-onnx/          # multi-person pose (RTMPose, ONNX)
│   ├── prj-annotation-cpu/        # management of annotations
│   ├── prj-annotator/             # video overlay + RTSP out
│   ├── prj-jerseyocr-gpu/         # jersey OCR (optional)
│   └── prj-cmc-cpu/               # CMC module (optional)
```

> Note again: the **Contanos** sources are **not** present in this repository. Containers install `contanos-core` during image build; for local runs, clone `contanos-core` and use only the `contanos` folder.

## License

This project is released under the **MIT License**.

Copyright (c) 2025.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

> The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

## Contributing and Contact Info

Contributions are welcome! Bug reports, feature requests, and pull requests help improve STRIDE.

- **Issues**: open an issue with details and repro steps.
- **Pull requests**: keep changes focused; include concise docs and tests where appropriate.
- **Style**: follow the existing structure and env‑var conventions (`IN_*`, `OUT_*`, `DEVICES`, etc.).

**Contact**: open an issue, or reach out to the maintainer on GitHub (`yyhtbs-yye`).

## Acknowledgement

This work has received funding as an Open Call project under the SPARTA project from the European Union's Horizon Europe programme (grant agreement No. 101069732).
