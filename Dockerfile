# Container image for the interview server.
#
# WHY A CONTAINER AND NOT A SERVERLESS BUNDLE
# -------------------------------------------
# This app holds an interview open. Five WebSockets carry the candidate's
# frames, both microphones, the interviewer's camera and the panel's live
# read; the session itself lives in a process-local dict (`web.server
# .SESSIONS`) alongside a MediaPipe graph and a Whisper model that are
# expensive to build and are reused across every frame of a call. None of
# that survives being cut into stateless invocations, which is what rules out
# the serverless platforms rather than any packaging detail.
#
# ONE INSTANCE. NOT NEGOTIABLE UNTIL WP6.
# ---------------------------------------
# `SESSIONS` and `web.live.HUB` are in-process memory. Two replicas behind a
# load balancer means the candidate's frames land on one and the panel's
# socket on the other, and neither can see the interview -- it does not
# degrade, it simply does not work. Scale by making the box bigger, and set
# the replica count to 1 wherever this is deployed. WP6 replaces the store,
# and that is the change that makes horizontal scale possible.
# --platform is deliberate. mediapipe publishes manylinux wheels for x86_64
# only: on linux/arm64 the newest available is 0.10.18, so the
# `mediapipe>=0.10.30,<1.0` pin in requirements.txt cannot be satisfied and
# the build dies at pip install. Every mainstream host (Render, Railway,
# Fly's shared-cpu-x86, EC2) runs amd64, so this pins the build to the
# platform that is actually deployed to -- and, as a side effect, makes a
# build on an Apple Silicon laptop produce the same image as the one in
# production rather than a different one that happens to work.
#
# The cost is that a local build on Apple Silicon runs under emulation and is
# slow. A build on the host is native and is not.
FROM --platform=linux/amd64 python:3.11-slim

# System libraries the wheels link against but do not vendor.
#
#   libgl1, libglib2.0-0  OpenCV links them even for decode-only work.
#   libgles2, libegl1     MediaPipe's C bindings dlopen libGLESv2.so.2 at
#                         FaceLandmarker.create_from_options -- long after
#                         `import mediapipe` has succeeded. Without them the
#                         container starts, serves every page, accepts the
#                         candidate's WebSocket, and only then dies with
#                         "libGLESv2.so.2: cannot open shared object file"
#                         when the first frame arrives. A missing package
#                         here does not look like a missing package; it looks
#                         like the interview being broken.
#   ffmpeg                faster-whisper decodes the browser's webm/opus
#                         chunks through it.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libgles2 \
        libegl1 \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.cache/huggingface

# Dependencies first, so a code change does not reinstall 700 MB of wheels.
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Bake the downloaded models into the image.
#
# Both of these are fetched on FIRST USE otherwise -- Whisper when the first
# word is spoken, the embedder when the first answer is scored for relevance.
# That download would land in the middle of a live interview, on a container
# with no warm cache, and the first candidate of every deploy would pay for
# it. ~230 MB of image to avoid a stall nobody can explain to the person in
# the chair.
RUN python -c "from faster_whisper import WhisperModel; \
               WhisperModel('base.en', compute_type='int8')" && \
    python -c "from huggingface_hub import hf_hub_download; \
               hf_hub_download('sentence-transformers/all-MiniLM-L6-v2', \
                               'onnx/model.onnx'); \
               hf_hub_download('sentence-transformers/all-MiniLM-L6-v2', \
                               'tokenizer.json')"

COPY . .

# Fail the BUILD rather than the first interview if a weight is missing.
# fetch_models.py --verify is the same check CI runs; a container that starts
# and then cannot find a landmarker is a worse outcome than one that never
# starts.
RUN python fetch_models.py --verify

# Written at runtime: session records, interview records, consent, the audit
# log, uploaded CVs and recordings. Mount a persistent volume here or every
# deploy erases the consent records, which is the one directory that must
# outlive the container.
VOLUME ["/app/out"]

EXPOSE 8000

# One worker, for the reason in the header. $PORT is honoured because most
# hosts inject it; 8000 is the fallback for a plain `docker run`.
CMD ["sh", "-c", "uvicorn web.server:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]
