FROM ubuntu:22.04

WORKDIR /content

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=True
ENV PATH="/home/camenduru/.local/bin:/usr/local/cuda/bin:${PATH}"

RUN apt update -y && apt install -y software-properties-common build-essential \
    libgl1 libglib2.0-0 zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev libssl-dev libreadline-dev libffi-dev && \
    add-apt-repository -y ppa:git-core/ppa && apt update -y && \
    apt install -y python-is-python3 python3-pip sudo nano aria2 curl wget git git-lfs unzip unrar ffmpeg && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://developer.download.nvidia.com/compute/cuda/12.6.2/local_installers/cuda_12.6.2_560.35.03_linux.run -d /content -o cuda_12.6.2_560.35.03_linux.run && sh cuda_12.6.2_560.35.03_linux.run --silent --toolkit && \
    echo "/usr/local/cuda/lib64" >> /etc/ld.so.conf && ldconfig && \
    git clone https://github.com/aristocratos/btop /content/btop && cd /content/btop && make && make install && \
    adduser --disabled-password --gecos '' camenduru && \
    adduser camenduru sudo && \
    echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers && \
    chown -R camenduru:camenduru /content && \
    chmod -R 777 /content && \
    chown -R camenduru:camenduru /home && \
    chmod -R 777 /home
    
USER camenduru

RUN pip install torch==2.5.1+cu124 torchvision==0.20.1+cu124 torchaudio==2.5.1+cu124 torchtext==0.18.0 torchdata==0.8.0 --extra-index-url https://download.pytorch.org/whl/cu124 && \
    pip install xformers==0.0.28.post3 && \
    pip install opencv-contrib-python av runpod && \
    pip install accelerate==1.1.1 albumentations==1.4.21 audio-separator==0.24.1 black==23.12.1 diffusers==0.31.0 einops==0.8.0 ffmpeg-python==0.2.0 funasr==1.0.27 huggingface-hub==0.26.2 imageio==2.36.0 && \
    pip install imageio-ffmpeg==0.5.1 insightface==0.7.3 hydra-core==1.3.2 jax==0.4.35 mediapipe==0.10.18 modelscope==1.20.1 moviepy==1.0.3 numpy==1.26.4 omegaconf==2.3.0 onnxruntime-gpu==1.20.1 && \
    pip install opencv-python-headless==4.10.0.84 pillow==10.4.0 scikit-learn==1.5.2 scipy==1.14.1 transformers==4.46.3 tqdm==4.67.1 matplotlib matplotlib-inline && \
    git clone -b dev https://github.com/camenduru/memo /content/memo && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/memoavatar/memo/raw/main/audio_proj/config.json -d /content/memo/checkpoints/audio_proj -o config.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/memoavatar/memo/resolve/main/audio_proj/diffusion_pytorch_model.safetensors -d /content/memo/checkpoints/audio_proj -o diffusion_pytorch_model.safetensors && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/memoavatar/memo/raw/main/diffusion_net/config.json -d /content/memo/checkpoints/diffusion_net -o config.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/memoavatar/memo/resolve/main/diffusion_net/diffusion_pytorch_model.safetensors -d /content/memo/checkpoints/diffusion_net -o diffusion_pytorch_model.safetensors && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/memoavatar/memo/raw/main/image_proj/config.json -d /content/memo/checkpoints/image_proj -o config.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/memoavatar/memo/resolve/main/image_proj/diffusion_pytorch_model.safetensors -d /content/memo/checkpoints/image_proj -o diffusion_pytorch_model.safetensors && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/memoavatar/memo/raw/main/misc/audio_emotion_classifier/config.json -d /content/memo/checkpoints/misc/audio_emotion_classifier -o config.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/memoavatar/memo/resolve/main/misc/audio_emotion_classifier/diffusion_pytorch_model.safetensors -d /content/memo/checkpoints/misc/audio_emotion_classifier -o diffusion_pytorch_model.safetensors && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/memoavatar/memo/resolve/main/misc/face_analysis/models/1k3d68.onnx -d /content/memo/checkpoints/misc/face_analysis/models -o 1k3d68.onnx && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/memoavatar/memo/resolve/main/misc/face_analysis/models/2d106det.onnx -d /content/memo/checkpoints/misc/face_analysis/models -o 2d106det.onnx && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/memoavatar/memo/resolve/main/misc/face_analysis/models/face_landmarker_v2_with_blendshapes.task -d /content/memo/checkpoints/misc/face_analysis/models -o face_landmarker_v2_with_blendshapes.task && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/memoavatar/memo/resolve/main/misc/face_analysis/models/genderage.onnx -d /content/memo/checkpoints/misc/face_analysis/models -o genderage.onnx && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/memoavatar/memo/resolve/main/misc/face_analysis/models/glintr100.onnx -d /content/memo/checkpoints/misc/face_analysis/models -o glintr100.onnx && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/memoavatar/memo/resolve/main/misc/face_analysis/models/scrfd_10g_bnkps.onnx -d /content/memo/checkpoints/misc/face_analysis/models -o scrfd_10g_bnkps.onnx && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/memoavatar/memo/resolve/main/misc/vocal_separator/Kim_Vocal_2.onnx -d /content/memo/checkpoints/misc/vocal_separator -o Kim_Vocal_2.onnx && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/memoavatar/memo/raw/main/misc/vocal_separator/download_checks.json -d /content/memo/checkpoints/misc/vocal_separator -o download_checks.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/memoavatar/memo/raw/main/misc/vocal_separator/mdx_model_data.json -d /content/memo/checkpoints/misc/vocal_separator -o mdx_model_data.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/memoavatar/memo/raw/main/misc/vocal_separator/vr_model_data.json -d /content/memo/checkpoints/misc/vocal_separator -o vr_model_data.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/memoavatar/memo/raw/main/reference_net/config.json -d /content/memo/checkpoints/reference_net -o config.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/memoavatar/memo/resolve/main/reference_net/diffusion_pytorch_model.safetensors -d /content/memo/checkpoints/reference_net -o diffusion_pytorch_model.safetensors && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/stabilityai/sd-vae-ft-mse/resolve/main/diffusion_pytorch_model.safetensors -d /content/memo/checkpoints/vae -o diffusion_pytorch_model.safetensors && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/stabilityai/sd-vae-ft-mse/raw/main/config.json -d /content/memo/checkpoints/vae -o config.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/facebook/wav2vec2-base-960h/resolve/main/model.safetensors -d /content/memo/checkpoints/wav2vec2 -o model.safetensors && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/facebook/wav2vec2-base-960h/raw/main/config.json -d /content/memo/checkpoints/wav2vec2 -o config.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/facebook/wav2vec2-base-960h/raw/main/preprocessor_config.json -d /content/memo/checkpoints/wav2vec2 -o preprocessor_config.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/emotion2vec/emotion2vec_plus_large/resolve/main/model.pt -d /content/memo/checkpoints/emotion2vec_plus_large -o model.pt && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/emotion2vec/emotion2vec_plus_large/raw/main/config.yaml -d /content/memo/checkpoints/emotion2vec_plus_large -o config.yaml && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/emotion2vec/emotion2vec_plus_large/raw/main/configuration.json -d /content/memo/checkpoints/emotion2vec_plus_large -o configuration.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/emotion2vec/emotion2vec_plus_large/raw/main/tokens.txt -d /content/memo/checkpoints/emotion2vec_plus_large -o tokens.txt
    
COPY ./worker_runpod.py /content/memo/worker_runpod.py
WORKDIR /content/memo
CMD python worker_runpod.py