pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 \
  --index-url https://mirrors.aliyun.com/pytorch-wheels/cu128 \
  --extra-index-url https://mirrors.aliyun.com/pypi/simple

pip install vllm==0.19.1 \
  -i https://mirrors.aliyun.com/pypi/simple \
  --extra-index-url https://mirrors.aliyun.com/pytorch-wheels/cu128

pip install ms-swift==4.4.2

conda install -y nvidia/label/cuda-12.8.0::cuda-toolkit

export CUDA_HOME="$CONDA_PREFIX"
export CUDA_PATH="$CONDA_PREFIX"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

pip install deepspeed==0.19.3

pip install "qwen-vl-utils>=0.0.14" "decord"

conda install -c conda-forge gcc_linux-64 gxx_linux-64 -y
