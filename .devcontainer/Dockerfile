FROM nvidia/cuda:11.1.1-cudnn8-devel


SHELL ["/bin/bash", "-c"]
WORKDIR /root/

RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys \
        A4B469963BF863CC && \
    apt-get update && apt-get install -y \
    build-essential wget

RUN mkdir -p ~/miniconda3 && \
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh && \
    bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3 && \
    rm ~/miniconda3/miniconda.sh

ENV PATH=$PATH:~/miniconda3/bin/

RUN conda create -y -n flattening python=3.9 && \
    conda init --all && \
    source ~/miniconda3/bin/activate && \
    conda activate flattening && \
    pip install torch==1.10.1+cu111 torchvision==0.11.2+cu111 -f \
        https://download.pytorch.org/whl/cu111/torch_stable.html && \
    pip install numpy==1.22.4
