FROM nvidia/cuda:12.6.2-base-ubuntu22.04
ARG pip_source=https://pypi.org/simple
LABEL authors="wangwm"
#ENV HOME=/workspace
WORKDIR /workspace
COPY . /workspace/heliumos-bixi-embeddings
#RUN cp /root/.bashrc /workspace/
RUN rm /bin/sh && ln -s /bin/bash /bin/sh
RUN apt-get update -y && apt-get install -y python3.11 curl net-tools openssh-server vim python3-pip python3.11-venv
RUN mkdir -p ~/.ssh && chmod 700 /root/.ssh && mkdir -p /run/sshd && touch /root/.ssh/authorized_keys \
    && chmod 600 /root/.ssh/authorized_keys
RUN ln -snf /usr/bin/python3.11 /usr/bin/python3
RUN python3 -m venv heliumos-env
RUN --mount=type=cache,target=/root/.cache/pip \
    source /workspace/heliumos-env/bin/activate \
    && python3 -m pip install --upgrade pip -i ${pip_source}
RUN --mount=type=cache,target=/root/.cache/pip \
    source /workspace/heliumos-env/bin/activate \
    && cd /workspace/heliumos-bixi-embeddings \
    && wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.5.6/flash_attn-2.5.6+cu122torch2.3cxx11abiFALSE-cp311-cp311-linux_x86_64.whl \
    && pip3 install flash_attn-2.5.6+cu122torch2.3cxx11abiFALSE-cp311-cp311-linux_x86_64.whl torch==2.4.1 FlagEmbedding==1.3.0 -i ${pip_source} \
    && pip3 install . -i ${pip_source}
RUN pip3 cache purge && cd /workspace && rm -rf /workspace/heliumos-bixi-embeddings
