FROM nvidia/cuda:12.6.2-runtime-ubuntu22.04
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
RUN source /workspace/heliumos-env/bin/activate \
 && python3 -m pip install --no-cache-dir --upgrade pip -i ${pip_source}
RUN source /workspace/heliumos-env/bin/activate \
    && cd /workspace/heliumos-bixi-embeddings \
    && pip3 install --no-cache-dir . -i ${pip_source}
RUN pip3 cache purge && cd /workspace && rm -rf /workspace/heliumos-bixi-embeddings
