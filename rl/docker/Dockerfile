# Fortress bots: dedicated server with the RL bridge plus the Python trainer.
#
#   docker build -f rl/docker/Dockerfile -t fortbots .           (from the repository root)
#   docker run --rm -p 4534:4534/udp fortbots                    (public 6v6 bot server)
#   docker run --rm -v $PWD/runs:/opt/fortrl/rl/runs fortbots \
#       fortrl.train configs/train_6v6.toml --run-dir runs/my6v6 --init models/fort_1v1.pt

# Docker Hub rate limits? --build-arg BASE=public.ecr.aws/ubuntu/ubuntu:24.04
ARG BASE=ubuntu:24.04

########################################
FROM ${BASE} AS build

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        autoconf automake bison ca-certificates g++ libboost-dev libboost-system-dev \
        libboost-thread-dev libcurl4-openssl-dev libprotobuf-dev libtool libxml2-dev make \
        patch pkg-config protobuf-compiler python3 \
    && rm -rf /var/lib/apt/lists/*

COPY . /src
WORKDIR /src
RUN ./bootstrap.sh && mkdir -p /build && cd /build \
    && /src/configure --enable-dedicated --disable-glout --disable-master --disable-sysinstall \
        --disable-desktop --disable-etc --disable-useradd --prefix=/opt/armagetronad CXXFLAGS="-O2" \
    && make -j"$(nproc)" && make install

########################################
FROM ${BASE}

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        ca-certificates libboost-thread1.83.0 libcurl4t64 libprotobuf32t64 libxml2 \
        python3 python3-venv \
    && rm -rf /var/lib/apt/lists/*

# CPU build of PyTorch by default; pass --build-arg TORCH_INDEX=https://pypi.org/simple for the CUDA build
ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu
RUN python3 -m venv /opt/venv \
    && /opt/venv/bin/pip install --no-cache-dir --index-url "$TORCH_INDEX" torch \
    && /opt/venv/bin/pip install --no-cache-dir numpy matplotlib

COPY --from=build /opt/armagetronad /opt/armagetronad
COPY rl /opt/fortrl/rl

ENV PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    FORTRL_SERVER=/opt/armagetronad/bin/armagetronad-dedicated
WORKDIR /opt/fortrl/rl

EXPOSE 4534/udp
ENTRYPOINT ["python", "-m"]
CMD ["fortrl.serve", "--policy", "models/fort_6v6.pt", "--game", "configs/fort_6v6.cfg", \
     "--name", "Fort Bots 6v6", "--port", "4534"]
