# Runtime image with the headless ALIEN console build (cli), meant for rented GPU
# instances such as vast.ai.
#
# The binary is not compiled here. .github/workflows/docker-nightly-deploy.yml
# builds it natively on the CI runner and stages it next to resources/ in
# docker/payload, which is also the build context of this file:
#
#     docker build -f Dockerfile -t <user>/alien:nightly docker/payload
#
# The context therefore holds nothing but the payload, so neither the CUDA
# toolkit nor the vcpkg build tree can end up in the published layers.

# The base image carries the SSH, Jupyter and instance portal setup that vast.ai
# instances are operated through. Its CUDA version has to match the toolkit the
# binary was built with: the CUDA runtime is linked dynamically, so a build made
# with CUDA 13.0 needs libcudart.so.13. CUDA 13.0 in turn requires an NVIDIA
# driver of 580 or newer, which is a filter criterion when renting an instance.
ARG BASE_IMAGE=vastai/base-image:cuda-13.0.3-auto
FROM ${BASE_IMAGE}

# The engine links OpenGL and X11 for the CUDA/OpenGL interop of the geometry
# buffers (source/EngineInterface/GeometryBuffers.cpp), and Base links ImGui for
# settings serialization. cli never opens a window and calls none of it, but the
# shared libraries still have to resolve when the executable is loaded.
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        libgl1 \
        libglu1-mesa \
        libx11-6 \
        libxext6 \
        libxrandr2 \
        libxinerama1 \
        libxcursor1 \
        libxi6 \
 && rm -rf /var/lib/apt/lists/*

# vast.ai starts Jupyter with --preferred-dir /workspace, but the base image does
# not carry that directory, which makes the server abort during initialization.
# It is unrelated to ALIEN and only affects the Jupyter launch mode.
RUN mkdir -p /workspace

COPY cli /opt/alien/cli
COPY resources /opt/alien/resources

# NetworkService reads ./resources/ca-bundle.crt through a relative path, so the
# working directory has to be the one that holds resources/.
WORKDIR /opt/alien
ENV PATH="/opt/alien:${PATH}"

# Deliberately no ENTRYPOINT and no CMD: the base image starts SSH, Jupyter and
# the instance portal from its own entrypoint, and overriding it would take those
# down. The simulation is started by hand on the instance, inside tmux.
