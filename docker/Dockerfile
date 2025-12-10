# Stage 1: Build ROS workspace
ARG CUDA_VERSION=12.4

FROM ros:humble-ros-core AS ros-builder

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-colcon-common-extensions \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /ros2_ws/src
WORKDIR /ros2_ws

COPY ros2_ws/src /ros2_ws/src

RUN bash -c 'source /opt/ros/humble/setup.bash && \
    cd /ros2_ws && \
    colcon build --symlink-install'

# Stage 2: Final image with PyTorch
# ensure that your cuda version matches on the system
# the torch version may be deprecated if you use this a few years down the line and require refactoring
FROM pytorch/pytorch:2.4.1-cuda${CUDA_VERSION}-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive

# Install ROS 2
RUN apt-get update && apt-get install -y \
    curl \
    gnupg2 \
    lsb-release \
    && export ROS_APT_SOURCE_VERSION=$(curl -s https://api.github.com/repos/ros-infrastructure/ros-apt-source/releases/latest | grep -F "tag_name" | awk -F\" '{print $4}') \
    && curl -L -o /tmp/ros2-apt-source.deb "https://github.com/ros-infrastructure/ros-apt-source/releases/download/${ROS_APT_SOURCE_VERSION}/ros2-apt-source_${ROS_APT_SOURCE_VERSION}.$(. /etc/os-release && echo ${UBUNTU_CODENAME:-${VERSION_CODENAME}})_all.deb" \
    && dpkg -i /tmp/ros2-apt-source.deb \
    && apt-get update && apt-get install -y ros-humble-ros-base \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip3 install --no-cache-dir \
    numpy \
    scipy \
    scikit-learn \
    pandas

# Copy built workspace from stage 1
COPY --from=ros-builder /ros2_ws /ros2_ws

# Copy entrypoints
COPY docker/entrypoints /entrypoints
RUN chmod +x /entrypoints/*.sh

# Setup environment
RUN mkdir -p /results

WORKDIR /ros2_ws

CMD ["/bin/bash"]