# Use your base
FROM ros:humble-ros-base-jammy

SHELL ["/bin/bash", "-lc"]
ENV DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC

# ---- Tools you likely need (adjust as you like) ----
RUN apt-get update && apt-get install --no-install-recommends -y \
    build-essential git sudo \
    python3-pip python3-setuptools python3-wheel \
    python3-colcon-common-extensions python3-rosdep python3-vcstool \
    && rm -rf /var/lib/apt/lists/*

# ---- Non-root dev user ----
ARG USERNAME=dev
ARG UID=1000
ARG GID=1000
RUN groupadd -g "${GID}" "${USERNAME}" \
 && useradd -m -u "${UID}" -g "${GID}" -s /bin/bash "${USERNAME}" \
 && usermod -aG sudo "${USERNAME}" \
 && echo "${USERNAME} ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/${USERNAME}

# ---- rosdep init/update ----
RUN rosdep init || true
USER ${USERNAME}
RUN rosdep update || true

# ---- Workspace ----
ENV WS_DIR=/home/${USERNAME}/ws
WORKDIR ${WS_DIR}
RUN mkdir -p ${WS_DIR}/src

ENV ROS_DISTRO=humble

COPY . /home/dev/ws/src/dynamic-box

USER root

RUN echo "source /opt/ros/$ROS_DISTRO/setup.bash" >> /etc/bash.bashrc

RUN git clone https://github.com/RoboSense-LiDAR/rslidar_sdk.git /home/${USERNAME}/ws/src/rslidar_sdk \
    && cd /home/dev/ws/src/rslidar_sdk \
    && git submodule init \
    && git submodule update \
    && sudo apt-get update \
    && sudo apt-get install -y libyaml-cpp-dev \
    && sudo apt-get install -y libpcap-dev
    
RUN git clone https://github.com/RoboSense-LiDAR/rslidar_msg.git /home/${USERNAME}/ws/src/rslidar_msg

RUN source /opt/ros/$ROS_DISTRO/setup.bash \
    && cd /home/dev/ws/src \
    && colcon build --packages-select rslidar_msg rslidar_sdk \
    && cd /home/dev/ws/src/dynamic-box/my_rosbag_reader \
    && colcon build --packages-select my_rosbag_reader 

CMD ["bash"]
