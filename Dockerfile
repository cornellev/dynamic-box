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

RUN apt-get update \
  && apt-get install -y \
  git openssh-client \
  libopencv-dev \
  libopen3d-dev \
  ros-humble-pcl-ros
  
# ---- Non-root dev user ----
ARG USERNAME=dev
ARG UID=1000
ARG GID=1000
RUN groupadd -g "${GID}" "${USERNAME}" 2>/dev/null || true \
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
  && apt-get update \
  && apt-get install -y libyaml-cpp-dev libpcap-dev libgl1-mesa-glx libgl1-mesa-dev

RUN cp /home/dev/ws/src/dynamic-box/config.yaml /home/dev/ws/src/rslidar_sdk/config/config.yaml

RUN apt-get install ros-$ROS_DISTRO-rviz2 -y

RUN apt-get update && \
    apt-get install -y \
        libsdl2-2.0-0 \
        libsdl2-dev \
        libsdl2-image-2.0-0 \
        libsdl2-image-dev

RUN git clone https://github.com/RoboSense-LiDAR/rslidar_msg.git /home/${USERNAME}/ws/src/rslidar_msg && git clone https://github.com/cornellev/cev_msgs.git src/cev_msgs && source /opt/ros/$ROS_DISTRO/setup.bash

COPY /my_rosbag_reader/my_rosbag_reader/requirements.txt .

RUN pip install --no-cache-dir --ignore-installed -r requirements.txt

RUN --mount=type=ssh \
  if git ls-remote https://github.com/cornellev/Obstacle_node.git &> /dev/null; then \
    echo "Access granted: cloning Obstacle_node"; \
    git clone --branch tmp_changes https://github.com/cornellev/Obstacle_node.git /home/${USERNAME}/ws/src/Obstacle_node; \
  else \
    echo "No access to Obstacle_node, skipping clone"; \
  fi

RUN git clone https://github.com/cornellev/icp.git /home/${USERNAME}/ws/src/Obstacle_node/lib/cev_icp

RUN cp /home/dev/ws/src/Obstacle_node/overlay/vanilla_3d.cpp /home/dev/ws/src/Obstacle_node/lib/cev_icp/lib/icp/impl/vanilla_3d.cpp
RUN cp /home/dev/ws/src/Obstacle_node/overlay/vanilla_3d.h /home/dev/ws/src/Obstacle_node/lib/cev_icp/include/icp/impl/vanilla_3d.h
RUN cp /home/dev/ws/src/Obstacle_node/overlay/icp.h /home/dev/ws/src/Obstacle_node/lib/cev_icp/include/icp/icp.h

RUN cd /home/dev/ws/src/Obstacle_node/lib/cev_icp \
    && sudo make install LIB_INSTALL=/usr/local/lib HEADER_INSTALL=/usr/local/include

RUN source /opt/ros/$ROS_DISTRO/setup.bash \
    && cd /home/dev/ws/src \
    && colcon build --packages-select rslidar_msg rslidar_sdk cev_msgs obstacle \
    && source install/setup.bash \
    && cd /home/dev/ws/src/dynamic-box/my_rosbag_reader \
    && colcon build --packages-select my_rosbag_reader \
    && source install/setup.bash \
    && cd /home/dev/ws/src/dynamic-box/my_rosbag_reader/my_rosbag_reader \
    && python3 setup.py build_ext --inplace
CMD ["bash"]