# Use your base
FROM nvcr.io/nvidia/isaac/ros:aarch64-ros2_humble_adc428c7077de4984a00b63c55903b0a

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

RUN git clone https://github.com/RoboSense-LiDAR/rslidar_msg.git /home/${USERNAME}/ws/src/rslidar_msg && git clone https://github.com/cornellev/cev_msgs.git src/cev_msgs && source /opt/ros/$ROS_DISTRO/setup.bash

COPY /my_rosbag_reader/my_rosbag_reader/requirements.txt .

RUN pip install --no-cache-dir --ignore-installed -r requirements.txt

RUN source /opt/ros/$ROS_DISTRO/setup.bash \
    && cd /home/dev/ws/src \
    && colcon build --packages-select rslidar_msg rslidar_sdk cev_msgs \
    && source install/setup.bash \
    && cd /home/dev/ws/src/dynamic-box/my_rosbag_reader \
    && colcon build --packages-select my_rosbag_reader \
    && source install/setup.bash \
    && cd /home/dev/ws/src/dynamic-box/my_rosbag_reader/my_rosbag_reader \
    && python3 setup.py build_ext --inplace

CMD ["bash"]