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
  && apt-get update \
  && apt-get install -y libyaml-cpp-dev libpcap-dev libgl1-mesa-glx libgl1-mesa-dev

RUN apt-get install ros-$ROS_DISTRO-rviz2 -y

RUN git clone https://github.com/RoboSense-LiDAR/rslidar_msg.git /home/${USERNAME}/ws/src/rslidar_msg && git clone https://github.com/cornellev/cev_msgs.git src/cev_msgs && source /opt/ros/$ROS_DISTRO/setup.bash

RUN pip install --no-cache-dir \
  flask_cors \
  google-cloud \
  google-auth \
  google-cloud-storage \
  Flask==3.0.0 \
  gunicorn \
  flask_socketio \
  websocket-client \
  pybind11 \
  matplotlib \
  open3d[full] --no-deps --trusted-host pypi.org --trusted-host files.pythonhosted.org \
  dash \
  plotly
  
RUN source /opt/ros/$ROS_DISTRO/setup.bash \
    && cd /home/dev/ws/src \
    && colcon build --packages-select rslidar_msg rslidar_sdk \
    && cd /home/dev/ws/src/dynamic-box/my_rosbag_reader \
    && colcon build --packages-select my_rosbag_reader 

CMD ["bash"]
