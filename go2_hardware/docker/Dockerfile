# For deployment (ARM64) - build separately
# FROM dustynv/ros:humble-desktop-pytorch-l4t-r35.4.1
FROM ros:humble-ros-base

# Use standard Ubuntu + ROS Humble base image for x86_64 (local development)
# FROM osrf/ros:humble-desktop-full
# To make sure installing tzdata will not prompt questions
ARG DEBIAN_FRONTEND=noninteractive

# ROS GPG KEY FIX
RUN sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg

RUN apt-get update && apt-get install -y --no-install-recommends \
    zsh neovim tmux git htop curl wget \
    net-tools iputils-ping\
    python3-pip \
    # init certificate for curl
    ca-certificates \
    # For x11 forwarding tests (xeyes, xclock etc.)
    x11-apps \
    # For pytorch
    libopenblas-dev \
    cmake sudo \
    software-properties-common

ARG USERNAME=real
# Defaults make plain `docker build` work too; devcontainer overrides these.
ARG USER_UID=1000
ARG USER_GID=1000


# Add the new user (userid should be consistent to the one in the host machine)
# Change the password to "real". You can use other password by changing the following line
RUN groupadd -g ${USER_GID} ${USERNAME} && \
    useradd ${USERNAME}  -u ${USER_UID} -g ${USER_GID} -m -p "$(openssl passwd -1 real)"
# Add sudo access (not recommended, but is hulepful for development)
RUN usermod -aG sudo ${USERNAME}
# Give ownership of the user's home directory to the new user
RUN chown -R ${USERNAME}:${USERNAME} /home/${USERNAME}
USER ${USERNAME}
# Set the user's home directory as the working directory
WORKDIR /home/${USERNAME}


############### Development Tools ###############

# install and setup zsh (with oh-my-zsh and plugins)
# Make oh-my-zsh install non-interactive for Docker builds
RUN sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)" 
ARG PLUGIN_DIR=/home/${USERNAME}/.oh-my-zsh/custom/plugins/
RUN git clone https://github.com/marlonrichert/zsh-autocomplete.git $PLUGIN_DIR/zsh-autocomplete && \
    git clone https://github.com/zsh-users/zsh-autosuggestions $PLUGIN_DIR/zsh-autosuggestions  && \
    git clone https://github.com/zsh-users/zsh-syntax-highlighting.git $PLUGIN_DIR/zsh-syntax-highlighting

# Set up zsh plugins and theme
# And some key bindings: ctrl-R to search in the history, ctlr-backspace to delete a word, etc.
RUN mv /home/${USERNAME}/.zshrc /home/${USERNAME}/.zshrc.bak && \
    echo 'export ZSH="$HOME/.oh-my-zsh"\nplugins=(\n  git\n  zsh-autosuggestions\n  zsh-autocomplete\n  zsh-syntax-highlighting\n  themes\n)\n' >> /home/${USERNAME}/.zshrc && \
    echo 'ZSH_THEME="eastwood"\nsource $ZSH/oh-my-zsh.sh' >> /home/${USERNAME}/.zshrc && \
    echo 'bindkey -M menuselect "\\r" .accept-line' >> /home/${USERNAME}/.zshrc && \
    echo 'bindkey -M menuselect -s "^R" "^_^_^R" "^S" "^_^_^S"' >> /home/${USERNAME}/.zshrc && \
    echo 'bindkey -M menuselect "\\e[D" .backward-char "\\eOD" .backward-char "\\e[C" .forward-char "\\eOC" .forward-char' >> /home/${USERNAME}/.zshrc && \
    echo "bindkey '^H' backward-kill-word" >> /home/${USERNAME}/.zshrc

# Setup python env (we prefer mamba to conda for faster installation)
# RUN ARCH=$(uname -m) && wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-${ARCH}.sh && \
#     bash Miniforge3-Linux-${ARCH}.sh -b -p /home/${USERNAME}/miniforge3 && \
#     rm Miniforge3-Linux-${ARCH}.sh 
# RUN /home/${USERNAME}/miniforge3/bin/conda init zsh

# Install pytorch. 
# It does not have access to cuda kernels as the Jetpack version on Unitree jetson is usually lower than 6.0.
# In our project, the whole-body controller only uses CPU and diffusion policy is run on a separate PC
#   thus cuda compute is not required in our project
# RUN wget https://developer.download.nvidia.cn/compute/redist/jp/v511/pytorch/torch-2.0.0+nv23.05-cp38-cp38-linux_aarch64.whl -O torch-2.1.0-cp310-cp310-linux_aarch64.whl && \
#     /home/${USERNAME}/miniforge3/bin/pip install --no-cache torch-2.1.0-cp310-cp310-linux_aarch64.whl && \
#     rm torch-2.1.0-cp310-cp310-linux_aarch64.whl


# RUN /home/${USERNAME}/miniforge3/bin/pip install numpy

USER root
# # Add ROS 2 repository
# RUN echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | tee /etc/apt/sources.list.d/ros2.list > /dev/null

RUN apt-get update && \
    apt-get install -y \
    ros-humble-rmw-cyclonedds-cpp ros-humble-rosidl-generator-dds-idl \
    ros-humble-geometry-msgs ros-humble-ament-lint-auto ros-humble-rosidl-default-generators \
    ros-humble-rviz2

USER ${USERNAME}
# Install unitree ros
RUN git clone https://github.com/unitreerobotics/unitree_ros2.git && \
    cd unitree_ros2/cyclonedds_ws/src && \
    git clone https://github.com/ros2/rmw_cyclonedds -b humble && \
    git clone https://github.com/eclipse-cyclonedds/cyclonedds -b releases/0.10.x && \
    cd .. && \
    colcon build --packages-select cyclonedds

RUN cd unitree_ros2/cyclonedds_ws && \
    pip install empy==3.3.2 catkin_pkg pyparsing lark && \
    /bin/zsh -c "source /opt/ros/humble/setup.zsh && colcon build"


# RUN cd unitree_ros2/cyclonedds_ws && \
#     /home/${USERNAME}/miniforge3/bin/pip install empy==3.3.2 catkin_pkg pyparsing lark && \
#     /bin/zsh -c "cd ~/unitree_ros2/cyclonedds_ws && source /opt/ros/humble/install/setup.zsh && colcon build"

RUN sed -i "s/\\\\\\\\/\\\\/g" /home/${USERNAME}/.zshrc && \
    sed -i "s/\\\\n/\\n/g" /home/${USERNAME}/.zshrc


# Install the customized unitree sdk for 
# 1) a python module to calculate CRC code for ros2 packages
# 2) a script to disable sports mode of Go2
USER ${USERNAME}
RUN git clone https://github.com/yihuai-gao/unitree_sdk2.git && \
    git clone https://github.com/unitreerobotics/unitree_sdk2_python.git && \
    cd unitree_sdk2 && \
    git checkout 92506a409843c78be358d761d19aa254053a045f && \
    mkdir build && cd build && \
    cmake .. && make -j$(($(nproc)-1)) && \
    pip install pybind11 && \
    cd ../python && mkdir build && cd build &&  \
    cmake .. -DPYBIND_PATH=$(pip show pybind11 | grep Location | cut -d' ' -f2) && \
    make -j$(($(nproc)-1)) && \
    echo "export PYTHONPATH=\$PYTHONPATH:/home/${USERNAME}/unitree_sdk2/python" >> /home/${USERNAME}/.zshrc

USER root
RUN cd /home/${USERNAME}/unitree_sdk2 && \
    ./install.sh

# Manually build and install CycloneDDS C library
WORKDIR /opt
RUN git clone --branch 0.10.2 https://github.com/eclipse-cyclonedds/cyclonedds.git \
	&& cd cyclonedds \
	&& mkdir build && cd build \
	&& cmake .. -DCMAKE_INSTALL_PREFIX=/usr/local \
	&& make -j$(nproc) \
	&& make install

ENV CMAKE_PREFIX_PATH=/usr/local:$CMAKE_PREFIX_PATH
# Set environment variable so CMake can locate the installed CycloneDDS
RUN echo "export CMAKE_PREFIX_PATH=/usr/local:$CMAKE_PREFIX_PATH" >> /home/${USERNAME}/.zshrc && \
    cd /home/${USERNAME}/unitree_sdk2_python && \
    pip install -e .

USER root

RUN echo "source /opt/ros/humble/setup.zsh" >> /home/${USERNAME}/.zshrc && \
    echo "source /home/\${USERNAME}/unitree_ros2/cyclonedds_ws/install/setup.zsh" >> /home/${USERNAME}/.zshrc && \
    echo "export CYCLONEDDS_URI=/home/\${USERNAME}/unitree_ros2/cyclonedds_ws/src/cyclonedds.xml" >> /home/${USERNAME}/.zshrc && \
    sed -i "s/enp2s0/eth0/g" /home/${USERNAME}/unitree_ros2/cyclonedds_ws/src/cyclonedds.xml

RUN echo "source /opt/ros/humble/setup.zsh" >> /home/${USERNAME}/.zshrc
# Install ApproximateFoRL dependencies
RUN pip install \
    hydra-core \
    hydra-colorlog \
    pyglet \
    # gymnasium \
    wandb \
    # transforms3d \
    filterpy \
    numba \
    omegaconf \
    gitpython \
    zarr \
    rich \
    python-socketio \
    eventlet \
    pynput \
    matplotlib \ 
    seaborn

USER root

RUN apt-get update && \
    apt-get install -y git zsh curl git vim net-tools wget htop \
    x11-apps neofetch build-essential libboost-all-dev libeigen3-dev && \
    apt-get install libtinyxml2-dev libcppunit-dev liborocos-kdl-dev liburdfdom-dev iproute2 -y
RUN ln -s /usr/include/eigen3/Eigen /usr/local/include/Eigen

RUN git clone https://github.com/ros/kdl_parser.git && \
    cd kdl_parser && \
    git checkout 74d4ee3bc6938de8ae40a700997baef06114ea1b && \
    cd kdl_parser && \
    sed -i 's/kdl_parser\/visibility_control.hpp/visibility_control.hpp/g' include/kdl_parser/kdl_parser.hpp && \
    mkdir build && cd build && \
    cmake .. && make -j$(($(nproc)-1)) && make install


USER root

# Setup spacenav and rosbag2 storage
RUN apt-get update && \
    # apt-get install -y ros-humble-rosbag2-storage-mcap && \
    apt-get install -y libspnav-dev spacenavd

USER ${USERNAME}

# Note: ApproximateFoRL and Genesis will be installed from the mounted workspace
# The workspace is mounted via devcontainer.json, so dependencies will be installed
# when you run `pip install .` inside the container
