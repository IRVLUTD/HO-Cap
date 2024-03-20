# Use CUDA and cuDNN images from gitlab.com/nvidia/cuda as base image
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04

# Set labels for the docker image
LABEL maintainer="Jikai Wang <jikai.wang@utdallas.edu>"
LABEL version="0.1.0"
LABEL description="Dockerfile for building a docker image for HO101"

# Set NVIDIA GPU architecture for compiling CUDA code
ARG CUDA_ARCH="6.0 6.1 7.0 7.5 8.0 8.6 8.9"

###############################################################################
# Install developing tools & dependencies
###############################################################################

# Set working directory
WORKDIR /

# Change the default shell to Bash
SHELL ["/bin/bash", "--login", "-c"]

# Set noninteractive mode for apt-get
ENV DEBIAN_FRONTEND=noninteractive

# Install basic dependencies
RUN apt-get update --fix-missing && \
    apt-get install -qq -y --no-install-recommends build-essential gcc g++ cmake cmake-curses-gui make checkinstall binutils && \
    apt-get install -qq -y --no-install-recommends ca-certificates apt-transport-https software-properties-common && \
    apt-get install -qq -y --no-install-recommends libosmesa6-dev freeglut3 freeglut3-dev libglfw3 libglfw3-dev libglew-dev libfreeimage-dev libglu1-mesa libglu1-mesa-dev libx11-dev libxmu-dev libxi-dev libgl1-mesa-dev && \
    apt-get install -qq -y --no-install-recommends wget curl git vim tmux unzip zlib1g libzmq3-dev zsh sudo && \
    # Install dependencies for OpenCV
    apt-get install -qq -y --no-install-recommends pkg-config qt5-default libgtk2.0-dev libjpeg8-dev libtiff5-dev libpng-dev libavcodec-dev libavformat-dev libswscale-dev libdc1394-22-dev libxine2-dev libv4l-dev libxvidcore-dev libx264-dev libtbb-dev libatlas-base-dev libfaac-dev libmp3lame-dev libtheora-dev libvorbis-dev libblas-dev liblapack-dev liblapacke-dev gfortran yasm libopencore-amrnb-dev libopencore-amrwb-dev x264 v4l-utils libprotobuf-dev protobuf-compiler libgoogle-glog-dev libgflags-dev libgphoto2-dev libhdf5-dev doxygen && \
    # Install dependencies for PCL
    apt-get install -qq -y --no-install-recommends libboost-all-dev libflann-dev libqhull-dev libgtest-dev libpcap-dev proj-data libproj-dev libxmu-dev libxi-dev libopenni-dev libopenni2-dev libopenni-sensor-primesense-dev && \
    # Install dependencies for librealsense
    apt-get install -qq -y --no-install-recommends libssl-dev libusb-1.0-0-dev libudev-dev pkg-config libgtk-3-dev cmake libglfw3-dev libgl1-mesa-dev libglu1-mesa-dev at && \
    # Install dependencies for Azure Kinect
    apt-get install -qq -y --no-install-recommends libsoundio1

# Install ROS Noetic
RUN echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list && curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | sudo apt-key add - && \
    apt-get update && apt-get install -qq -y --no-install-recommends ros-noetic-desktop-full && \
    source /opt/ros/noetic/setup.bash && \
    apt-get update && apt-get install --quiet --no-install-recommends -y python3-rosdep python3-rosinstall python3-rosinstall-generator python3-wstool ros-noetic-ddynamic-reconfigure

# Install Azure Kinect & Body Tracking SDK
RUN wget -q https://packages.microsoft.com/ubuntu/18.04/prod/pool/main/libk/libk4a1.4/libk4a1.4_1.4.1_amd64.deb && \
    wget -q https://packages.microsoft.com/ubuntu/18.04/prod/pool/main/libk/libk4a1.4-dev/libk4a1.4-dev_1.4.1_amd64.deb && \
    wget -q https://packages.microsoft.com/ubuntu/18.04/prod/pool/main/libk/libk4abt1.1/libk4abt1.1_1.1.2_amd64.deb && \
    wget -q https://packages.microsoft.com/ubuntu/18.04/prod/pool/main/libk/libk4abt1.1-dev/libk4abt1.1-dev_1.1.2_amd64.deb && \
    wget -q https://packages.microsoft.com/ubuntu/18.04/prod/pool/main/k/k4a-tools/k4a-tools_1.4.1_amd64.deb && \
    echo 'libk4a1.4 libk4a1.4/accepted-eula-hash string 0f5d5c5de396e4fee4c0753a21fee0c1ed726cf0316204edda484f08cb266d76' | debconf-set-selections && \
    echo 'libk4abt1.1 libk4abt1.1/accepted-eula-hash string 03a13b63730639eeb6626d24fd45cf25131ee8e8e0df3f1b63f552269b176e38' | debconf-set-selections && \
    dpkg -i libk4a1.4_1.4.1_amd64.deb && rm libk4a1.4_1.4.1_amd64.deb && \
    dpkg -i libk4a1.4-dev_1.4.1_amd64.deb && rm libk4a1.4-dev_1.4.1_amd64.deb && \
    dpkg -i libk4abt1.1_1.1.2_amd64.deb && rm libk4abt1.1_1.1.2_amd64.deb && \
    dpkg -i libk4abt1.1-dev_1.1.2_amd64.deb && rm libk4abt1.1-dev_1.1.2_amd64.deb && \
    dpkg -i k4a-tools_1.4.1_amd64.deb && rm k4a-tools_1.4.1_amd64.deb && \
    mkdir -p /etc/udev/rules.d && wget -q https://raw.githubusercontent.com/microsoft/Azure-Kinect-Sensor-SDK/develop/scripts/99-k4a.rules -O /etc/udev/rules.d/99-k4a.rules

# Install clean up apt cache
RUN apt-get -y autoremove && apt-get clean autoclean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Reset noninteractive mode for apt-get
ENV DEBIAN_FRONTEND=dialog


###############################################################################
# Install developing tools & dependencies
###############################################################################

# Set environment variables
ENV PATH=/usr/local/cuda/bin:$PATH
ENV LD_LIBRARY_PATH=/usr/lib64:/usr/local/cuda/lib64:/usr/local/cuda/lib:/usr/local/cuda/extras/CUPTI/lib64:${LD_LIBRARY_PATH}
ENV TORCH_CUDA_ARCH_LIST=${CUDA_ARCH}
ENV FORCE_CUDA=1
ENV OPENCV_IO_ENABLE_OPENEXR=1

# Install Eigen3 3.4.0
RUN wget -q https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz && tar -xzf eigen-3.4.0.tar.gz && \
    cd eigen-3.4.0 && mkdir build && cd build && \
    cmake .. -Wno-dev -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS=-std=c++14 && \
    make install && \
    cd / && rm -rf eigen-3.4.0 eigen-3.4.0.tar.gz

# Install Yaml-CPP 0.7.0
RUN git clone --recursive https://github.com/jbeder/yaml-cpp && \
    cd yaml-cpp && git checkout yaml-cpp-0.7.0 && \
    mkdir build && cd build && \
    cmake .. -Wno-dev -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS=-std=c++14 -DBUILD_TESTING=OFF -DINSTALL_GTEST=OFF -DYAML_CPP_BUILD_TESTS=OFF -DYAML_BUILD_SHARED_LIBS=ON && \
    make -j$(nproc) && \
    make install && \
    cd / && rm -rf yaml-cpp

# Install Pybind11 2.11.1
RUN git clone --recursive https://github.com/pybind/pybind11 && \
    cd pybind11 && git checkout v2.11.1 && \
    mkdir build && cd build && \
    cmake .. -Wno-dev -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS=-std=c++14 -DPYBIND11_INSTALL=ON -DPYBIND11_TEST=OFF && \
    make -j$(nproc) && \
    make install && \
    cd / && rm -rf pybind11

# Install OpenCV 3.4.16 with CUDA support
RUN git clone --recursive https://github.com/opencv/opencv && cd opencv && git checkout 3.4.16
RUN git clone --recursive https://github.com/opencv/opencv_contrib.git && cd opencv_contrib && git checkout 3.4.16
RUN cd opencv && mkdir build && cd build && \
    cmake .. -Wno-dev -DCMAKE_BUILD_TYPE=Release -DOPENCV_EXTRA_MODULES_PATH=/opencv_contrib/modules -DCMAKE_CXX_FLAGS=-std=c++14 -DOPENCV_ENABLE_NONFREE=ON -DENABLE_PRECOMPILED_HEADERS=OFF -DWITH_MATLAB=OFF -DWITH_QT=ON -DWITH_OPENCL=OFF -WITH_OPENEXR=ON -DWITH_TIFF=OFF -DWITH_WEBP=OFF -DWITH_OPENMP=ON -DMKL_WITH_OPENMP=ON -DWITH_CUDA=ON -DCUDA_ARCH_BIN="${CUDA_ARCH}" -DCUDA_FAST_MATH=ON -DENABLE_FAST_MATH=ON -DBUILD_CUDA_STUBS=OFF -DBUILD_DOCS=OFF -DOPENCV_DNN_OPENCL=OFF -DINSTALL_PYTHON_EXAMPLES=OFF -DBUILD_opencv_cudacodec=OFF -DBUILD_opencv_xfeatures2d=ON -DBUILD_opencv_java=OFF -DBUILD_opencv_python=OFF -DBUILD_opencv_python2=OFF -DBUILD_opencv_python3=OFF && \
    make -j$(nproc) && \
    make install && \
    cd / && rm -rf opencv opencv_contrib

# Install PCL 1.10.1
RUN git clone --recursive https://github.com/PointCloudLibrary/pcl && \
    cd pcl && git checkout pcl-1.10.1 && \
    mkdir build && cd build && \
    cmake .. -Wno-dev -DCMAKE_BUILD_TYPE=Release -DBUILD_GPU=OFF -DBUILD_CUDA=OFF -DBUILD_apps=OFF -DBUILD_examples=OFF -DBUILD_global_tests=OFF -DBUILD_simulation=OFF -DCUDA_BUILD_EMULATION=OFF -DCMAKE_CXX_FLAGS=-std=c++14 -DPCL_ENABLE_SSE=ON -DPCL_SHARED_LIBS=ON && \
    make -j$(nproc) && \
    make install && \
    cd / && rm -rf pcl

# Install librealsense 2.50.0
RUN git clone --recursive https://github.com/IntelRealSense/librealsense.git && \
    cd librealsense && git checkout v2.50.0
RUN cd librealsense && mkdir build && cd build && \
    cmake .. -Wno-dev -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr/local -DCMAKE_CXX_FLAGS=-std=c++14 -DENABLE_CCACHE=OFF -DBUILD_WITH_OPENMP=ON -DBUILD_WITH_CUDA=ON -DENFORCE_METADATA=ON -DBUILD_PYTHON_BINDINGS=OFF -DBUILD_SHARED_LIBS=ON -DBUILD_WITH_TM2=OFF -DIMPORT_DEPTH_CAM_FW=OFF && \
    make -j$(nproc) && \
    make install && \
    mkdir -p /etc/udev/rules.d && cp /librealsense/config/99-realsense-libusb.rules /etc/udev/rules.d/ && \
    cd / && rm -rf librealsense

# Install miniconda3
RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh && bash miniconda.sh -b -p /opt/conda && rm miniconda.sh && \
    source /opt/conda/etc/profile.d/conda.sh && \
    conda config --set ssl_verify no && \
    conda config --add channels conda-forge && \
    conda config --remove channels defaults && \
    conda update -n base -c conda-forge conda -qy && \
    conda clean -afqy


###############################################################################
# Add non-root user
###############################################################################
ARG USERNAME="my_user"
ARG UID
ARG GID

RUN addgroup --gid $GID ${USERNAME}
RUN adduser --disabled-password --home /home/${USERNAME} --shell /bin/zsh --gecos '' --uid $UID --gid $GID ${USERNAME} && \
    echo "${USERNAME} ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers && \
    # create catkin workspace
    mkdir -p /opt/catkin_ws && \
    chown -R ${USERNAME}:${USERNAME} /opt/catkin_ws && \
    chown -R ${USERNAME}:${USERNAME} /opt/conda


# ------------------- Run container as non-root user from here onwards ------------------- #
SHELL ["/bin/zsh", "--login", "-c"]
USER ${USERNAME}

# Set working directory
WORKDIR /home/${USERNAME}

# Install oh-my-zsh
RUN /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)" "" --unattended && \
    sed -i 's/robbyrussell/candy/g' /home/${USERNAME}/.zshrc && \
    zstyle ':omz:update' mode disabled

###############################################################################
# Install ros-packages
###############################################################################

RUN echo "source /opt/ros/noetic/setup.zsh" >> /home/${USERNAME}/.zshrc && \
    # Initialize catkin workspace
    source /home/${USERNAME}/.zshrc && sudo rosdep init && rosdep update && \
    mkdir -p /opt/catkin_ws/src && cd /opt/catkin_ws/src/ && catkin_init_workspace && \
    # Download Realsense-ROS 2.3.2
    cd /opt/catkin_ws/src && git clone --recursive https://github.com/IntelRealSense/realsense-ros.git && cd realsense-ros && git checkout 2.3.2 && \
    # Download Azure_Kinect_ROS_Driver
    sudo sed -i 's/find_dependency(k4a 1.4 REQUIRED)/find_dependency(k4a 1.4.1 REQUIRED)/g' /usr/lib/cmake/k4abt/k4abtConfig.cmake && \
    cd /opt/catkin_ws/src && git clone --recursive https://github.com/microsoft/Azure_Kinect_ROS_Driver.git && \
    # Compile ROS packages
    cd /opt/catkin_ws && catkin_make --force-cmake -DCMAKE_BUILD_TYPE=Release -DCATKIN_ENABLE_TESTING=False && catkin_make install && \
    echo "source /opt/catkin_ws/devel/setup.zsh" >> /home/${USERNAME}/.zshrc && \
    # Download Apriltag 3.3.0
    mkdir -p /opt/catkin_ws/apriltag_ros/src && \
    cd /opt/catkin_ws/apriltag_ros/src && git clone --recursive https://github.com/AprilRobotics/apriltag.git && cd apriltag && git checkout v3.3.0 && \
    # Download Apriltag ROS 3.2.1
    cd /opt/catkin_ws/apriltag_ros/src && git clone --recursive https://github.com/AprilRobotics/apriltag_ros.git && cd apriltag_ros && git checkout 3.2.1 && \
    # Compile ROS packages
    cd /opt/catkin_ws/apriltag_ros/ && catkin_make_isolated --install -DCMAKE_BUILD_TYPE=Release && \
    echo "source /opt/catkin_ws/apriltag_ros/devel_isolated/setup.zsh" >> /home/${USERNAME}/.zshrc

COPY ./settings.yaml /opt/catkin_ws/apriltag_ros/src/apriltag_ros/apriltag_ros/config/settings.yaml
COPY ./tags.yaml /opt/catkin_ws/apriltag_ros/src/apriltag_ros/apriltag_ros/config/tags.yaml

###############################################################################
# Create Conda environment & Install packages
###############################################################################

# Create new conda environment
ARG CONDA_ENV_NAME="py310"
RUN echo "source /opt/conda/etc/profile.d/conda.sh" >> /home/${USERNAME}/.zshrc && \
    source /home/${USERNAME}/.zshrc && \
    conda create -n ${CONDA_ENV_NAME} python=3.10 -qy && \
    echo "conda activate ${CONDA_ENV_NAME}" >> /home/${USERNAME}/.zshrc

# Install packages from requirements.txt
COPY ./requirements.txt /home/${USERNAME}/requirements.txt
RUN source /home/${USERNAME}/.zshrc && python -m pip install --quiet --no-cache-dir -r requirements.txt && rm requirements.txt && \
    # Chumpy
    python -m pip install --quiet --no-cache-dir git+https://github.com/gobanana520/chumpy.git && \
    # Modified version for PyRender
    python -m pip install --quiet --no-cache-dir git+https://github.com/mmatl/pyopengl.git && \
    # Kaolin
    python -m pip install --quiet --no-cache-dir kaolin==0.15.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.1.1_cu118.html && \
    # PyTorch3D
    python -m pip install --quiet --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu118_pyt211/download.html && \
    # SAM
    python -m pip install --quiet --no-cache-dir git+https://github.com/facebookresearch/segment-anything.git && \
    # MobileSAM
    python -m pip install --quiet --no-cache-dir git+https://github.com/ChaoningZhang/MobileSAM.git

# Change the working directory
WORKDIR /home/${USERNAME}/code
