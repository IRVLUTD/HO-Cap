#!/bin/bash

PROJ_ROOT=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

LOG_FILE=${PROJ_ROOT}/build.log

# Build manopath
echo "############################################################"
echo "# Building manopath"
echo "############################################################"

cd ${PROJ_ROOT}/external/manopth && \
rm -rf build *egg* *.so && \
time python -m pip install -e .

# Build meshsdf_loss
echo "############################################################"
echo "# Building meshsdf_loss"
echo "############################################################"
cd ${PROJ_ROOT}/external/mycuda/meshsdf_loss && \
rm -rf build *egg* *.so && \
time python -m pip install -e .

# # Build mycuda (for BundleSDF)
# echo "############################################################"
# echo "# Building mycuda (for BundleSDF)"
# echo "############################################################"

# cd ${PROJ_ROOT}/external/mycuda && \
# rm -rf build *egg* *.so && \
# time python -m pip install -e .

# # Build BundleTrack
# echo "############################################################"
# echo "# Building BundleTrack (for BundleSDF)"
# echo "############################################################"

# cd ${PROJ_ROOT}/lib/BundleSDF/BundleTrack && \
# rm -rf build && mkdir build && cd build && \
# cmake .. -Wno-dev > ${LOG_FILE} 2>&1 && \
# time make -j$(nproc) >> ${LOG_FILE} 2>&1
