#!/bin/bash

PROJ_ROOT=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# Build meshsdf_loss
echo "############################################################"
echo "# Building meshsdf_loss"
echo "############################################################"
cd ${PROJ_ROOT}/external/mycuda/meshsdf_loss && \
rm -rf build *egg* *.so && \
time python -m pip install -e .
