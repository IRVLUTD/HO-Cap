#! /bin/bash
# author: Jikai Wang
# email: jikai.wang AT utdallas DOT edu

CONDA_PYTHON_PATH="${HOME}/mambaforge/envs/ho-cap/bin/python"
CURR_DIR=$(realpath $( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd ))

SCRIPT_FILE="${CURR_DIR}/sequence_render.py"

# Set the GPU ID
if [ -z "$1" ]
then
    GPU_ID=0
else
    GPU_ID=$1
fi

# Sequences
ALL_SEQUENCES=(
/home/jikaiwang/GitHub/HandObjectPipeline/data/HOT_Dataset/subject_1/20231025_165502
/home/jikaiwang/GitHub/HandObjectPipeline/data/HOT_Dataset/subject_1/20231025_165807
/home/jikaiwang/GitHub/HandObjectPipeline/data/HOT_Dataset/subject_1/20231025_170105
/home/jikaiwang/GitHub/HandObjectPipeline/data/HOT_Dataset/subject_1/20231025_170650

/home/jikaiwang/GitHub/HandObjectPipeline/data/HOT_Dataset/subject_1/20231025_170959
/home/jikaiwang/GitHub/HandObjectPipeline/data/HOT_Dataset/subject_2/20231022_200657
/home/jikaiwang/GitHub/HandObjectPipeline/data/HOT_Dataset/subject_2/20231022_202617
/home/jikaiwang/GitHub/HandObjectPipeline/data/HOT_Dataset/subject_2/20231022_203100

/home/jikaiwang/GitHub/HandObjectPipeline/data/HOT_Dataset/subject_2/20231023_164741
/home/jikaiwang/GitHub/HandObjectPipeline/data/HOT_Dataset/subject_3/20231024_154531
/home/jikaiwang/GitHub/HandObjectPipeline/data/HOT_Dataset/subject_3/20231024_154810
/home/jikaiwang/GitHub/HandObjectPipeline/data/HOT_Dataset/subject_3/20231024_161306

/home/jikaiwang/GitHub/HandObjectPipeline/data/HOT_Dataset/subject_3/20231024_162409
/home/jikaiwang/GitHub/HandObjectPipeline/data/HOT_Dataset/subject_4/20231026_162155
/home/jikaiwang/GitHub/HandObjectPipeline/data/HOT_Dataset/subject_4/20231026_162248
/home/jikaiwang/GitHub/HandObjectPipeline/data/HOT_Dataset/subject_4/20231026_163223

/home/jikaiwang/GitHub/HandObjectPipeline/data/HOT_Dataset/subject_4/20231026_164812
/home/jikaiwang/GitHub/HandObjectPipeline/data/HOT_Dataset/subject_6/20231025_112546
/home/jikaiwang/GitHub/HandObjectPipeline/data/HOT_Dataset/subject_7/20231022_190534
/home/jikaiwang/GitHub/HandObjectPipeline/data/HOT_Dataset/subject_7/20231022_192832

/home/jikaiwang/GitHub/HandObjectPipeline/data/HOT_Dataset/subject_7/20231022_193506
/home/jikaiwang/GitHub/HandObjectPipeline/data/HOT_Dataset/subject_7/20231022_193630
/home/jikaiwang/GitHub/HandObjectPipeline/data/HOT_Dataset/subject_7/20231022_193809
/home/jikaiwang/GitHub/HandObjectPipeline/data/HOT_Dataset/subject_7/20231023_162803

/home/jikaiwang/GitHub/HandObjectPipeline/data/HOT_Dataset/subject_7/20231023_163653
/home/jikaiwang/GitHub/HandObjectPipeline/data/HOT_Dataset/subject_8/20231024_181413
/home/jikaiwang/GitHub/HandObjectPipeline/data/HOT_Dataset/subject_9/20231027_123403
/home/jikaiwang/GitHub/HandObjectPipeline/data/HOT_Dataset/subject_9/20231027_123725

/home/jikaiwang/GitHub/HandObjectPipeline/data/HOT_Dataset/subject_9/20231027_123814
/home/jikaiwang/GitHub/HandObjectPipeline/data/HOT_Dataset/subject_9/20231027_124057
/home/jikaiwang/GitHub/HandObjectPipeline/data/HOT_Dataset/subject_9/20231027_125019
)

# Run Pose Solver
for SEQUENCE in ${ALL_SEQUENCES[@]} ; do
    echo "###############################################################################"
    echo "# Rendering Sequence ${SEQUENCE}"
    echo "###############################################################################"
    CUDA_VISIBLE_DEVICES=${GPU_ID} ${CONDA_PYTHON_PATH} ${SCRIPT_FILE} \
        --sequence_folder ${SEQUENCE}
done