
export PYTHONPATH="$(pwd)":"$(pwd)/openx_utils/":"$(pwd)/../":"$(pwd)/../embodied_foundation/rt1_pytorch":$PYTHONPATH


python openx_utils/evaluate_from_pose.py -e PickSingleYCB-v0 -t 1

