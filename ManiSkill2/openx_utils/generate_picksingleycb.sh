export PYTHONPATH="$(pwd)":"$(pwd)/../":$PYTHONPATH

START_INDEX=$1
END_INDEX=$2


MS2_ASSET_DIR="/.../maniskill2/assets" \
SAVE_TO_CEPH="True" \
CEPH_SAVE_BUCKET="s3://your save path" \
python picksingleycb_bridge.py --num_proc 20 --start_model_index $START_INDEX --end_model_index $END_INDEX
# -e StackCube-v0
