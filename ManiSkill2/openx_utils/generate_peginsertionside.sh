export PYTHONPATH="$(pwd)":"$(pwd)/../":$PYTHONPATH


SAVE_TO_CEPH="True" \
CEPH_SAVE_BUCKET="s3://your default save path" \
python peginsertionside_bridge.py --num_proc 20
# -e StackCube-v0
