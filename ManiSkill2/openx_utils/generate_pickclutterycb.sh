export PYTHONPATH="$(pwd)":"$(pwd)/../":$PYTHONPATH


MS2_ASSET_DIR="/xxx/xxx/share_data/Anonymous/maniskill2/assets" \
SAVE_TO_CEPH="True" \
CEPH_SAVE_BUCKET="s3://your save path" \
python pickclutterycb_bridge.py --num_proc 20
# -e StackCube-v0
