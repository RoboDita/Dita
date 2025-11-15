export PYTHONPATH="$(pwd)":"$(pwd)/../":$PYTHONPATH


CEPH_SAVE_BUCKET="s3://your default save path" \
python check_and_generate_filelist.py
