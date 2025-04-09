model_name=openvla
tasks=(
  pick_coke_can_variant_agg1.sh
)

ckpts=(
  openvla/openvla-7b
)

action_ensemble_temp=-0.8
for ckpt_path in ${ckpts[@]}; do
  base_dir=$(dirname $ckpt_path)

  # evaluation in simulator
  # logging_dir=$base_dir/simpler_env/$(basename $ckpt_path)${action_ensemble_temp}
  logging_dir=results1/$(basename $ckpt_path)${action_ensemble_temp}
  mkdir -p $logging_dir
  for i in ${!tasks[@]}; do
    task=${tasks[$i]}
    echo "🚀 running $task ..."
    device=0
    session_name=CUDA${device}-$(basename $logging_dir)-${task}
    bash scripts/$task $ckpt_path $model_name $action_ensemble_temp $logging_dir $device
  done
  # statistics evalution results
  echo "🚀 all tasks DONE! Calculating metrics..."
  /mnt/petrelfs/houzhi/anaconda/envs/rt_dp/bin/python tools/calc_metrics_evaluation_videos.py \
    --log-dir-root $logging_dir \
    >>$logging_dir/total.metrics
done
