import os


import numpy as np
import tensorflow as tf


from scripts import llama_dp
from simpler_env.evaluation.argparse import get_args
from simpler_env.evaluation.maniskill2_evaluator import maniskill2_evaluator

try:
    from simpler_env.policies.octo.octo_model import OctoInference
except ImportError as e:
    print("Octo is not correctly imported.")
    print(e)


if __name__ == "__main__":
    args = get_args()

    os.environ["DISPLAY"] = ""
    # prevent a single jax process from taking up all the GPU memory
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    gpus = tf.config.list_physical_devices("GPU")
    if len(gpus) > 0:
        # prevent a single tf process from taking up all the GPU memory
        tf.config.set_logical_device_configuration(
            gpus[0],
            [tf.config.LogicalDeviceConfiguration(memory_limit=args.tf_memory_limit)],
        )
    print(f"**** {args.policy_model} ****")
    # policy model creation; update this if you are using a new policy model
    if args.policy_model == "rt1":
        from simpler_env.policies.rt1.rt1_model import RT1Inference
        assert args.ckpt_path is not None
        model = RT1Inference(
            saved_model_path=args.ckpt_path,
            policy_setup=args.policy_setup,
            action_scale=args.action_scale,
        )
    elif "octo" in args.policy_model:
        from simpler_env.policies.octo.octo_server_model import OctoServerInference
        if args.ckpt_path is None or args.ckpt_path == "None":
            args.ckpt_path = args.policy_model
        if "server" in args.policy_model:
            model = OctoServerInference(
                model_type=args.ckpt_path,
                policy_setup=args.policy_setup,
                action_scale=args.action_scale,
            )
        else:
            model = OctoInference(
                model_type=args.ckpt_path,
                policy_setup=args.policy_setup,
                init_rng=args.octo_init_rng,
                action_scale=args.action_scale,
            )
    elif args.policy_model == "openvla":
        assert args.ckpt_path is not None
        from simpler_env.policies.openvla.openvla_model import OpenVLAInference
        model = OpenVLAInference(
            saved_model_path=args.ckpt_path,
            policy_setup=args.policy_setup,
            action_scale=args.action_scale,
            use_raw_action = args.use_raw_action,
        )
    elif args.policy_model == "dit_policy":
        assert args.ckpt_path is not None
        from simpler_env.policies.openvla.openvla_model import OpenVLAInference
        
        import hydra
        import sys
        current_path = os.getcwd()
        sys.path.append(current_path)
        sys.path.append(os.path.join(current_path, "../rt1_pytorch/"))
        from scripts.llama_dp import RobotTransformerNet
        from scripts.close_loop_eval_diffusion_simplerenv import PytorchDiffInference
        hydra.initialize(config_path=os.path.join("..", "..", "config"))  # 这里指定配置文件所在的目录

        # 使用 compose 手动加载主配置文件（config.yaml）
        cfg = hydra.compose(config_name="config_diffusion_openx")  # 加载名为 config.yaml 的文件
        import torch
        cfg.scheduler_type = 1

        if 'DENOISING_STEPS' in os.environ:
            cfg.scheduler_type = 1
        
        default_action_spec = {
        "world_vector": {
            "tensor": torch.empty((3,), dtype=torch.float32),
            "minimum": torch.tensor([-2.0], dtype=torch.float32),
            "maximum": torch.tensor([2.0], dtype=torch.float32),
        },
        "rotation_delta": {
            "tensor": torch.empty((4,), dtype=torch.float32),
            "minimum": torch.tensor([-np.pi / 2.0], dtype=torch.float32),
            "maximum": torch.tensor([np.pi / 2.0], dtype=torch.float32),
        },
        "gripper_closedness_action": {
            "tensor": torch.empty((1,), dtype=torch.float32),
            "minimum": torch.tensor([-1.0], dtype=torch.float32),
            "maximum": torch.tensor([1.0], dtype=torch.float32),
        },
        "terminate_episode": {
            "tensor": torch.empty((3,), dtype=torch.int32),
            "minimum": torch.tensor([0], dtype=torch.int32),
            "maximum": torch.tensor([1], dtype=torch.int32),
        }}
        sequence_length = 32
        num_pred_action = 31
        network = RobotTransformerNet(
            output_tensor_spec=default_action_spec,
            vocab_size=cfg.model.vocab_size,
            time_sequence_length=sequence_length,
            num_layers=cfg.model.num_layers,
            dropout_rate=cfg.model.dropout_rate,
            include_prev_timesteps_actions=cfg.model.include_prev_timesteps_actions,
            freeze_backbone=cfg.model.freeze_backbone,
            use_qformer=cfg.model.use_qformer,
            use_wrist_img=cfg.model.use_wrist_img,
            use_depth_img=cfg.model.use_depth_img,
            prediction_type=cfg.prediction_type,
            dim_align_type=cfg.dim_align_type if 'dim_align_type' in cfg else 0,
            use_action_head_diff=cfg.use_action_head_diff,
            scheduler_type=cfg.scheduler_type,
            trajectory_dim=7,
            num_of_obs=2 if 'no_history_action' in cfg else (sequence_length - num_pred_action + 1) 
        )
        ckpt = torch.load(args.ckpt_path, 'cpu')
        print(network.load_state_dict(ckpt["parameter"]))
        network = network.cuda()
        
        model = PytorchDiffInference(model=network, sequence_length = sequence_length, use_wrist_img=cfg.model.use_wrist_img,
                                 num_pred_action=num_pred_action, stride=1, use_action_head_diff=cfg.use_action_head_diff,
                                 policy_setup=args.policy_setup,)
        print('begin init model....')
    elif args.policy_model == "cogact":
        from simpler_env.policies.sim_cogact import CogACTInference
        assert args.ckpt_path is not None
        model = CogACTInference(
            saved_model_path=args.ckpt_path,  # e.g., CogACT/CogACT-Base
            policy_setup=args.policy_setup,
            action_scale=args.action_scale,
            action_model_type='DiT-L',
            cfg_scale=1.5                     # cfg from 1.5 to 7 also performs well
        )
    else:
        raise NotImplementedError()

    # run real-to-sim evaluation
    success_arr = maniskill2_evaluator(model, args)
    print(args)
    print(" " * 10, "Average success", np.mean(success_arr))
