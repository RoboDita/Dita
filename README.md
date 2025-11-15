# Dita: Scaling Diffusion Transformer for Generalist Vision-Language-Action Policy

[![arXiv](https://img.shields.io/badge/arXiv-2503.19757-df2a2a.svg)](http://arxiv.org/abs/2503.19757) 
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Static Badge](https://img.shields.io/badge/Project-Page-a)](https://RoboDita.github.io/)


### Installation

To run the code, you should install the requiresments. The code is run on python3.9 and pytorch 2, tensorflow==2.15.0, CUDA 12.1. 

clone the code as follow,

```
git clone https://github.com/RoboDita/Dita
```

then, please consider install the base environment,

```
 pip install -r requirements_base.txt
```

If you only evaluate it on calvin, you might use requirements_calvin.txt. To avoid the conflicts, we suggest you install tensorflow-probability==0.22.0 independently.

Meanwhile, you might need to install pytorch3d (This is not necessary for pretraining). We build pytorch3d from this  git+https://github.com/facebookresearch/pytorch3d.git@89653419d0973396f3eff1a381ba09a07fffc2ed#egg=pytorch3d.



### Model Checkpoints

We provide the corresponding models, that can be utilized for finetuing.



| Model        |Description                                                                                                 | Checkpoint Path                                |
| ------------ | ----------------------------------------------------------------------------------------------------------- | ---------------------------------------------- |
| Dita    |  Diffusion Transformer Policy | [Google Drive](https://drive.google.com/file/d/1jaaoT0QGX4xwdzvTr_ki8OJ9-XkNOvub/view?usp=sharing)      |
| Dita    |  Diffusion Transformer Policy(Pretrained on Droid) | [Google Drive](https://drive.google.com/file/d/1a3nkXCieQL7GaemZH_NFpXYNd_LO3_ZU/view?usp=sharing) |
| Dita    |  Diffusion Transformer Policy (w/o image augmentation) | [Google Drive](https://drive.google.com/file/d/1qpyDYsMrUISve9koP-4_BCSEFgthn70P/view?usp=sharing)      |
| Diffusion MLP Head | Transformer with Diffusion Head Policy (w/o image augmentation)  | [Google Drive](https://drive.google.com/file/d/1vdWLre4v_MlNEEII6Z97VLGH-3yxmr1O/view?usp=sharing) |

## Training & Finetuning

### PRETRAINING on OXE dataset

Before you run the code, you should update the s3 key "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "S3_ENDPOINT". We train the network with 32 GPUs. Meanwhile, please update 'data_path = "s3://openx"' in `scripts/train_diffusion_oxe.py'. In our experiments, we do not find a significant difference between dataset.traj_length=32 and dataset.traj_length=16 on OXE pretraining. Therefore, we suggest `dataset.traj_length=16 num_pred_action=15


```
torchrun --nproc_per_node=8 --nnodes=4 --node_rank=0 --master_port=$MASTER_PORT --master_addr=$MASTER_ADDR scripts/train_diffusion_oxe.py task_name=openx_full_train_o2_p32 dataset.traj_length=16 num_pred_action=15 scheduler_type=1 shuffle_buffer_size=256000 dataname=oxe_magic_soup_plus task_name=oxe_full_train_o2_p15_wotimestep_oxe_noclamp_filter batch_size=256 
```

```
torchrun --nproc_per_node=8 --nnodes=4 --node_rank=1 --master_port=$MASTER_PORT --master_addr=$MASTER_ADDR scripts/train_diffusion_oxe.py task_name=openx_full_train_o2_p32 dataset.traj_length=16 num_pred_action=15 scheduler_type=1 shuffle_buffer_size=256000 dataname=oxe_magic_soup_plus task_name=oxe_full_train_o2_p15_wotimestep_oxe_noclamp_filter batch_size=256 
```

```
torchrun --nproc_per_node=8 --nnodes=4 --node_rank=2 --master_port=$MASTER_PORT --master_addr=$MASTER_ADDR scripts/train_diffusion_oxe.py task_name=openx_full_train_o2_p32 dataset.traj_length=16 num_pred_action=15 scheduler_type=1 shuffle_buffer_size=256000 dataname=oxe_magic_soup_plus task_name=oxe_full_train_o2_p15_wotimestep_oxe_noclamp_filter batch_size=256 
```


```
torchrun --nproc_per_node=8 --nnodes=4 --node_rank=3 --master_port=$MASTER_PORT --master_addr=$MASTER_ADDR scripts/train_diffusion_oxe.py task_name=openx_full_train_o2_p32 dataset.traj_length=16 num_pred_action=15 scheduler_type=1 shuffle_buffer_size=256000 dataname=oxe_magic_soup_plus task_name=oxe_full_train_o2_p15_wotimestep_oxe_noclamp_filter batch_size=256 
```



We observe that image augmentation is beneficial for SimplerEnv in our experiments. If you want to use image augmentation, please add ``+image_aug=1''

### Finetuning with Lora

Here, we provide an example for finetuning with lora, i.e., the 10-shot finetuning code on Real-Franka Arm.

```

torchrun --nproc_per_node=4 --nnodes=1 --node_rank=0 scripts/finetune_realdata.py +pretrained_path=dit_policy_checkpoint.pth dataset.traj_per_episode=16 dataset.traj_length=1 task_name=new_test_nodiffhead_few10_250124 num_pred_action=1 dataname=lab_907_1 batch_size=32 dataset.train_data_list=you pkl dataname file to include the collected pkl files name use_lora=True scheduler_type=0 dataset.num_given_observation=1  max_iters=10000
```

scheduler_type=0 indicates we use 100 DDPM training steps.

**We would like to highlight that the finetuning without lora can achieve cleanly better performance (robustness to variances, more horizon tasks).** 

```
torchrun --nproc_per_node=4 --nnodes=1 --node_rank=0 scripts/finetune_realdata.py +pretrained_path=dit_policy_checkpoint.pth dataset.traj_per_episode=16 dataset.traj_length=1 task_name=new_test_nodiffhead_few10_250124 num_pred_action=1 dataname=lab_907_1 batch_size=32 dataset.train_data_list=you pkl dataname file to include the collected pkl files name use_lora=False scheduler_type=0 dataset.num_given_observation=1  max_iters=10000
```

### Fully Finetuning on CALVIN

At first, you should follow the [instruction-calvin](https://github.com/mees/calvin) to install CALVIN environment.

we train the network with 4GPUs.

```
torchrun --nproc_per_node=4 --nnodes=1 --node_rank=0 scripts/train_diffusion_sim.py --config-name config_diffusion_calvin batch_size=32 dataset.traj_length=11 num_pred_action=10 task_name=calvin_exp dataset.num_given_observation=2 use_close_loop_eval=True close_loop_eval.test_episodes_num=32 taskname=task_ABC_D dataname=calvin_mc close_loop_eval.eval_iters=10000 close_loop_eval.test_episodes_num=250 scheduler_type=0 wrap_grmg_data=2 +pretrained_path=dit_policy_checkpoint.pth +use_adjust_scheduler=true lr=0.0001 epoch=15 +min_lr_scale=0.01 scheduler.warmup_epochs=1 num_inference_steps=10
```

### Finetuning on LIBERO

Firstly, please follow [OpenVLA](https://github.com/openvla/openvla?tab=readme-ov-file#libero-simulation-benchmark-evaluations) to set up the LIBERO benchmark and get the modified version of the dataset. 

We train and evaluate the model with 8 NVIDIA GPUs. 

Here is an example of the training script.

```
torchrun --nproc_per_node=8 --nnodes=1 --node_rank=0  scripts/train_diffusion_oxe.py task_name=finetuning_LIBERO dataname=libero_spatial_no_noops dataset.traj_length=11 num_pred_action=10 scheduler_type=0 shuffle_buffer_size=128000 batch_size=64 use_close_loop_eval=True +trajectory_dim=7 +pretrained_path=dit_policy_checkpoint.pth +use_adjust_scheduler=true lr=0.0001 +min_lr_scale=0.01 +image_aug=true num_inference_steps=10
```



```
torchrun --nproc_per_node=8 --nnodes=1 --node_rank=0  scripts/train_diffusion_oxe.py task_name=finetuning_LIBERO dataname=libero_10_no_noops dataset.traj_length=11 num_pred_action=10 scheduler_type=0 shuffle_buffer_size=128000 batch_size=64 use_close_loop_eval=True +trajectory_dim=7 +pretrained_path=dit_policy_checkpoint.pth +use_adjust_scheduler=true lr=0.0005 +min_lr_scale=0.01 +image_aug=true num_inference_steps=10
```

### Simulation Benchmark Evaluations

#### LIBERO Simulation Benchmark Evaluations

| Method | LIBERO-Spatial | LIBERO-Object | LIBERO-Goal | LIBERO-Long | Average |
|--------|----------------|---------------|-------------|-------------|---------|
| Diffusion Policy from scratch | 78.3 | 92.5% | 68.3 % | 50.5 % | 72.4 % |
| Octo fine-tuned | 78.9 % | 85.7 % | 84.6% | 51.1 % | 75.1 % |
| OpenVLA fine-tuned| **84.7 %** | 88.4 % | 79.2 % | 53.7 % | 76.5 % |
| ours fine-tuned| 84.2% | **96.3%** | **85.4%** | **63.8%** | **82.4%**


#### Calvin (ABC->D)

| Method | Input | 1 | 2 | 3 | 4 | 5| Avg.Len.
|--------|----------------|----------------|----------------|----------------|---------------|-------------|-------------|
| RoboFlamingo      | S-RGB, G-RGB              | 82.4% | 61.9% | 46.6%   | 33.1%   | 23.5%   | 2.47  |
| SuSIE             | S-RGB                     | 87.0% | 69.0% | 49.0%   | 38.0%   | 26.0%   | 2.69  |
| GR-1              | S-RGB, G-RGB, P          | 85.4% | 71.2% | 59.6%   | 49.7%   | 40.1%   | 3.06  |
| 3D Diffuser       | S-RGBD, G-RGBD, Proprio, Cam | 92.2% | 78.7% | 63.9%   | 51.2%   | 41.2%   | 3.27  |
| ours w/o pretraining | Static-RGB | 89.5% | 63.3%  |39.8%  |27.3%  |18.5%  | 2.38
| ours | Static-RGB | **94.5%** | **82.5%**|  **72.8%**|  **61.3%**|  **50.0%**|  **3.61**| 

Here is the checkpoint for calvin (ABC->D)[calvin_abc-d.pth](https://drive.google.com/file/d/1L3X3op-oVNVVGEw8EdpiiedYp0I0gzXP/view?usp=sharing)

You can eval the model as follows,

```
PYTHONPATH=. python3 scripts/train_diffusion_sim.py --config-name config_diffusion_calvin batch_size=32 dataset.traj_length=11 num_pred_action=10 task_name=calvin_exp dataset.num_given_observation=2 use_close_loop_eval=True close_loop_eval.test_episodes_num=32 taskname=task_ABC_D dataname=calvin_mc close_loop_eval.eval_iters=10000 close_loop_eval.test_episodes_num=250 scheduler_type=0 +wrap_grmg_data=2 +pretrained_path=calvin_abc-d.pth +use_adjust_scheduler=true lr=0.0001 epoch=15 +min_lr_scale=0.01 scheduler.warmup_epochs=1 num_inference_steps=10 +eval_only=1
```

The evaluation environment requirements are provided in calvin_python3.10_requirements.txt. You can refer this requirements to config your environment for calvin.

Simulation Benchmark Evaluations

#### SimplerEnv

This evaluation is based on [SimplerEnv](https://github.com/simpler-env/SimplerEnv)


| models                                             | Dita(ours)                 | RT-1-X |  Octo-Base | OpenVLA |
|:---------------------------------------------------|:---------------------|:-------|:-------|:----------|
| coke_can/matching                             | **0.837**   | 0.567  |  0.17      | 0.163   |
| coke_can/variant                               | **0.855**   | 0.490   |  0.006     | 0.545   |
| move_near/matching                                 | **0.760**  | 0.317  | 0.042     | 0.462   |
| move_near/variant                                  | **0.730**   | 0.323  | 0.031     |  0.477   |
| drawer/matching                               | 0.463   | **0.597**  | 0.227     |  0.356   |
| drawer/variant                                | **0.375**   | 0.294  | 0.011     | 0.177   |



#### Real Franka Demonstration

Please refer to the [project page](https://RoboDita.github.io/).

## Usage for [OC-VLA](https://arxiv.org/pdf/2508.13103)
### Training
This Codebase can also be used to run [OC-VLA](https://arxiv.org/pdf/2508.13103), set the `use_baseframe_action=False` in the config and codes to training the model in the third-person camera coordinate. Here is an example:

```
torchrun --nproc_per_node=8 --nnodes=1 --node_rank=0 scripts/train_diffusion_sim.py batch_size=64 dataset.traj_length=11 num_pred_action=10 task_name=maniskill dataset.num_given_observation=2 use_close_loop_eval=True close_loop_eval.test_episodes_num=175 dataset.use_baseframe_action=False dataname=maniskill dataset.include_target=1
```
You can also use the `scripts/train_discrete_sim.py` to run a model predicting the actions in discrete domain with a similar way.

## Data Generation in ManiSkill2
Both Dita and [OC-VLA](https://arxiv.org/pdf/2508.13103) use a customized dataset generated by ourslves in ManiSkill2. This dataset can be generated as follows. Codes can be found in [ManiSkill2](./ManiSkill2/) folder.

- Download the assets from the offical repo of [ManiSkill2](https://github.com/haosulab/ManiSkill/tree/v0.5.3).

- Unzip. Put the folder named `assets` to the `ManiSkill2/openx_utils/data` and put the folder named `demos` to the `ManiSkill2/openx_utils/demos`.

- Use the script under `ManiSkill2/openx_utils/` to generate the data according to the tasks separately. For example:
    ```
    sh ManiSkill2/openx_utils/generate_pickcube.sh
    ```
    Modify and Edit several Environment Variables before you use it, such as `PYTHONPATH` and path to save you data etc.

    Multiprocessing is recommended or it will take several days to generate.  

## Acknowledgement

The dataloader code of OXE and part of the code of libero setup are based on [OpenVLA](https://github.com/openvla/openvla), The dataloader code of CALVIN is based on [GR-MG](https://github.com/bytedance/GR-MG), The architecture is based on transformers. If you have any questions, feel free to contact Zhi Hou (zhou9878 at uni dot sydney dot edu dot au) or Tianyi Zhang (tianyizhang0213 at zju dot edu dot cn)



## Citation

If you find our code or models useful in your work, please consider to cite our paper:

```
@article{hou2025dita,
 title={Dita: Scaling Diffusion Transformer for Generalist Vision-Language-Action Policy},
 author={Hou, Zhi and Zhang, Tianyi and Xiong, Yuwen and Duan, Haonan and Pu, Hengjun and Tong, Ronglei and Zhao, Chengyang and Zhu, Xizhou and Qiao, Yu and Dai, Jifeng and Chen, Yuntao},
 journal={arXiv preprint arXiv:2503.19757},
 year={2025}
}

@article{zhang2025grounding,
  title={Grounding actions in camera space: Observation-centric vision-language-action policy},
  author={Zhang, Tianyi and Duan, Haonan and Hao, Haoran and Qiao, Yu and Dai, Jifeng and Hou, Zhi},
  journal={arXiv preprint arXiv:2508.13103},
  year={2025}
}
```

An earlier preprint version is on [Arxiv](https://arxiv.org/abs/2410.15959). 
