# Dita: Scaling Diffusion Transformer for Generalist Vision-Language-Action Policy

[![arXiv](https://img.shields.io/badge/arXiv-2406.09246-df2a2a.svg)](https://arxiv.org/abs/2410.15959) 
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Static Badge](https://img.shields.io/badge/Project-Page-a)](https://RoboDita.github.io/)


### Installation

To run the code, you should install the requiresments. The code is run on python3.10 and pytorch 2.2.0, tensorflow==2.15.0, CUDA 12.1.

```
 pip install -r requirements.txt
```


Then, clone the code as follow,

```
git clone https://github.com/RoboDita/Dita
```


### Model Checkpoints

We provide the corresponding models, that can be utilized for finetuing.



| Model        |Description                                                                                                 | Checkpoint Path                                |
| ------------ | ----------------------------------------------------------------------------------------------------------- | ---------------------------------------------- |
| Dita    |  Diffusion Transformer Policy | [Google Drive](https://drive.google.com/file/d/1jaaoT0QGX4xwdzvTr_ki8OJ9-XkNOvub/view?usp=sharing)      |
| Dita    |  Diffusion Transformer Policy (w/o image augmentation) | [Google Drive](https://drive.google.com/file/d/1qpyDYsMrUISve9koP-4_BCSEFgthn70P/view?usp=sharing)      |
| Diffusion MLP Head | Transformer with Diffusion Head Policy (w/o image augmentation)  | [Google Drive](https://drive.google.com/file/d/1vdWLre4v_MlNEEII6Z97VLGH-3yxmr1O/view?usp=sharing) |

## Training & Finetuning

### PRETRAINING on OXE dataset

Before you run the code, you should update the s3 key "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "S3_ENDPOINT". We train the network with 32 GPUs. 


```
python scripts/train_diffusion_oxe.py task_name=openx_full_train_o2_p32 dataset.traj_length=32 num_pred_action=31 scheduler_type=1 shuffle_buffer_size=128000 dataname=oxe_magic_soup_plus task_name=oxe_full_train_o2_p32_wotimestep_oxe_noclamp_filter batch_size=256 
```

We observe that image augmentation is beneficial for SimplerEnv in our experiments. If you want to use image augmentation, please add ``+image_aug=1''

### Finetuning with Lora

Here, we provide an example for finetuning with lora, i.e., the 10-shot finetuning code on Real-Franka Arm.

```

python3 scripts/finetune_realdata.py +pretrained_path=dit_policy_checkpoint.pth dataset.traj_per_episode=16 dataset.traj_length=1 task_name=new_test_nodiffhead_few10_250124 num_pred_action=1 dataname=lab_907_1 batch_size=32 dataset.train_data_list=you pkl dataname file to include the collected pkl files name use_lora=True scheduler_type=0 dataset.num_given_observation=1  max_iters=10000
```

scheduler_type=0 indicates we use 100 DDPM training steps.

### Fully Finetuning on CALVIN

At first, you should follow the [instruction-calvin](https://github.com/mees/calvin) to install CALVIN environment.

we train the network with 4GPUs.

```
python scripts/train_diffusion_sim.py --config-name config_diffusion_calvin batch_size=32 dataset.traj_length=11 num_pred_action=10 task_name=calvin_exp dataset.num_given_observation=2 dataset=fix_camera use_close_loop_eval=True close_loop_eval.test_episodes_num=32 dataset.use_baseframe_action=True taskname=task_ABC_D dataname=calvin_mc close_loop_eval.eval_iters=10000 close_loop_eval.test_episodes_num=250 scheduler_type=0 wrap_grmg_data=2 +pretrained_path=dit_policy_checkpoint.pth +use_adjust_scheduler=true lr=0.0001 epoch=15 +min_lr_scale=0.01 scheduler.warmup_epochs=1
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



In our experiments, we use the Bridge_orig from tfds in google cloud, in which the image has been resized (480\*512->224\*224) and caused **image distortion**.

#### Real Franka Demonstration

Please refer to the [project page](https://RoboDita.github.io/).

### Acknowledgement

The dataloader code of OXE is based on [OpenVLA](https://github.com/openvla/openvla), The dataloader code of CALVIN is based on [GR-MG](https://github.com/bytedance/GR-MG), The architecture is based on transformers.

### Citation

If you find our code or models useful in your work, please cite [our paper](https://arxiv.org/abs/2410.15959):



```

