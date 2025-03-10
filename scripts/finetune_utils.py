from dataclasses import dataclass
import os
import sys

import torch


current_path = os.getcwd()
sys.path.append(current_path)
sys.path.append(os.path.join(current_path, "scripts/openx_utils/"))
sys.path.append(os.path.join(current_path, "../embodied_foundation/scripts"))
sys.path.append(os.path.join(current_path, "../embodied_foundation/openvla"))
sys.path.append(os.path.join(current_path, "openvla/"))
from torch.utils.data import DataLoader
from openvla.prismatic.vla.datasets.datasets_finetune import LabDataset_warp

def get_training_data(cfg):
    if cfg.dataname == 'lab':
        print('use lab data')
        train_dataset = LabDataset_warp(data_path = "Lab",
            language_embedding_path= "records_banana_embeddings_77token_v0",
            traj_per_episode=cfg.dataset.traj_per_episode,
                traj_length=cfg.dataset.traj_length,
            include_target=1,
            img_preprocess_type=1,
            use_baseframe_action=True,
            split_type='fix_traj',
            data_cam_list='lab.pkl',
            obs_n_frames=2,
            stride=4, euler_delta=cfg.euler_delta,
            selected_list=['left'])
    


    train_sampler = None
    if cfg.dataname in ['lab']:
        @dataclass
        class DataAdapterForOpenx:        
            def __call__(self, rlds_batch ):
                
                loss_weight = torch.logical_not(torch.tensor(rlds_batch['action_past_goal']))
                # print(loss_weight.shape)
                # print(rlds_batch["action"].shape)
                dataset_name, action = rlds_batch["dataset_name"], torch.tensor(rlds_batch["action"])
                state = torch.tensor(rlds_batch['state'])
                lang = rlds_batch["task"]["language_instruction"]
                dataset_name = 'lab'

                pixel_values = torch.tensor(rlds_batch["observation"]["image_primary"])
                # print(pixel_values.shape)
                # Normalize 
                # pixel_values = (pixel_values / 255. - torch.tensor(IMAGENET_DEFAULT_MEAN)) / torch.tensor(IMAGENET_DEFAULT_STD)
                # pixel_values = pixel_values.permute(0, 3, 1, 2)
                del rlds_batch
                return dict(pixel_values=pixel_values, action=action, state=state, dataset_name=dataset_name, language_instruction= lang, loss_weight=loss_weight)
        train_dataset.set_batch_transform(DataAdapterForOpenx())
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=cfg.batch_size,
            sampler=train_sampler,
            drop_last=True,
            num_workers= 1 if 'overfit' in cfg else 8,
            prefetch_factor=4,
            pin_memory=True,
            persistent_workers=True,
        )    

    return train_dataloader, train_sampler

