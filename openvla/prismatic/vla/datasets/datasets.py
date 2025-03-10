"""
datasets.py

Lightweight PyTorch Dataset Definition for wrapping RLDS TFDS Pipeline; just defines transform from RLDS default
format to OpenVLA, IterableDataset shim.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple, Type
import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, IterableDataset
from transformers import PreTrainedTokenizerBase

from prismatic.models.backbones.llm.prompting import PromptBuilder
from prismatic.models.backbones.vision import ImageTransform
from prismatic.util.data_utils import tree_map
from prismatic.vla.action_tokenizer import ActionTokenizer, ActionSingleTokenizer
from prismatic.vla.datasets.rlds import make_interleaved_dataset, make_single_dataset,apply_frame_transforms
from prismatic.vla.datasets.rlds.oxe import OXE_NAMED_MIXTURES, get_oxe_dataset_kwargs_and_weights
from prismatic.vla.datasets.rlds.utils.data_utils import NormalizationType

# HuggingFace Default / LLaMa-2 IGNORE_INDEX (for labels)
IGNORE_INDEX = -100


@dataclass
class RLDSBatchTransform:
    action_tokenizer: ActionTokenizer
    base_tokenizer: PreTrainedTokenizerBase
    image_transform: ImageTransform
    prompt_builder_fn: Type[PromptBuilder]
    predict_stop_token: bool = True

    def __call__(self, rlds_batch: Dict[str, Any]) -> Dict[str, Any]:
        """Converts a RLDS batch to the format expected by the OpenVLA collator/models."""
        dataset_name, action = rlds_batch["dataset_name"], rlds_batch["action"][0]
        img = Image.fromarray(rlds_batch["observation"]["image_primary"][0])
        lang = rlds_batch["task"]["language_instruction"]

        # Construct Chat-based Prompt =>> Input is default query + language instruction, output are the action tokens
        prompt_builder = self.prompt_builder_fn("openvla")
        conversation = [
            {"from": "human", "value": f"What action should the robot take to {lang}?"},
            {"from": "gpt", "value": self.action_tokenizer(action)},
        ]
        for turn in conversation:
            prompt_builder.add_turn(turn["from"], turn["value"])

        # Tokenize (w/ `base_tokenizer`)
        input_ids = self.base_tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).input_ids
        labels = list(input_ids)

        # Tensorize =>> Run Image Transform to get `pixel_values` =>> Return
        #   =>> IMPORTANT :: IF WE'RE USING HF LLM.forward(..., labels=labels), SHIFTING HAPPENS _INSIDE_ MODEL!
        input_ids, labels = torch.tensor(input_ids), torch.tensor(labels)
        pixel_values = self.image_transform(img)
        
        if 'DEBUG' in os.environ:
            import ipdb;ipdb.set_trace()

        # [CRITICAL] We do not want to take the loss for anything but the predicted action tokens!
        labels[: -(len(action) + 1)] = IGNORE_INDEX
        if not self.predict_stop_token:
            labels[-1] = IGNORE_INDEX

        return dict(pixel_values=pixel_values, input_ids=input_ids, labels=labels, dataset_name=dataset_name)


@dataclass
class RLDSBatchTransform1:
    action_tokenizer: ActionTokenizer
    base_tokenizer: PreTrainedTokenizerBase
    image_transform: ImageTransform
    prompt_builder_fn: Type[PromptBuilder]
    predict_stop_token: bool = True
    window_size: int = 1
    two_inps: int = 1
    pad_inp: int = 1

    def __call__(self, rlds_batch: Dict[str, Any]) -> Dict[str, Any]:
        """Converts a RLDS batch to the format expected by the OpenVLA collator/models."""
        
        dataset_name, action = rlds_batch["dataset_name"], rlds_batch["action"][:, :]
        # lang = rlds_batch["task"]["language_instruction"]

        # # Construct Chat-based Prompt =>> Input is default query + language instruction, output are the action tokens
        # prompt_builder = self.prompt_builder_fn("openvla")
        # conversation = [
        #     {"from": "human", "value": f"What action should the robot take to {lang}?"},
        #     {"from": "gpt", "value": self.action_tokenizer(action)},
        # ]
        # for turn in conversation:
        #     prompt_builder.add_turn(turn["from"], turn["value"])
        # import os
        # if 'DEBUG' in os.environ:
        #     import ipdb;ipdb.set_trace()
        if 'input_ids' in rlds_batch:

            input_ids = torch.tensor(rlds_batch['input_ids'])[:, 0]
            labels = torch.tensor(rlds_batch['labels'])[:, 0]
            mask_ = torch.any(labels!= self.base_tokenizer.pad_token_id, dim=0)
            t = mask_.argwhere()[-1].item()
            attention_mask = torch.tensor(rlds_batch['attention_mask'])[:, 0]
            input_ids = input_ids[:, :t+1]
            labels = labels[:, :t+1]
            attention_mask = attention_mask[:, :t+1]
        else:
            prompt = [item[0].decode() if type(item) == list else item.decode() for item in rlds_batch["prompt"].tolist()]

            # import ipdb;ipdb.set_trace()
            # Tokenize (w/ `base_tokenizer`)
            if len(action.shape) == 4:
                action = np.reshape(action, (action.shape[0], action.shape[1]*action.shape[2], action.shape[3]))
            
            input_ids = self.base_tokenizer(prompt, add_special_tokens=True).input_ids
            labels = [torch.tensor(item) for item in input_ids]
            input_ids = [torch.tensor(item) for item in input_ids]
            new_labels = []
            for item in labels:
                # [CRITICAL] We do not want to take the loss for anything but the predicted action tokens!
                
               
                if isinstance(self.action_tokenizer, ActionSingleTokenizer):
                    item[: -(action.shape[1] + 1)] = IGNORE_INDEX
                else:
                    item[: -(action.shape[1]  * (action.shape[2] + 1))] = IGNORE_INDEX  # TODO check the sequence length

                if not self.predict_stop_token:
                    item[:, -1] = IGNORE_INDEX
                new_labels.append(item)
            labels = new_labels
            # import ipdb;ipdb.set_trace()
            pad_token_id = self.action_tokenizer.tokenizer.pad_token_id
            model_max_length = self.action_tokenizer.tokenizer.model_max_length
            from torch.nn.utils.rnn import pad_sequence
            input_ids = pad_sequence(input_ids, batch_first=True, padding_value=pad_token_id)
            labels = pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)


            # import ipdb;ipdb.set_trace()

            # For now, we only support Tokenizers with `padding_side = "right"` during training
            #   => Handle padding via RNN Utils => `pad_sequence`
            # assert self.padding_side == "right", f"Invalid Tokenizer `{self.padding_side = }`"
            # import ipdb;ipdb.set_trace()
            # Truncate (if necessary)
            input_ids, labels = input_ids[:, : model_max_length], labels[:, : model_max_length]

            # Get `attention_mask` by checking for `pad_token_id`
            attention_mask = input_ids.ne(pad_token_id)

        # output = dict(
        #     pixel_values=pixel_values,
        #     input_ids=input_ids,
        #     attention_mask=attention_mask,
        #     labels=labels,
        # )
        # if dataset_names is not None:
        #     output["dataset_names"] = dataset_names
        # import ipdb;ipdb.set_trace()
        if self.pad_inp:
            mask = torch.logical_and((labels != IGNORE_INDEX), labels !=2)
            input_ids[mask] = pad_token_id


        pixel_values = torch.tensor(rlds_batch["observation"]["image_primary"])
        
        if len(pixel_values.shape) == 6:
            pixel_values = pixel_values[:, 0]
        
        # 1 
        pixel_values_dino = (pixel_values / 255. - torch.tensor([0.4850, 0.4560, 0.4060])) / torch.tensor([0.2290, 0.2240, 0.2250])
        pixel_values_dino = pixel_values_dino.permute(0, 1, 4, 2, 3)

        # 2
        pixel_values_siglip = (pixel_values / 255. - torch.tensor([0.5000, 0.5000, 0.5000])) / torch.tensor([0.5000, 0.5000, 0.5000])
        pixel_values_siglip = pixel_values_siglip.permute(0, 1, 4, 2, 3)
        # Normalize(mean=tensor([0.4850, 0.4560, 0.4060]), std=tensor([0.2290, 0.2240, 0.2250]))
        # Normalize(mean=tensor([0.5000, 0.5000, 0.5000]), std=tensor([0.5000, 0.5000, 0.5000]))

        if 'image_secondary' in rlds_batch["observation"]:
            pixel_values_second = torch.tensor(rlds_batch["observation"]["image_secondary"])
            pixel_values_dino_second = (pixel_values_second / 255. - torch.tensor([0.4850, 0.4560, 0.4060])) / torch.tensor([0.2290, 0.2240, 0.2250])
            pixel_values_dino_second = pixel_values_dino_second.permute(0, 1, 4, 2, 3)
            pixel_values_dino = torch.cat([pixel_values_dino, pixel_values_dino_second], dim=1)

            pixel_values_siglip_second = (pixel_values_second / 255. - torch.tensor([0.4850, 0.4560, 0.4060])) / torch.tensor([0.2290, 0.2240, 0.2250])
            pixel_values_siglip_second = pixel_values_siglip_second.permute(0, 1, 4, 2, 3)
            pixel_values_siglip = torch.cat([pixel_values_siglip, pixel_values_siglip_second], dim=1)
                
        # torch.save(pixel_values, 'pixel_values.pth', )
        # torch.save(pixel_values_dino, 'pixel_values_dino.pth', )
        # torch.save(pixel_values_siglip, 'pixel_values_siglip.pth', )
        # torch.save(input_ids, 'input_ids.pth')
        # print(prompt)
        # torch.save(prompt, 'prompt.pth')
       # output = dict(
        #     pixel_values=pixel_values,
        #     input_ids=input_ids,
        #     attention_mask=attention_mask,
        #     labels=labels,
        # )
        loss_weight = torch.logical_not(torch.tensor(rlds_batch['action_past_goal']))
        if self.two_inps == 1:
            if 'DIFFHEAD' in os.environ:
                return dict(pixel_values={'dino': pixel_values_dino, 'siglip': pixel_values_siglip}, 
                    input_ids=input_ids, 
                    labels=labels, 
                    action = action,
                    attention_mask=attention_mask,
                    dataset_names=dataset_name)
            return dict(pixel_values={'dino': pixel_values_dino, 'siglip': pixel_values_siglip}, 
                        input_ids=input_ids, 
                        labels=labels, 
                        action = action,
                        loss_weight = loss_weight,
                        attention_mask=attention_mask,
                        dataset_names=dataset_name)
        elif self.two_inps == 2:
            pixel_values_siglip = pixel_values_siglip.flatten(0, 1)
            return dict(pixel_values=pixel_values_siglip, 
                    input_ids=input_ids, 
                    labels=labels, 
                    attention_mask=attention_mask,
                    dataset_names=dataset_name)
        else:
            # if type(pixel_values) == dict:
            #         pixel_values = pixel_values['dino']
            #         if len(pixel_values.shape) == 5:
            pixel_values_dino = pixel_values_dino.flatten(0, 1)
            return dict(pixel_values=pixel_values_dino, 
                    input_ids=input_ids, 
                    labels=labels, 
                    attention_mask=attention_mask,
                    dataset_names=dataset_name)


class RLDSDataset(IterableDataset):
    def __init__(
        self,
        data_root_dir: Path,
        data_mix: str,
        batch_transform: RLDSBatchTransform,
        resize_resolution: Tuple[int, int],
        shuffle_buffer_size: int = 256_000,
        train: bool = True,
        image_aug: bool = False,
        window_size= 1, 
        future_action_window_size= 0,
        batch_size = None,
        batchfy=False,
        prompt_builder_fn=None,
        action_tokenizer=None,
        base_tokenizer=None,
        gen_prompt_transform=0,
        num_image_token=32,
        subsample_length=None,
        subsample_type=None,
        use_common_statisticcs=False,
        center_crop=False,
        max_action=None, # 2.5
        max_proprio=None,
        clamp_value=False,
        load_proprio=False,
        prompt_type= 0,
        with_loc_task=False,
        action_proprio_normalization_type=NormalizationType.BOUNDS_Q99,
        load_camera_views=("primary",),
    ) -> None:
        """Lightweight wrapper around RLDS TFDS Pipeline for use with PyTorch/OpenVLA Data Loaders."""
        self.data_root_dir, self.data_mix, self.batch_transform = data_root_dir, data_mix, batch_transform
        self.buffer_size, self.batchfy = shuffle_buffer_size, batchfy

        # Configure RLDS Dataset(s)
        if self.data_mix in OXE_NAMED_MIXTURES:
            mixture_spec = OXE_NAMED_MIXTURES[self.data_mix]
        else:
            # Assume that passed "mixture" name is actually a single dataset -- create single-dataset "mix"
            mixture_spec = [(self.data_mix, 1.0)]

        # fmt: off
        per_dataset_kwargs, weights = get_oxe_dataset_kwargs_and_weights(
            self.data_root_dir,
            mixture_spec,
            # ('primary', 'secondary')
            # load_camera_views=("primary", "secondary",),
            load_camera_views=load_camera_views,
            load_depth=False,
            load_proprio=load_proprio,
            load_language=True,
            action_proprio_normalization_type=action_proprio_normalization_type,
        )
        rlds_config = dict(
            traj_transform_kwargs=dict(
                window_size=window_size,                                      # If we wanted to feed / predict more than one step
                future_action_window_size=future_action_window_size,                        # For action chunking
                skip_unlabeled=True,                                # Skip trajectories without language labels
                # goal_relabeling_strategy="uniform",                 # Goals are currently unused
                subsample_length=subsample_length,
                subsample_type=subsample_type,
                max_action=max_action,
                max_proprio=max_proprio,
                with_loc_task=with_loc_task,

            ),
            frame_transform_kwargs=dict(
                resize_size=resize_resolution,
                num_parallel_calls=4,                          # For CPU-intensive ops (decoding, resizing, etc.)
                center_crop=center_crop,
            ),
            dataset_kwargs_list=per_dataset_kwargs,
            shuffle_buffer_size=shuffle_buffer_size,
            sample_weights=weights,
            balance_weights=True,
            traj_transform_threads=len(mixture_spec),
            traj_read_threads=len(mixture_spec),
            train=train,
            gen_prompt_transform = gen_prompt_transform,
            gen_prompt_transform_kwargs=dict(action_tokenizer = action_tokenizer,
            prompt_builder_fn = prompt_builder_fn, 
            num_image_token=num_image_token,
            num_images = len(load_camera_views),
            prompt_type=prompt_type,
            base_tokenizer=base_tokenizer),
            use_common_statisticcs=use_common_statisticcs,
            clamp_value=clamp_value,
        )

        self.train = train
        
        # If applicable, enable image augmentations
        if image_aug:
            # rlds_config["frame_transform_kwargs"].update({"image_augment_kwargs" : dict(
            #     random_resized_crop=dict(scale=[0.9, 0.9], ratio=[1.0, 1.0]),
            #     random_brightness=[0.2],
            #     random_contrast=[0.8, 1.2],
            #     random_saturation=[0.8, 1.2],
            #     random_hue=[0.05],
            #     augment_order=[
            #         "random_resized_crop",
            #         "random_brightness",
            #         "random_contrast",
            #         "random_saturation",
            #         "random_hue",
            #     ],
            # )})
            rlds_config["frame_transform_kwargs"].update({"image_augment_kwargs" : dict(
                random_resized_crop=dict(scale=[0.8, 1.0], ratio=[0.9, 1.1]),
                random_brightness=[0.1],
                random_contrast=[0.9, 1.1],
                random_saturation=[0.9, 1.1],
                random_hue=[0.05],
                augment_order=[
                    "random_resized_crop",
                    "random_brightness",
                    "random_contrast",
                    "random_saturation",
                    "random_hue",
                ],
            )})            
            # octo
    #             primary_augment_kwargs = dict(
    #     random_resized_crop=dict(scale=[0.8, 1.0], ratio=[0.9, 1.1]),
    #     random_brightness=[0.1],
    #     random_contrast=[0.9, 1.1],
    #     random_saturation=[0.9, 1.1],
    #     random_hue=[0.05],
    #     augment_order=[
    #         "random_resized_crop",
    #         "random_brightness",
    #         "random_contrast",
    #         "random_saturation",
    #         "random_hue",
    #     ],
    # )
        # fmt: on

        # rlds_config['traj_transform_kwargs']['window_size'] =2
        # rlds_config['traj_transform_kwargs']['future_action_window_size'] = 16
        # aa, bb, cc = self.make_dataset(rlds_config)
        # aaa = next(aa.as_numpy_iterator())
        # import ipdb;ipdb.set_trace()
        # Initialize RLDS Dataset
        if batch_size is not None:
            rlds_config['batch_size'] = batch_size
        self.batch_size = batch_size
        self.dataset, self.dataset_length, self.dataset_statistics = self.make_dataset(rlds_config)

    def make_dataset(self, rlds_config):
        print('rlds_config', rlds_config['clamp_value'])
        return make_interleaved_dataset(**rlds_config)

    def __iter__(self) -> Dict[str, Any]:
        if self.batchfy:
            for i in range(0, self.dataset_length, self.buffer_size):
                print('next batch', i)
                for rlds_batch in self.batch_iter(self.buffer_size, i).as_numpy_iterator():
                    yield self.batch_transform(rlds_batch)
        else:
            while True:
                try:
                    for rlds_batch in self.dataset.as_numpy_iterator():
                        yield self.batch_transform(rlds_batch)
                except Exception as e:
                    # exit()
                    import traceback
                    traceback.print_exc()
                    print(e)

    def __len__(self) -> int:
        return self.dataset_length

    # === Explicitly Unused ===
    def __getitem__(self, idx: int) -> None:
        raise NotImplementedError("IterableDataset does not implement map-style __getitem__; see __iter__ instead!")


class EpisodicRLDSDataset(RLDSDataset):
    """Returns full episodes as list of steps instead of individual transitions (useful for visualizations)."""

    def make_dataset(self, rlds_config):
        per_dataset_kwargs = rlds_config["dataset_kwargs_list"]
        assert len(per_dataset_kwargs) == 1, "Only support single-dataset `mixes` for episodic datasets."

        return make_single_dataset(
            per_dataset_kwargs[0],
            train=rlds_config["train"],
            traj_transform_kwargs=rlds_config["traj_transform_kwargs"],
            frame_transform_kwargs=rlds_config["frame_transform_kwargs"],
        )

    def __iter__(self) -> Dict[str, Any]:
        for rlds_batch in self.dataset.as_numpy_iterator():
            out = [
                self.batch_transform(tree_map(lambda x: x[i], rlds_batch))  # noqa: B023
                for i in range(rlds_batch["action"].shape[0])
            ]
            yield out



class DummyDataset1(Dataset):
    def __init__(
        self,
        data_root_dir: Path,
        data_mix: str,
        batch_transform: RLDSBatchTransform,
        resize_resolution: Tuple[int, int],
        shuffle_buffer_size: int = 256_000,
        train: bool = True,
        image_aug: bool = False,
        window_size= 1, 
        future_action_window_size= 0,
        batch_size = None,
        batchfy=False,
        prompt_builder_fn=None,
        action_tokenizer=None,
        base_tokenizer=None,
        gen_prompt_transform=0,
        num_image_token=32,
        subsample_length=None,
        subsample_type=None,
        use_common_statisticcs=False,
        center_crop=False,
        max_action=1.,
        max_proprio=None,
        clamp_value=True,
        load_proprio=False,
        action_proprio_normalization_type=NormalizationType.BOUNDS_Q99,
    ) -> None:
        self.action_tokenizer = action_tokenizer
        self.base_tokenizer = base_tokenizer
        self.image_transform = None
        self.prompt_builder_fn = prompt_builder_fn

        # Note =>> We expect the dataset to store statistics for action de-normalization. Specifically, we store the
        # per-dimension 1st and 99th action quantile. The values below correspond to "no normalization" for simplicity.
        self.dataset_statistics = {
            "dummy_dataset": {
                "action": {"q01": np.zeros((7,), dtype=np.float32), "q99": np.ones((7,), dtype=np.float32)}
            }
        }
        # self.aaa['dataset_names'] = ['test']*64
        # self.aaa['image_flags'] = torch.ones([64],)
        # self.aaa.pop('image_flags')
        self.aaa = torch.load('inputs_debug.pt')

    def __len__(self):
        # TODO =>> Replace with number of elements in your dataset!
        return 10000

    def __getitem__(self, idx):
        return self.aaa


class DummyDataset(Dataset):
    def __init__(
        self,
        action_tokenizer: ActionTokenizer,
        base_tokenizer: PreTrainedTokenizerBase,
        image_transform: ImageTransform,
        prompt_builder_fn: Type[PromptBuilder],
    ) -> None:
        self.action_tokenizer = action_tokenizer
        self.base_tokenizer = base_tokenizer
        self.image_transform = image_transform
        self.prompt_builder_fn = prompt_builder_fn

        # Note =>> We expect the dataset to store statistics for action de-normalization. Specifically, we store the
        # per-dimension 1st and 99th action quantile. The values below correspond to "no normalization" for simplicity.
        self.dataset_statistics = {
            "dummy_dataset": {
                "action": {"q01": np.zeros((7,), dtype=np.float32), "q99": np.ones((7,), dtype=np.float32)}
            }
        }

    def __len__(self):
        # TODO =>> Replace with number of elements in your dataset!
        return 10000

    def __getitem__(self, idx):
        # TODO =>> Load image, action and instruction from disk -- we use dummy values
        image = Image.fromarray(np.asarray(np.random.rand(224, 224, 3) * 255.0, dtype=np.uint8))
        action = np.asarray(np.random.rand(7), dtype=np.float32)
        instruction = "do something spectacular"

        # Add instruction to VLA prompt
        prompt_builder = self.prompt_builder_fn("openvla")
        conversation = [
            {"from": "human", "value": f"What action should the robot take to {instruction}?"},
            {"from": "gpt", "value": self.action_tokenizer(action)},
        ]
        for turn in conversation:
            prompt_builder.add_turn(turn["from"], turn["value"])

        # Tokenize (w/ `base_tokenizer`)
        input_ids = self.base_tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).input_ids
        labels = list(input_ids)

        # Tensorize =>> Run Image Transform to get `pixel_values` =>> Return
        #   =>> IMPORTANT :: IF WE'RE USING HF .forward(..., labels=labels), SHIFTING HAPPENS _INSIDE_ MODEL!
        input_ids, labels = torch.tensor(input_ids), torch.tensor(labels)
        pixel_values = self.image_transform(image)

        # [CRITICAL] We do not want to take the loss for anything but the predicted action tokens!
        labels[: -(len(action) + 1)] = IGNORE_INDEX

        return dict(pixel_values=pixel_values, input_ids=input_ids, labels=labels)
