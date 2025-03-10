"""
materialize.py

Factory class for initializing Open-X RLDS-backed datasets, given specified data mixture parameters; provides and
exports individual functions for clear control flow.
"""

from pathlib import Path
from typing import Tuple, Type

from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

from prismatic.models.backbones.llm.prompting import PromptBuilder
from prismatic.models.backbones.vision import ImageTransform
from prismatic.util.data_utils import PaddedCollatorForActionPrediction
from prismatic.vla.action_tokenizer import ActionSingleTokenizer, ActionTokenizer
from prismatic.vla.datasets.datasets import DummyDataset1, RLDSBatchTransform_lab
from openvla_warp.datasets_finetune import CalvinDataset_warp



def get_vla_dataset_and_collator(
    data_root_dir: Path,
    data_mix: str,
    image_transform: ImageTransform,
    tokenizer: PreTrainedTokenizerBase,
    prompt_builder_fn: Type[PromptBuilder],
    default_image_resolution: Tuple[int, int, int],
    padding_side: str = "right",
    predict_stop_token: bool = True,
    shuffle_buffer_size: int = 100_000,
    train: bool = True,
    episodic: bool = False,
    image_aug: bool = False,
    window_size= 1,
    future_action_window_size= 0,
    batch_size=None,
    gen_prompt_transform=0,
    num_image_token=32,
    dumpy_data=0,
    bins=256,
    num_of_used_token_ids=7,
    two_inps = 1,
    load_proprio=False,
    prompt_type=0,
    center_crop=False,
    with_loc_task = False,
    pad_inp = 0,
    use_diffusion_loss=0,
    load_camera_views=("primary",),
    dataset_name=None,
    seq_len=1,
    act_len=1, 
    forward_n_max=25, 
    mode=None,
    subfolder=None,
    use_data_augmentation=False,
    task_num=10000,
    use_play=False,
    use_labeled=True,
    wrap_grmg_data=1
) -> Tuple[Dataset, ActionTokenizer, PaddedCollatorForActionPrediction]:
    """Initialize RLDS Dataset (wraps TFDS), ActionTokenizer, and initialize transform/collation functions."""
    from prismatic.vla.datasets import EpisodicRLDSDataset, RLDSBatchTransform, RLDSBatchTransform1, RLDSDataset
    

    if batch_size is not None:
        
        action_tokenizer = ActionTokenizer(tokenizer)
        batch_transform = RLDSBatchTransform1(
            action_tokenizer, tokenizer, image_transform, prompt_builder_fn, predict_stop_token=predict_stop_token, window_size=window_size,
            two_inps=two_inps, pad_inp=pad_inp
        )
    else:
        action_tokenizer = ActionTokenizer(tokenizer)
        batch_transform = RLDSBatchTransform(
            action_tokenizer, tokenizer, image_transform, prompt_builder_fn, predict_stop_token=predict_stop_token
        )

    collator = PaddedCollatorForActionPrediction(
        tokenizer.model_max_length, tokenizer.pad_token_id, padding_side=padding_side
    )
    if dumpy_data:
        dataset = DummyDataset1(data_root_dir,
            data_mix,
            batch_transform,
            resize_resolution=default_image_resolution[1:],
            shuffle_buffer_size=shuffle_buffer_size,
            train=train,
            image_aug=image_aug,
            window_size= window_size,
            future_action_window_size= future_action_window_size,
            batch_size=batch_size,
            prompt_builder_fn=prompt_builder_fn,
            action_tokenizer=action_tokenizer,
            base_tokenizer= tokenizer,
            gen_prompt_transform=gen_prompt_transform,
            num_image_token = num_image_token,)
        return dataset, action_tokenizer, collator

    cls = RLDSDataset if not episodic else EpisodicRLDSDataset
    dataset = cls(
        data_root_dir,
        data_mix,
        batch_transform,
        resize_resolution=default_image_resolution[1:],
        shuffle_buffer_size=shuffle_buffer_size,
        train=train,
        image_aug=image_aug,
        window_size= window_size,
        future_action_window_size= future_action_window_size,
        batch_size=batch_size,
        prompt_builder_fn=prompt_builder_fn,
        action_tokenizer=action_tokenizer,
        base_tokenizer= tokenizer,
        gen_prompt_transform=gen_prompt_transform,
        num_image_token = num_image_token,
        load_camera_views=load_camera_views,
        load_proprio=load_proprio,
        prompt_type = prompt_type,
        center_crop=center_crop,
        with_loc_task = with_loc_task,
    )


    return dataset, action_tokenizer, collator
