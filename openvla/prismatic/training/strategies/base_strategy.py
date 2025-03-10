"""
base_strategy.py

Abstract class definition of a (distributed) training strategy, with full annotations of class methods, utility
functions, and initialization logic.

Training Strategies (DDP, FSDP-Grad, FSDP-Full) tend to have a lot of repeated components; this class does a lot of
heavy lifting.
"""

from abc import ABC, abstractmethod
import os
from pathlib import Path
from typing import Callable, Optional

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, DistributedSampler, IterableDataset
from tqdm import tqdm
from transformers.modeling_outputs import CausalLMOutputWithPast

from prismatic.models.vlms import PrismaticVLM
from prismatic.overwatch import initialize_overwatch
from prismatic.training.metrics import Metrics, VLAMetrics
from prismatic.util import check_bloat16_supported
from prismatic.util.batching_utils import SplitModalitySampler
from prismatic.util.data_utils import IGNORE_INDEX, PaddedCollatorForActionPrediction, PaddedCollatorForLanguageModeling
from prismatic.vla.action_tokenizer import ActionTokenizer

# Initialize Overwatch =>> Wraps `logging.Logger`
overwatch = initialize_overwatch(__name__)


# === Abstract Base Class for an arbitrary Training Strategy ===
class TrainingStrategy(ABC):
    def __init__(
        self,
        vlm: PrismaticVLM,
        device_id: int,
        stage: str,
        epochs: int,
        max_steps: Optional[int],
        global_batch_size: int,
        per_device_batch_size: int,
        learning_rate: float,
        weight_decay: float,
        max_grad_norm: float,
        lr_scheduler_type: str,
        warmup_ratio: float,
        enable_gradient_checkpointing: bool = True,
        enable_mixed_precision_training: bool = True,
        reduce_in_full_precision: bool = False,
        mixed_precision_dtype: torch.dtype = torch.bfloat16,
        worker_init_fn: Optional[Callable[[int], None]] = None,
        **_: str,
    ) -> None:
        self.vlm, self.device_id, self.stage = vlm, device_id, stage

        # Get relevant VLM instance parameters before they get (potentially) wrapped
        self.all_module_keys, self.trainable_module_keys = self.vlm.all_module_keys, self.vlm.trainable_module_keys
        
        self.llm_transformer_layer_cls = self.vlm.llm_backbone.transformer_layer_cls

        # Optimization Parameters
        self.epochs, self.max_steps = epochs, max_steps
        self.global_batch_size, self.per_device_batch_size = global_batch_size, per_device_batch_size

        self.learning_rate, self.weight_decay, self.max_grad_norm = learning_rate, weight_decay, max_grad_norm
        self.lr_scheduler_type, self.warmup_ratio = lr_scheduler_type, warmup_ratio

        # Generic Strategy Parameters
        self.enable_gradient_checkpointing = enable_gradient_checkpointing
        self.enable_mixed_precision_training = enable_mixed_precision_training
        self.reduce_in_full_precision = reduce_in_full_precision
        self.mixed_precision_dtype = mixed_precision_dtype

        # DataLoader Parameters
        self.worker_init_fn = worker_init_fn

        # Optimizers & Scheduler (initialized in `run_setup`)
        self.optimizer, self.lr_scheduler = None, None

        # Lightweight Validation
        assert (
            self.global_batch_size % self.per_device_batch_size == 0
        ), "Per-device batch size must evenly divide global batch size!"
        self.grad_accumulation_steps = self.global_batch_size // self.per_device_batch_size // overwatch.world_size()
        self.grad_accumulation_steps = 1
        if self.enable_mixed_precision_training:
            assert self.mixed_precision_dtype == torch.bfloat16, "Only BF16 mixed precision training is supported!"
            assert check_bloat16_supported(), "BFloat16 is not supported on this hardware; unset `mixed_precision`"

    @abstractmethod
    def save_checkpoint(
        self,
        run_dir: Path,
        global_step: int,
        epoch: int,
        train_loss: Optional[float] = None,
        only_trainable: bool = True,
    ) -> None: ...

    @abstractmethod
    def run_setup(self, run_dir: Path, n_train_examples: int) -> None: ...

    @abstractmethod
    def clip_grad_norm(self) -> None: ...

    def run_training(
        self,
        dataset: Dataset,
        collator: PaddedCollatorForLanguageModeling,
        metrics: Metrics,
        stage: str = "finetune",
        batch_construction_strategy: str = "split-modality",
        seed: int = 7,
    ) -> None:
        """Run the training loop for the given `dataset` and `collator`; log losses, results to `metrics`"""
        if "finetune" in stage and batch_construction_strategy == "split-modality":
            # Instantiate the split-modality sampler; if you want to extend with other batch construction schemes,
            #   (e.g., grouping by length) =>> can easily add them here!
            modality_lengths = dataset.get_modality_lengths()
            sampler = SplitModalitySampler(
                dataset,
                modality_lengths,
                global_batch_size=self.global_batch_size,
                num_replicas=overwatch.world_size(),
                rank=overwatch.rank(),
                seed=seed,
                drop_last=False,
            )

        else:
            sampler = DistributedSampler(
                dataset,
                num_replicas=overwatch.world_size(),
                rank=overwatch.rank(),
                shuffle=True,
                seed=seed,
                drop_last=False,
            )

        # Create a DataLoader with the initialized sampler, per-device-bsz, and collator
        dataloader = DataLoader(
            dataset,
            batch_size=self.per_device_batch_size,
            sampler=sampler,
            collate_fn=collator,
            num_workers=2,
            worker_init_fn=self.worker_init_fn,
        )

        # Max Steps vs. Epochs Computation
        steps_per_epoch = len(dataloader) // self.grad_accumulation_steps
        if self.max_steps is not None and steps_per_epoch < self.max_steps:
            # Just set `epochs` to some large number --> we'll short-circuit based on steps anyway
            self.epochs = 100

        # === Train ===
        status = metrics.get_status()
        with tqdm(
            total=(
                (self.epochs * (len(dataloader) // self.grad_accumulation_steps))
                if self.max_steps is None
                else self.max_steps
            ),
            desc=status,
            leave=False,
            disable=not overwatch.is_rank_zero(),
        ) as progress:
            for epoch in range(self.epochs):
                self.vlm.train()
                sampler.set_epoch(epoch)

                # Zero-Gradients (just in case)
                self.optimizer.zero_grad()

                # Note that we'll unpack batch (and let AMP/FSDP do its thing) in the VLM.forward() call
                #   => Basically, if we're using mixed precision (or not), autocast()/FSDP will move to device!
                for train_idx, batch in enumerate(dataloader):
                    # if 'DEBUG' in os.environ:
                    #     import ipdb;ipdb.set_trace()
                    # [Contract] self.vlm.forward() must automatically compute `loss` and return!
                    with torch.autocast(
                        "cuda",
                        dtype=self.mixed_precision_dtype,
                        enabled=self.enable_mixed_precision_training,
                    ):
                        output: CausalLMOutputWithPast = self.vlm(
                            input_ids=batch["input_ids"],
                            attention_mask=batch["attention_mask"],
                            pixel_values=batch["pixel_values"],
                            labels=batch["labels"],
                            multimodal_indices=batch["multimodal_indices"],
                        )
                        loss = output.loss

                    # Commit Loss (Prior to Gradient Accumulation Normalization)
                    metrics.commit(loss=loss)

                    # Normalize Loss to account for Gradient Accumulation --> Backward!
                    # [IMPORTANT] Technically speaking, doing gradient accumulation in this way is "incorrect"; this is
                    #             because in general, each batch has a *different number of masked out tokens* (because
                    #             we're instruct-tuning). Taking the mean over two unbalanced means != the right thing!
                    #
                    #             HOWEVER -- at least at the 7B scale, the "naive" approach is just as performant as
                    #             the "correct" implementation, without adding extra complexity.
                    #
                    # That being said =>> at the 13B scale, *no matter what we tried, ANY gradient accumulation is just
                    #   really bad for downstream performance. Initial investigation shows that BF16 accumulation
                    #   just really tanks in precision... and don't have a good/clean way to fix this. Would love for
                    #   someone to PR and fix this (and I'd greatly appreciate it!!!)
                    normalized_loss = loss / self.grad_accumulation_steps
                    normalized_loss.backward()

                    # Step =>> Only if Done w/ Gradient Accumulation
                    if (train_idx + 1) % self.grad_accumulation_steps == 0:
                        metrics.commit(update_step_time=True)

                        # Clip Gradients --> this is custom, per-strategy because of DDP vs. FSDP locality-assumptions
                        self.clip_grad_norm()

                        # Optimizer & LR Scheduler Step
                        self.optimizer.step()
                        self.lr_scheduler.step()
                        self.optimizer.zero_grad()

                        # Push Metrics
                        metrics.commit(global_step=metrics.global_step + 1, lr=self.lr_scheduler.get_last_lr()[0])
                        status = metrics.push()

                        # Check for Termination & Save Final Checkpoint (in case `max_steps` is not None)
                        if self.max_steps is not None and metrics.global_step >= self.max_steps:
                            self.save_checkpoint(metrics.run_dir, metrics.global_step, epoch, loss.item())
                            dist.barrier()

                            return

                        # Update Progress Bar
                        progress.update()
                        progress.set_description(status)

            # Save checkpoint at end each epoch (if `self.max_steps` is None)
            if self.max_steps is None:
                self.save_checkpoint(metrics.run_dir, metrics.global_step, epoch, loss.item())
                dist.barrier()

    # === VLA Training ===

    def run_vla_training(
        self,
        vla_dataset: IterableDataset,
        collator: PaddedCollatorForActionPrediction,
        action_tokenizer: ActionTokenizer,
        metrics: VLAMetrics,
        save_interval: int = 2500,
        save_full_model: bool = True,
        num_of_images=1,
    ) -> None:
        """Run the VLA training loop for the given `dataset` and `collator`; log losses, action metrics to `metrics`."""
        assert isinstance(vla_dataset, IterableDataset), "VLA training expects an IterableDataset!"
        assert self.grad_accumulation_steps == 1, "VLA training does not support gradient accumulation!"

        # Create a DataLoader =>> Set `num_workers` to 0; RLDS loader handles parallelism!
        if vla_dataset.batch_size is not None:
            dataloader = vla_dataset
        else:
            dataloader = DataLoader(
                vla_dataset,
                batch_size=self.per_device_batch_size,
                sampler=None,
                collate_fn=collator,
                num_workers=0,
                worker_init_fn=self.worker_init_fn,
            )

        # === Train ===
        status = metrics.get_status()
        with tqdm(
            total=(self.epochs * len(dataloader)) if self.max_steps is None else self.max_steps,
            desc=status,
            leave=False,
            disable=not overwatch.is_rank_zero(),
        ) as progress:
            self.vlm.train()

            # Zero Gradients (just in case)
            self.optimizer.zero_grad()

            # [Contract] DataLoader wraps RLDS Loader (`.as_numpy_iterator() =>> implicit `.repeat()`)
            #   => This means looping over the DataLoader is basically "infinite" (so no outer loop over epochs).
            #      Slightly breaks default PyTorch semantics, which is why we adaptively compute `epoch` below.
            while True:
                for batch in dataloader:
                    # Note that we'll unpack batch (and let AMP/FSDP do its thing) in the VLM.forward() call
                    #   => Basically, if we're using mixed precision (or not), autocast()/FSDP will move to device!

                    # if 'DEBUG' in os.environ:
                    #     import ipdb;ipdb.set_trace()
                    with torch.autocast(
                        "cuda", dtype=self.mixed_precision_dtype, enabled=self.enable_mixed_precision_training
                    ):
                        # [Contract] self.vlm.forward() must automatically compute `loss` and return!

                        

                        output: CausalLMOutputWithPast = self.vlm(
                            input_ids=batch["input_ids"],
                            attention_mask=batch["attention_mask"],
                            pixel_values=batch["pixel_values"],
                            labels=batch["labels"],

                        )
                        loss = output.loss
                        if 'DIFFHEAD' in os.environ:
                            seq_len = batch["labels"].shape[1]
                            output_logits = output['hidden_states'][-1][:, -seq_len:-1]
                            trajectory = torch.tensor(batch['action']).to(output_logits.device)
                            #import ipdb;ipdb.set_trace()
                            noise = torch.randn(trajectory.shape, device=output_logits.device)
                            bsz = trajectory.shape[0]
                            # model.vision_backbone.num_patches
                            
                            # import ipdb;ipdb.set_trace()

                            action_gt = batch["labels"][:, 1:].to(output_logits.device)
                            mask = torch.logical_and((action_gt != IGNORE_INDEX), action_gt !=2)
                            timesteps = torch.randint(0, self.vlm.noise_scheduler.config.num_train_timesteps, (bsz,), device=trajectory.device).long().to(output_logits.device)
                            # Add noise to the clean images according to the noise magnitude at each timestep
                            # (this is the forward diffusion process)
                            noisy_action_tokens = self.vlm.noise_scheduler.add_noise(trajectory, noise, timesteps)
                            timestep_tokens = self.vlm.time_emb(timesteps)
                            import torch.nn.functional as F
                            b, t, c = noisy_action_tokens.shape
                            t_act = mask[0].sum()
                            
                            # self.vlm.token_embedding_size
                            noisy_action_tokens1 = F.pad(noisy_action_tokens, (0, 4096 -noisy_action_tokens.shape[-1]), mode='constant', value=0).to(output_logits.device)
                            
                            noise_pred = self.vlm.diffuse_action_head(noisy_action_tokens1.flatten(0, 1), output_logits[mask], timestep_tokens.repeat(1, t_act, 1).flatten(0, 1))


                            target = noise
                            b, num, dim = noise_pred.shape
                            logits = noise_pred

                            diff_loss = F.mse_loss(logits, target, reduction='none').mean()
                            loss = loss + diff_loss
         
                        
                    # Commit Loss =>> Backward!
                    metrics.commit(loss=loss)
                    loss.backward()

                    
                    # if overwatch.is_rank_zero() and int(os.environ['RANK'])==0:
                    #     import ipdb;ipdb.set_trace()
                    #     print('dino', self.vlm.vision_backbone.dino_featurizer.blocks[4].mlp.fc1.weight.data.mean())
                    #     print('dino', self.vlm.vision_backbone.dino_featurizer.blocks[16].norm2.weight.data.mean())
                    #     print('dino', self.vlm.vision_backbone.dino_featurizer.blocks[8].norm2.weight.data.mean())
                    #     print('dino', self.vlm.vision_backbone.dino_featurizer.blocks[4].norm2.weight.data.mean())

                    
                    if 'DEBUG' in os.environ:
                        import ipdb;ipdb.set_trace()
                    # === Compute Action Token Accuracy & L1 Loss ===

                    # To compute action token accuracy, we need to identify the locations of the action tokens
                    # in both `output.logits` and `batch["labels"]`. We know that when "right" padding, we
                    # insert `self.vlm.vision_backbone.num_patches` at index 1.
                    #
                    # Computing `action_prediction_accuracy` is then pretty straightforward:
                    #   1) Extract "aligned" predictions & labels
                    #   2) Compute boolean "mask" where "labels > 2" (where 2 is ID for `EOS_TOKEN`)
                    #           => If masking out EOS, then it's just "labels != -100 (IGNORE_INDEX)
                    #   3) Compute masked accuracy as `(preds == logits) & mask` --> sum/divide by # unmasked!
                    # import ipdb;ipdb.set_trace()
                    try:
                        # import ipdb;ipdb.set_trace()
                        if num_of_images == 2:
                            # import ipdb;ipdb.set_trace()
                            action_preds = output.logits[:, self.vlm.vision_backbone.num_patches * 2 : -1].argmax(dim=2)
                        else:
                            action_preds = output.logits[:, self.vlm.vision_backbone.num_patches : -1].argmax(dim=2)
                        action_gt = batch["labels"][:, 1:].to(action_preds.device)
                        mask = torch.logical_and((action_gt != IGNORE_INDEX), action_gt !=2)

                        # # import ipdb;ipdb.set_trace()
                        # if mask[0].sum().item() > 7 * num_of_images:
                        #     # two windows
                        #     # TODO change 29892, this is sep
                        #     b_idx, sep_idx = torch.where((action_gt == 29892))
                        #     num_of_chunk = (b_idx == 0).sum().item()
                            
                        #     # 1, - 8
                        #     # 2, -0
                        #     mask_ = torch.arange(mask.size(1)).to(mask.device).unsqueeze(0) <= (sep_idx[0::num_of_chunk].unsqueeze(1) + (num_of_images-2) * 8)

                        #     mask_1 = torch.arange(mask.size(1)).to(mask.device).unsqueeze(0) >= (sep_idx[1::num_of_chunk].unsqueeze(1) + (num_of_images-2) * 8)
                        #     mask  = torch.logical_and(~mask_, mask)
                        #     mask = torch.logical_and(~mask_1, mask)
                        


                        #import ipdb;ipdb.set_trace()

                        long_pred_indices = mask.sum(-1) > 7
                        mask_long_pred = mask[long_pred_indices]
                        mask_short = mask[~long_pred_indices]

                        
                        # if mask[0].sum().item() > 7 * num_of_images:
                        # two windows
                        # TODO change 29892, this is sep
                        b_idx, sep_idx = torch.where((action_gt[long_pred_indices] == 29892))
                        num_of_chunk = (b_idx == 0).sum().item()
                        # import ipdb;ipdb.set_trace()
                        # 1, - 8
                        # 2, -0
                        if num_of_chunk > 0:
                            mask_ = torch.arange(mask_long_pred.size(1)).to(mask.device).unsqueeze(0) <= (sep_idx[0::num_of_chunk].unsqueeze(1) + (num_of_images-2) * 8)
                            mask_long_pred  = torch.logical_and(~mask_, mask_long_pred)
                            if num_of_chunk > 1:
                                mask_1 = torch.arange(mask_long_pred.size(1)).to(mask.device).unsqueeze(0) >= (sep_idx[1::num_of_chunk].unsqueeze(1) + (num_of_images-2) * 8)
                                mask_long_pred = torch.logical_and(~mask_1, mask_long_pred)
                        
                        mask[long_pred_indices] = mask_long_pred
                        mask[~long_pred_indices] = mask_short
                        
                        
                        # mask = action_gt > action_tokenizer.action_token_begin_idx
                        assert (mask.sum(-1) == 7).all(), (mask.sum(-1), )



                        # Compute Accuracy
                        correct_preds = (action_preds == action_gt) & mask
                        action_accuracy = correct_preds.sum().float() / mask.sum().float()

                        # Compute L1 Loss on Predicted (Continuous) Actions
                        continuous_actions_pred = torch.tensor(
                            action_tokenizer.decode_token_ids_to_actions(action_preds[mask].cpu().numpy())
                        )
                        continuous_actions_gt = torch.tensor(
                            action_tokenizer.decode_token_ids_to_actions(action_gt[mask].cpu().numpy())
                        )
                        action_l1_loss = torch.nn.functional.l1_loss(continuous_actions_pred, continuous_actions_gt)

                        # Commit Metrics
                        metrics.commit(action_accuracy=action_accuracy, action_accuracy_a = action_accuracy, l1_loss=action_l1_loss, update_step_time=True)

                        # Compute metrics per dataset --> only on rank_zero since we don't log them on other workers anyways
                        if overwatch.is_rank_zero() and int(os.environ['RANK'])==0:
                            if len(batch["dataset_names"].shape) == 2:
                                datasets = set(batch["dataset_names"][:, 0])
                            else:
                                datasets = set(batch["dataset_names"])
                            if len(datasets) > 1:
                                for ds in datasets:
                                    ds_mask = torch.tensor([elem == ds for elem in batch["dataset_names"]])
                                    action_accuracy_ds = correct_preds[ds_mask].sum().float() / mask[ds_mask].sum().float()
                                    continuous_actions_pred_ds = torch.tensor(
                                        action_tokenizer.decode_token_ids_to_actions(
                                            action_preds[ds_mask][mask[ds_mask]].cpu().numpy()
                                        )
                                    )
                                    continuous_actions_gt_ds = torch.tensor(
                                        action_tokenizer.decode_token_ids_to_actions(
                                            action_gt[ds_mask][mask[ds_mask]].cpu().numpy()
                                        )
                                    )
                                    action_l1_loss_ds = torch.nn.functional.l1_loss(
                                        continuous_actions_pred_ds, continuous_actions_gt_ds
                                    )
                                    metrics.commit_for_dataset(
                                        dataset_name=ds.decode(), action_accuracy=action_accuracy_ds, l1_loss=action_l1_loss_ds
                                    )
                    except:
                        import traceback
                        traceback.print_exc()
                        pass
                    # === Gradient Step ===

                    # Clip Gradients --> this is custom, per-strategy because of DDP vs. FSDP locality assumptions
                    self.clip_grad_norm()

                    # Optimizer & LR Scheduler Step
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()

                    # Compute epoch value using number of completed gradient steps
                    epoch = (metrics.global_step + 1) // (len(vla_dataset) // self.global_batch_size)

                    try:
                        # Push Metrics
                        metrics.commit(global_step=metrics.global_step + 1, epoch=epoch, lr=self.lr_scheduler.get_last_lr()[0])
                        status = metrics.push()
                    except:
                        import traceback
                        traceback.print_exc()
                        pass

                    # Check for Save Interval or Max Steps & Save Checkpoint
                    if (terminate := (self.max_steps is not None and metrics.global_step >= self.max_steps)) or (
                        (metrics.global_step % save_interval) == 0
                    ):
                        self.save_checkpoint(
                           metrics.run_dir, metrics.global_step, epoch, loss.item(), only_trainable=not save_full_model
                        )
                        dist.barrier()

                        if terminate:
                            return

                    # Update Progress Bar
                    progress.update()
                    progress.set_description(status)


    def run_vla_validation(
        self,
        vla_dataset: IterableDataset,
        collator: PaddedCollatorForActionPrediction,
        action_tokenizer: ActionTokenizer,
        metrics: VLAMetrics,
        save_interval: int = 2500,
        save_full_model: bool = True,
        num_of_images=1,
    ) -> None:
        """Run the VLA training loop for the given `dataset` and `collator`; log losses, action metrics to `metrics`."""
        assert isinstance(vla_dataset, IterableDataset), "VLA training expects an IterableDataset!"
        assert self.grad_accumulation_steps == 1, "VLA training does not support gradient accumulation!"

        # Create a DataLoader =>> Set `num_workers` to 0; RLDS loader handles parallelism!
        if vla_dataset.batch_size is not None:
            dataloader = vla_dataset
        else:
            dataloader = DataLoader(
                vla_dataset,
                batch_size=self.per_device_batch_size,
                sampler=None,
                collate_fn=collator,
                num_workers=0,
                worker_init_fn=self.worker_init_fn,
            )

        # === Train ===
        status = metrics.get_status()
        with tqdm(
            total=(self.epochs * len(dataloader)) if self.max_steps is None else self.max_steps,
            desc=status,
            leave=False,
            disable=not overwatch.is_rank_zero(),
        ) as progress:
            self.vlm.eval()

            # Zero Gradients (just in case)
            self.optimizer.zero_grad()

            # [Contract] DataLoader wraps RLDS Loader (`.as_numpy_iterator() =>> implicit `.repeat()`)
            #   => This means looping over the DataLoader is basically "infinite" (so no outer loop over epochs).
            #      Slightly breaks default PyTorch semantics, which is why we adaptively compute `epoch` below.
            for batch in dataloader:
                # Note that we'll unpack batch (and let AMP/FSDP do its thing) in the VLM.forward() call
                #   => Basically, if we're using mixed precision (or not), autocast()/FSDP will move to device!

                # if 'DEBUG' in os.environ:
                #     import ipdb;ipdb.set_trace()
                with torch.autocast(
                    "cuda", dtype=self.mixed_precision_dtype, enabled=self.enable_mixed_precision_training
                ):
                    # [Contract] self.vlm.forward() must automatically compute `loss` and return!
                    # import ipdb;ipdb.set_trace()
                    output: CausalLMOutputWithPast = self.vlm(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        pixel_values=batch["pixel_values"],
                        labels=batch["labels"],

                    )
                    # import ipdb;ipdb.set_trace()
                    # with torch.autocast("cuda", dtype=self.mixed_precision_dtype, enabled=self.enable_mixed_precision_training):
                    #     # fmt: off
                    #     generated_ids = super(PrismaticVLM, self).generate(
                    #         input_ids=batch["input_ids"],                            # Shape: [1, seq]
                    #         pixel_values=batch["pixel_values"],                      # Shape: [1, 3, res, res] or Dict[str, ...]
                    #         max_new_tokens=7, # 7
                            
                    #     )
                    #     import ipdb;ipdb.set_trace()
                        # fmt: on

                    # pass

                    # torch.save(batch["labels"], 'labels1.pt') # 0.2313,
                    

                
                # if overwatch.is_rank_zero() and int(os.environ['RANK'])==0:
                #     import ipdb;ipdb.set_trace()
                #     print('dino', self.vlm.vision_backbone.dino_featurizer.blocks[4].mlp.fc1.weight.data.mean())
                #     print('dino', self.vlm.vision_backbone.dino_featurizer.blocks[16].norm2.weight.data.mean())
                #     print('dino', self.vlm.vision_backbone.dino_featurizer.blocks[8].norm2.weight.data.mean())
                #     print('dino', self.vlm.vision_backbone.dino_featurizer.blocks[4].norm2.weight.data.mean())

                
                if 'DEBUG' in os.environ:
                    import ipdb;ipdb.set_trace()
                # === Compute Action Token Accuracy & L1 Loss ===

                # To compute action token accuracy, we need to identify the locations of the action tokens
                # in both `output.logits` and `batch["labels"]`. We know that when "right" padding, we
                # insert `self.vlm.vision_backbone.num_patches` at index 1.
                #
                # Computing `action_prediction_accuracy` is then pretty straightforward:
                #   1) Extract "aligned" predictions & labels
                #   2) Compute boolean "mask" where "labels > 2" (where 2 is ID for `EOS_TOKEN`)
                #           => If masking out EOS, then it's just "labels != -100 (IGNORE_INDEX)
                #   3) Compute masked accuracy as `(preds == logits) & mask` --> sum/divide by # unmasked!
                # import ipdb;ipdb.set_trace()

                try:
                    if num_of_images == 2:
                        # import ipdb;ipdb.set_trace()
                        action_preds = output.logits[:, self.vlm.vision_backbone.num_patches * 2 : -1].argmax(dim=2)
                    else:
                        action_preds = output.logits[:, self.vlm.vision_backbone.num_patches : -1].argmax(dim=2)
                    action_gt = batch["labels"][:, 1:].to(action_preds.device)
                    mask = torch.logical_and((action_gt != IGNORE_INDEX), action_gt !=2)

                    import ipdb;ipdb.set_trace()
                    if mask[0].sum().item() > 7 * num_of_images:
                        # two windows
                        # TODO change 29892, this is sep
                        b_idx, sep_idx = torch.where((action_gt == 29892))
                        num_of_chunk = (b_idx == 0).sum().item()
                        
                        # 1, - 8
                        # 2, -0
                        mask_ = torch.arange(mask.size(1)).to(mask.device).unsqueeze(0) <= (sep_idx[0::num_of_chunk].unsqueeze(1) + (num_of_images-2) * 8)

                        mask_1 = torch.arange(mask.size(1)).to(mask.device).unsqueeze(0) >= (sep_idx[1::num_of_chunk].unsqueeze(1) + (num_of_images-2) * 8)
                        mask  = torch.logical_and(~mask_, mask)
                        mask = torch.logical_and(~mask_1, mask)
                    


                    # import ipdb;ipdb.set_trace()

                    # long_pred_indices = mask.sum(-1) > 7
                    # mask_long_pred = mask[long_pred_indices]
                    # mask_short = mask[~long_pred_indices]

                    
                    # # if mask[0].sum().item() > 7 * num_of_images:
                    # # two windows
                    # # TODO change 29892, this is sep
                    # b_idx, sep_idx = torch.where((action_gt[long_pred_indices] == 29892))
                    # num_of_chunk = (b_idx == 0).sum().item()
                    
                    # # 1, - 8
                    # # 2, -0
                    # if num_of_chunk > 0:
                    #     mask_ = torch.arange(mask_long_pred.size(1)).to(mask.device).unsqueeze(0) <= (sep_idx[0::num_of_chunk].unsqueeze(1) + (num_of_images-2) * 8)

                    #     mask_1 = torch.arange(mask_long_pred.size(1)).to(mask.device).unsqueeze(0) >= (sep_idx[1::num_of_chunk].unsqueeze(1) + (num_of_images-2) * 8)
                    #     mask_long_pred  = torch.logical_and(~mask_, mask_long_pred)
                    #     mask_long_pred = torch.logical_and(~mask_1, mask_long_pred)
                    
                    # mask[long_pred_indices] = mask_long_pred
                    # mask[~long_pred_indices] = mask_short
                    
                    
                    # mask = action_gt > action_tokenizer.action_token_begin_idx
                    assert (mask.sum(-1) == 7).all(), (mask.sum(-1), )



                    # Compute Accuracy
                    correct_preds = (action_preds == action_gt) & mask
                    action_accuracy = correct_preds.sum().float() / mask.sum().float()

                    # Compute L1 Loss on Predicted (Continuous) Actions
                    continuous_actions_pred = torch.tensor(
                        action_tokenizer.decode_token_ids_to_actions(action_preds[mask].cpu().numpy())
                    )
                    continuous_actions_gt = torch.tensor(
                        action_tokenizer.decode_token_ids_to_actions(action_gt[mask].cpu().numpy())
                    )
                    action_l1_loss = torch.nn.functional.l1_loss(continuous_actions_pred, continuous_actions_gt)
                    # Commit Metrics
                    metrics.commit(action_accuracy=action_accuracy, action_accuracy_a = action_accuracy, l1_loss=action_l1_loss, update_step_time=True)

                except:
                    import traceback
                    traceback.print_exc()
                    pass
                # === Gradient Step ===

                # Compute epoch value using number of completed gradient steps
                epoch = (metrics.global_step + 1) // (len(vla_dataset) // self.global_batch_size)

                try:
                    # Push Metrics
                    metrics.commit(global_step=metrics.global_step + 1, epoch=epoch, lr=self.lr_scheduler.get_last_lr()[0])
                    status = metrics.push()
                except:
                    import traceback
                    traceback.print_exc()
                    pass

                
                # Update Progress Bar
                progress.update()
                progress.set_description(status)
