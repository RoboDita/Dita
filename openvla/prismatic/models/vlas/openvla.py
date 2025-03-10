"""
openvla.py

PyTorch Module defining OpenVLA as a lightweight wrapper around a PrismaticVLM; defines custom logic around
discretizing actions with the ActionTokenizer.
"""

from typing import Dict, List, Optional

import numpy as np
import torch
from PIL import Image
from transformers import LlamaTokenizerFast

from prismatic.models.vlms.prismatic import PrismaticVLM
from prismatic.overwatch import initialize_overwatch
from prismatic.vla.action_tokenizer import ActionTokenizer

# Initialize Overwatch =>> Wraps `logging.Logger`
overwatch = initialize_overwatch(__name__)


class OpenVLA(PrismaticVLM):
    def __init__(
        self,
        *args,
        norm_stats: Dict[str, Dict[str, Dict[str, Dict[str, List[float]]]]],
        action_tokenizer: ActionTokenizer,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.norm_stats = norm_stats
        self.action_tokenizer = action_tokenizer
        self.image_observations = []
        self._supports_cache_class = False

    @torch.inference_mode()
    def predict_action(
        self, image: Image, instruction: str, num_of_obs=2, proprio=None, non_autoregressive=False, unnorm_key: Optional[str] = None, **kwargs: str
    ) -> np.ndarray:
        """
        Core function for VLA inference; maps input image and task instruction to continuous action (de-tokenizes).

        @param image: PIL Image as [height, width, 3]
        @param instruction: Task instruction string
        @param unnorm_key: Optional dataset name for retrieving un-normalizing statistics; if None, checks that model
                           was trained only on a single dataset, and retrieves those statistics.

        @return Unnormalized (continuous) action vector --> end-effector deltas.
        """
        image_transform, tokenizer = self.vision_backbone.image_transform, self.llm_backbone.tokenizer

        # Build VLA Prompt
        prompt_builder = self.get_prompt_builder()
        if proprio is not None:
            
            # conversation = [{"from": "human", "value": f"Current robot proprio is {self.action_tokenizer(proprio)}, 
            #                  What action should the robot take to {lang.numpy()}?"},{"from": "gpt", "value": obtain_action_tokenizer_convertion(act_inp)},]
            prompt_builder.add_turn(role="human", message=f"Current robot proprio is {self.action_tokenizer(proprio)}, What action should the robot take to {instruction.lower()}?")
        else:
            prompt_builder.add_turn(role="human", message=f"What action should the robot take to {instruction.lower()}?")
        prompt_text = prompt_builder.get_prompt()

        
        # Prepare Inputs
        # 
        input_ids = tokenizer(prompt_text, truncation=True, return_tensors="pt").input_ids.to(self.device)
        if isinstance(tokenizer, LlamaTokenizerFast):
            # Note: We need to add this special empty token ('') after the colon (':') token in "ASSISTANT:"
            #       in order for the predictions to match the training configuration and be accurate.
            input_ids = torch.cat(
                (input_ids, torch.unsqueeze(torch.Tensor([29871]).long(), dim=0).to(self.device)), dim=1
            )
        else:
            raise ValueError(f"Unsupported `tokenizer` type = {type(tokenizer)}")

        # historical images
        self.image_observations.append(image)
        if len(self.image_observations) < num_of_obs:
            self.image_observations.append(image)
        # Preprocess Image
        merged_pixel_values = []
        for img in self.image_observations[-num_of_obs:]:
            pixel_values = image_transform(img)
            merged_pixel_values.append(pixel_values)
        
        # for item in merged_pixel_values:
        # import ipdb;ipdb.set_trace()

        if isinstance(pixel_values, torch.Tensor):
            pixel_values = torch.stack(merged_pixel_values)[None, ...].to(self.device)
        elif isinstance(pixel_values, dict):
            pixel_values = {k: torch.stack([item[k] for item in merged_pixel_values])[None, ...].to(self.device) for k, v in pixel_values.items()}
        else:
            raise ValueError(f"Unsupported `pixel_values` type = {type(pixel_values)}")

        # Invoke super().generate --> taps into `GenerationMixin` which (redirects) to `forward()`
        # import ipdb;ipdb.set_trace()
        if non_autoregressive:
            # import ipdb;ipdb.set_trace()
            input_ids = torch.cat(
                (input_ids, torch.unsqueeze(torch.Tensor([32000]*7).long(), dim=0).to(self.device)), dim=1
            )
            pixel_values['dino'] = pixel_values['dino'].to(torch.bfloat16)
            pixel_values['siglip'] = pixel_values['siglip'].to(torch.bfloat16)
            results = self(input_ids, pixel_values=pixel_values,)
            generated_ids = results.logits.argmax(-1)[:, -8:-1]
        #             self,
        #     input_ids: Optional[torch.LongTensor] = None,
        #     attention_mask: Optional[torch.Tensor] = None,
        #     pixel_values: Optional[torch.FloatTensor] = None,
        #     labels: Optional[torch.LongTensor] = None,
        #     inputs_embeds: Optional[torch.FloatTensor] = None,
        #     past_key_values: Optional[List[torch.FloatTensor]] = None,
        #     use_cache: Optional[bool] = None,
        #     output_attentions: Optional[bool] = None,
        #     output_hidden_states: Optional[bool] = None,
        #     return_dict: Optional[bool] = None,
        #     multimodal_indices: Optional[torch.LongTensor] = None,
        # ) -> CausalLMOutputWithPast:
        else:
            autocast_dtype = self.llm_backbone.half_precision_dtype
            with torch.autocast("cuda", dtype=autocast_dtype, enabled=self.enable_mixed_precision_training):
                # fmt: off
                generated_ids = super(PrismaticVLM, self).generate(
                    input_ids=input_ids,                            # Shape: [1, seq]
                    pixel_values=pixel_values,                      # Shape: [1, 3, res, res] or Dict[str, ...]
                    max_new_tokens=(self.get_action_dim(unnorm_key)+1) * num_of_obs - 1, # 7
                    **kwargs
                )
            # fmt: on
        
        # Extract predicted action tokens and translate into (normalized) continuous actions
        predicted_action_token_ids = generated_ids[0, -self.get_action_dim(unnorm_key) :]
        normalized_actions = self.action_tokenizer.decode_token_ids_to_actions(predicted_action_token_ids.cpu().numpy())

        # Un-normalize Actions
        action_norm_stats = self.get_action_stats(unnorm_key)
        mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["q01"], dtype=bool))
        action_high, action_low = np.array(action_norm_stats["q99"]), np.array(action_norm_stats["q01"])
        import os
        if 'ZERO' in os.environ:
            print('zero norm')
            action_low = torch.tensor([-0.009082376956939697, -0.02768026292324066, -0.09064042553305625, -0.088255375623703, -0.07572497427463531, -0.10641985386610031, 0]).to(torch.float32).cpu().numpy()
            action_high = torch.tensor([0.049961209297180176, 0.029934369027614594, 0.06721316277980804, 0.06538952142000198, 0.03357397019863129, 0.17205530777573924, 1]).to(torch.float32).cpu().numpy()
        elif 'FEWSHOT' in os.environ:
            print('fewshot norm')
            action_low = torch.tensor([-0.06314074993133545, -0.1255711580812931, -0.10342398285865784, -0.1623569279909134, -0.21733978390693665, -0.25088581442832947, 0.0]).to(torch.float32).cpu().numpy()
            action_high = torch.tensor([0.07548637896776225, 0.10902893543243408, 0.10883220300078411, 0.2607732117176056, 0.1869153529405594, 0.29620999097824097, 1.0]).to(torch.float32).cpu().numpy()
        elif 'OBJ' in os.environ:
            print('fewshot norm')
            action_low = torch.tensor([-0.05637969195842743, -0.05230307340621948, -0.05837308406829834, -0.6183725643157959, -0.017089657485485077, -0.20603514194488526, 0]).to(torch.float32).cpu().numpy()
            action_high = torch.tensor([0.04498136043548584, 0.052343665957450866, -0.016356410980224608, -0.0002570152282714844, 0.01564564779400828, 0.3672934627532959, 1]).to(torch.float32).cpu().numpy()
        
        elif 'DROID' in os.environ:
            print('use_droid')
            action_low = torch.tensor([-0.03282684743404388, -0.045336535349488255, -0.042811885476112366, -0.10672842025756836, -0.1170416958630085, -0.1501714438199997, 0]).to(torch.float32).cpu().numpy()
            action_high = torch.tensor([0.03991392910480496, 0.04513102851808071, 0.04638494879007338, 0.10861503660678862, 0.1076623786985873, 0.150611503124237, 1.0]).to(torch.float32).cpu().numpy()
        
        else:
            action_low = torch.tensor([-0.07060414552688599, -0.21747050866484638, -0.2911102771759033, -0.18562862336635585, -0.1285559296607971, -0.4114302545785903, 0.0]).to(torch.float32).cpu().numpy()
            action_high = torch.tensor([0.11125240057706841, 0.1061392036080361, 0.12897171080112457, 0.1357136829197407, 0.10151379711925987, 0.4232045072317128, 1.0]).to(torch.float32).cpu().numpy()

        actions = np.where(
            mask,
            0.5 * (normalized_actions + 1) * (action_high - action_low) + action_low,
            normalized_actions,
        )

        return actions

    @staticmethod
    def _check_unnorm_key(norm_stats: Dict, unnorm_key: str) -> str:
        #print("norm_stats",norm_stats)
        if unnorm_key is None:
            assert len(norm_stats) == 1, (
                f"Your model was trained on more than one dataset, please pass a `unnorm_key` from the following "
                f"options to choose the statistics used for un-normalizing actions: {norm_stats.keys()}"
            )
            unnorm_key = next(iter(norm_stats.keys()))

        # Error Handling
        assert (
            unnorm_key in norm_stats
        ), f"The `unnorm_key` you chose is not in the set of available statistics; choose from: {norm_stats.keys()}"

        return unnorm_key

    def get_action_dim(self, unnorm_key: Optional[str] = None) -> int:
        """Dimensionality of the policy's action space."""
        unnorm_key = self._check_unnorm_key(self.norm_stats, unnorm_key)

        return len(self.norm_stats[unnorm_key]["action"]["q01"])

    def get_action_stats(self, unnorm_key: Optional[str] = None) -> Dict:
        """Dimensionality of the policy's action space."""
        unnorm_key = self._check_unnorm_key(self.norm_stats, unnorm_key)

        return self.norm_stats[unnorm_key]["action"]
