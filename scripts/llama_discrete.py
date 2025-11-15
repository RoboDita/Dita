import numpy as np
import torch
import torch.nn as nn
from action_tokenizers.action_tokenizer import DiscreteActionTokenizer
from vision_tokenizers.vit_tokenizer import RT1ViTImageTokenizer
from transformers import LlamaConfig, LlamaForCausalLM

class llama_discrete(nn.Module):
    """A transformer based actor network in PyTorch."""

    def __init__(
        self,
        output_tensor_spec,
        train_step_counter=0,
        vocab_size=768,
        token_embedding_size=768,
        intermediate_size=2048,
        num_layers=8,
        dropout_rate=0.1,
        time_sequence_length=1,
        crop_size=236,
        input_size=None,
        action_order=None,
        use_token_learner=True,
        return_attention_scores=False,
        include_prev_timesteps_actions=False,
        freeze_backbone=True,
        use_qformer=False,
        qformer_depth=4,
        use_wrist_img=False,
        use_depth_img=False,
        use_multi_linear=False,
    ):
        super(llama_discrete, self).__init__()

        # Placeholder for attention scores and other attributes
        self.output_tensor_spec = output_tensor_spec
        self.train_step_counter = train_step_counter
        self.actions = None
        self.returns = None
        self.vocab_size = vocab_size
        self.token_embedding_size = token_embedding_size
        self.time_sequence_length = time_sequence_length
        self.input_size = input_size
        self.use_token_learner = use_token_learner
        self.return_attention_scores = return_attention_scores
        self.include_prev_timesteps_actions = include_prev_timesteps_actions
        self.use_wrist_img = use_wrist_img
        self.use_depth_img = use_depth_img

        # Define network components
        self.image_tokenizer = RT1ViTImageTokenizer(
            embedding_output_dim=self.token_embedding_size,
            dropout_rate=dropout_rate,
            freeze_backbone=freeze_backbone,
            use_qformer=use_qformer,
            qformer_depth=qformer_depth,
            use_wrist_img=use_wrist_img,
            use_depth_img=use_depth_img,
            input_size=self.input_size,
        )

        # token type: [SoA], vocab_size * 3 (translation / rotation / gripper), [terminate]/[non-terminate]

        self.action_tokenizer = DiscreteActionTokenizer(
            self.output_tensor_spec, vocab_size=[self.vocab_size, self.vocab_size, self.vocab_size, 2], action_order=action_order
        )
        # import ipdb;ipdb.set_trace()
        # if self.include_prev_timesteps_actions:
        #     self.action_token_emb = nn.Linear(self.vocab_size, self.token_embedding_size)

        self.tokens_per_action = self.action_tokenizer.tokens_per_action
        self.tokens_per_context_image = self.image_tokenizer.tokens_per_context_image

        self.single_time_step_num_tokens = self.tokens_per_action + self.tokens_per_context_image

        self.total_vocab_size = 1 + self.vocab_size * 3 + 2  # [SoA], vocab_size * 3 (translation / rotation / gripper), [terminate]/[non-terminate]

        self.transformer = LlamaForCausalLM(
            LlamaConfig(
                vocab_size=self.total_vocab_size,
                # vocab_size=9,
                hidden_size=self.token_embedding_size,
                intermediate_size=self.token_embedding_size * 4 if intermediate_size is None else intermediate_size,
                num_hidden_layers=num_layers,
                num_attention_heads=self.token_embedding_size // 64,
                attention_dropout=0.1,
            ),
            # use_multi_linear=use_multi_linear
        )

    @property
    def attention_scores(self):
        """Return attention score. This is for debugging/visualization purpose."""
        return self.atten_scores

    def forward(self, obs: dict, act: dict, obs_tokens=None, act_tokens=None, num_given_observation=2):
        image = obs["image"] # b, t, c, h, w
        b = image.shape[0]
        t = image.shape[1]
        t = self.time_sequence_length
        
        image = image[:, :num_given_observation]
        wrist_image = obs["wrist_image"] if "wrist_image" in obs else None
        depth_image = obs["depth_image"] if "depth_image" in obs else None

        # import ipdb;ipdb.set_trace()
        if obs_tokens is None:
            # image = image.permute(0, 1, 4, 2, 3).contiguous()  # (b,t,c,h,w)
            context = obs["natural_language_embedding"]
            context_image_tokens = self.image_tokenizer(image, context[:, :num_given_observation], wrist_image, depth_image)  # [B x T x L x C]
        else:
            context_image_tokens = obs_tokens

        if act_tokens is None:
            action_tokens = self.action_tokenizer.tokenize(act)
        else:
            action_tokens = act_tokens

        assert (action_tokens < self.total_vocab_size).all()

        if self.include_prev_timesteps_actions:
            action_tokens = self.transformer.model.embed_tokens(action_tokens)
            full_tokens = torch.cat([context_image_tokens, action_tokens], axis=-2)
            full_tokens = full_tokens.reshape(b, num_given_observation * self.single_time_step_num_tokens, -1)
        else:
            # If we're not including the previous actions, then we can zero out
            # the action tokens. We do it here to ensure tokens are consistently
            # zero regardless of the input actions passed to the function.
            action_tokens = torch.zeros((b, t, self.tokens_per_action - 1, self.token_embedding_size)).to(context_image_tokens.device)
            soi_tokens = self.transformer.model.embed_tokens(torch.zeros((b, t, 1)).long().to(context_image_tokens.device))

            # Assemble the input tokens into a single sequence.
            image_tokens = context_image_tokens.reshape(b, num_given_observation * self.tokens_per_context_image, -1)
            image_tokens_num = num_given_observation * self.tokens_per_context_image
            all_action_tokens = torch.cat([soi_tokens, action_tokens], axis=-2)
            all_action_tokens = all_action_tokens.reshape(b, t * self.tokens_per_action, -1)
            full_tokens = torch.cat([image_tokens, all_action_tokens], axis=1)

        context = context[:, 0]
        B, L, D = context.shape
        if D < self.token_embedding_size:
            zero_pad = torch.zeros(B, L, self.token_embedding_size - D).to(context.device)
            context = torch.cat([context, zero_pad], dim=2)
        elif D > self.token_embedding_size:
            raise ValueError(f"Language Embedding dim {D} > token Embedding Dim {self.token_embedding_size}")
        full_tokens = torch.cat([context, full_tokens], dim=1)

        output_logits = self.transformer(inputs_embeds=full_tokens)[0]
        if output_logits.dim() == 3: # (batch, tokens_num, dim)
            output_logits = output_logits[:, context.shape[1] + image_tokens_num:, :]
        elif output_logits.dim() == 4: # (heads_num, batch, tokens_num, dim)
            output_logits = output_logits[:, :, context.shape[1] + image_tokens_num:, :]

        return output_logits

    # TODO
    @torch.no_grad()
    def inference(self, obs: dict, obs_tokens=None, num_given_observation=2, exec_steps = None):
        # for AutoRegressive Inference
        image = obs["image"]
        wrist_image = obs["wrist_image"] if "wrist_image" in obs else None
        b = image.shape[0]
        t = image.shape[1]
        num_given_observation = min(t,num_given_observation)
        image = image[:, -num_given_observation:]
        t = self.time_sequence_length

        if obs_tokens is None:
            # image = image.permute(0, 1, 4, 2, 3).contiguous()  # (b,t,c,h,w)
            context = obs["natural_language_embedding"]
            context_image_tokens = self.image_tokenizer(image, context[:, :num_given_observation], wrist_image)  # [B x T x L x C]
        else:
            context_image_tokens = obs_tokens

        action_tokens = torch.zeros((b, t, self.tokens_per_action - 1, self.token_embedding_size)).to(context_image_tokens.device)
        soi_tokens = self.transformer.model.embed_tokens(torch.zeros((b, t, 1)).long().to(context_image_tokens.device))
        # B, T, L, C
        # Assemble the input tokens into a single sequence.
        image_tokens = context_image_tokens.reshape(b, num_given_observation * self.tokens_per_context_image, -1)
        image_tokens_num = num_given_observation * self.tokens_per_context_image
        all_action_tokens = torch.cat([soi_tokens, action_tokens], axis=-2)
        all_action_tokens = all_action_tokens.reshape(b, t * self.tokens_per_action, -1)
        full_tokens = torch.cat([image_tokens, all_action_tokens], axis=1)

        # B, T * L, C
        # full_tokens = full_tokens.reshape(b, t * self.single_time_step_num_tokens, -1)
        context = context[:, 0]
        B, L, D = context.shape
        if D < self.token_embedding_size:
            zero_pad = torch.zeros(B, L, self.token_embedding_size - D).to(context.device)
            context = torch.cat([context, zero_pad], dim=2)
        elif D > self.token_embedding_size:
            raise ValueError(f"Language Embedding dim {D} > token Embedding Dim {self.token_embedding_size}")
        full_tokens = torch.cat([context, full_tokens], dim=1)

        output_logits = self.transformer(inputs_embeds=full_tokens)[0]
        output_logits = output_logits[:, context.shape[1] + image_tokens_num:, :]

        output_logits = output_logits.reshape(b, t, self.tokens_per_action, -1)
        if exec_steps is None:
            output_logits = output_logits[:, num_given_observation - 1 , :-1, :]
        else:
            output_logits = output_logits[:, num_given_observation - 1 : num_given_observation - 1 + exec_steps, :-1, :]

        return output_logits


if __name__ == "__main__":

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    action_spec = {
        "world_vector": {
            "tensor": torch.empty((3,), dtype=torch.float32).to(DEVICE),
            "minimum": torch.tensor([-2.0], dtype=torch.float32).to(DEVICE),
            "maximum": torch.tensor([2.0], dtype=torch.float32).to(DEVICE),
        },
        "rotation_delta": {
            "tensor": torch.empty((4,), dtype=torch.float32).to(DEVICE),
            "minimum": torch.tensor([-np.pi / 2.0], dtype=torch.float32).to(DEVICE),
            "maximum": torch.tensor([np.pi / 2.0], dtype=torch.float32).to(DEVICE),
        },
        "gripper_closedness_action": {
            "tensor": torch.empty((1,), dtype=torch.float32).to(DEVICE),
            "minimum": torch.tensor([-1.0], dtype=torch.float32).to(DEVICE),
            "maximum": torch.tensor([1.0], dtype=torch.float32).to(DEVICE),
        },
        "terminate_episode": {
            "tensor": torch.empty((3,), dtype=torch.int32).to(DEVICE),
            "minimum": torch.tensor([0], dtype=torch.int32).to(DEVICE),
            "maximum": torch.tensor([1], dtype=torch.int32).to(DEVICE),
        },
        # 'base_displacement_vertical_rotation': {'tensor': torch.empty((1,), dtype=torch.float32).to(DEVICE),
        #                 'minimum': torch.tensor([-np.pi], dtype=torch.float32).to(DEVICE),
        #                 'maximum': torch.tensor([np.pi], dtype=torch.float32).to(DEVICE)},
        # 'base_displacement_vector': {'tensor': torch.empty((2,), dtype=torch.float32).to(DEVICE),
        #                 'minimum': torch.tensor([-1.0], dtype=torch.float32).to(DEVICE),
        #                 'maximum': torch.tensor([1.0], dtype=torch.float32).to(DEVICE)}
    }

    model = llama_discrete(
        output_tensor_spec=action_spec,
        vocab_size=768,
        time_sequence_length=15,
        num_layers=12,
    )
    model = model.to(DEVICE)

    obs = {}
    obs["image"] = torch.randn((2, 15, 224, 224, 3), device=DEVICE)
    obs["natural_language_embedding"] = torch.randn((2, 16, 768), device=DEVICE)
    act = {}
    act_tokens = torch.randn((2, 14, 8), device=DEVICE)

    output = model(obs, act, act_tokens=act_tokens)
    # print(pred)
    # output = model(
    #     obs = obs,
    #     act = act,
    #     act_tokens = act_tokens
    # )
    # print(output.shape)
