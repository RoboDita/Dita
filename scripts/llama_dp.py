import numpy as np
import torch
import torch.nn as nn
from vision_tokenizers.vit_tokenizer import RT1ViTImageTokenizer
from transformers import LlamaConfig, LlamaForCausalLM
import math
import torch.nn.functional as F
import os

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb
class AdaLN(nn.Module):  
    def __init__(self, num_features, num_conds):  
        super(AdaLN, self).__init__()  
        self.num_features = num_features  
        self.gamma = nn.Parameter(torch.ones(num_features))  
        self.beta = nn.Parameter(torch.zeros(num_features))  
 
        self.cond_to_gamma = nn.Linear(num_conds, num_features)  
        self.cond_to_beta = nn.Linear(num_conds, num_features)  
  
    def forward(self, x, z):  
        gamma = self.cond_to_gamma(z) + self.gamma  
        beta = self.cond_to_beta(z) + self.beta  
  
        mean = x.mean(-1, keepdim=True)  
        std = x.std(-1, keepdim=True, unbiased=False)  
        x_normalized = (x - mean) / (std + 1e-5) 
        return gamma * x_normalized + beta  
  
class ResidualBlock(nn.Module):  
    def __init__(self, in_channels, out_channels, num_conds):  
        super(ResidualBlock, self).__init__()  
        self.norm = AdaLN(in_channels, num_conds)  
        self.linear1 = nn.Linear(in_channels, out_channels)  
        self.silu = nn.SiLU()  
        self.linear2 = nn.Linear(out_channels, out_channels)  
  
        self.equal_channels = in_channels == out_channels  
  
    def forward(self, x, z):  
        
        residual = x  
  
        out = self.norm(x, z)  
        out = self.linear1(out)  
        out = self.silu(out)  
        out = self.linear2(out)  
  
        if self.equal_channels:  
            return out + residual  
        else:  
            raise ValueError("Residual connection dimensions do not match.")  
  

class DenoisingMLP(nn.Module):  
    def __init__(self, in_channels, out_channels, num_blocks=3, width=1024, num_conds=None):  
        super(DenoisingMLP, self).__init__()  
        if num_conds is None:  
            raise ValueError("num_conds must be specified for the AdaLN layers.")  
  
        self.blocks = nn.ModuleList([  
            ResidualBlock(in_channels if i == 0 else width, width, num_conds)  
            for i in range(num_blocks)  
        ])  
  
        if out_channels != width:  
            self.final_linear = nn.Linear(width, out_channels)  
        else:  
            self.final_linear = nn.Identity()  
  
    def forward(self, x, z, t):  
        z = t + z
        for block in self.blocks:  
            x = block(x, z)  
        x = self.final_linear(x)  
        return x  
  
class AdaLN(nn.Module):
    def __init__(self, num_features, num_conds):
        super(AdaLN, self).__init__()
        self.num_features = num_features
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))
        self.cond_to_gamma = nn.Linear(num_conds, num_features)
        self.cond_to_beta = nn.Linear(num_conds, num_features)

    def forward(self, x, z):
        gamma = self.cond_to_gamma(z) + self.gamma
        beta = self.cond_to_beta(z) + self.beta
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True, unbiased=False)
        x_normalized = (x - mean) / (std + 1e-5)
        return gamma * x_normalized + beta


def unnormalize(x):
    x = x.clone()
    from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
    for i in range(3):
        x[..., i] = x[..., i] * IMAGENET_DEFAULT_STD[i] + IMAGENET_DEFAULT_MEAN[i]

    return x







class RobotTransformerNet(nn.Module):
    """A transformer based actor network in PyTorch."""

    def __init__(
        self,
        output_tensor_spec,
        train_step_counter=0,
        vocab_size=768,
        trajectory_dim = 11,
        token_embedding_size=768,
        intermediate_size=2048,
        hidden_size=768,
        num_layers=8,
        dropout_rate=0.1,
        time_sequence_length=-1,
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
        dim_align_type=0, # 0 pad
        prediction_type='epsilon',
        scheduler_type=0,
        num_inference_steps=10,
        attn_implementation='eager',
        use_action_head_diff=False,
        vit_forward_version = None,
    ):
        super(RobotTransformerNet, self).__init__()

        # Placeholder for attention scores and other attributes
        self.output_tensor_spec = output_tensor_spec
        self.train_step_counter = train_step_counter
        self.use_action_head_diff = use_action_head_diff
        self.trajectory_dim = trajectory_dim
        print('trajectory_dim:', self.trajectory_dim, self.use_action_head_diff)

        self.actions = None
        self.returns = None
        self.vocab_size = vocab_size
        self.token_embedding_size = token_embedding_size
        assert time_sequence_length != -1
        self.time_sequence_length = time_sequence_length
        self.intermediate_size = intermediate_size
        self.input_size = input_size
        self.num_inference_steps = num_inference_steps

        self.use_token_learner = use_token_learner
        self.return_attention_scores = return_attention_scores
        self.include_prev_timesteps_actions = include_prev_timesteps_actions
        self.use_wrist_img = use_wrist_img
        self.use_depth_img = use_depth_img
        
        self.dim_align_type = dim_align_type
        if self.dim_align_type == 1:
            self.linear_dim_aligner = torch.nn.Linear(in_features=trajectory_dim, out_features=token_embedding_size)

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
            vit_forward_version = None,
        )

        self.tokens_per_context_image = self.image_tokenizer.tokens_per_context_image


        
        if self.use_action_head_diff in [2]:
            # mlp

            self.diffuse_action_head = DenoisingMLP(in_channels=768, out_channels=self.trajectory_dim, num_blocks=3, width=768, num_conds=768)  
        
        elif self.use_action_head_diff == 5:
            # Octo style
            self.diffuse_action_head = DenoisingMLP(in_channels=768, out_channels=self.trajectory_dim*self.time_sequence_length, num_blocks=3, width=768, num_conds=768)  
            # self.diffuse_action_head = DenoisingTransformer(in_channels=768, out_channels=self.trajectory_dim, num_blocks=3, width=768, num_conds=768)  
            
        self.single_time_step_num_tokens = 1 + self.tokens_per_context_image

        self.total_vocab_size = 1 + self.vocab_size * 3 + 2  # [SoA], vocab_size * 3 (translation / rotation / gripper), [terminate]/[non-terminate]

        self.attn_implementation = attn_implementation
        # import ipdb;ipdb.set_trace()
        self.transformer = LlamaForCausalLM(
            LlamaConfig(
                vocab_size=(trajectory_dim if self.dim_align_type != 2 else self.total_vocab_size) if not self.use_action_head_diff else self.token_embedding_size,
                # vocab_size=9,
                hidden_size=self.token_embedding_size,
                intermediate_size=self.intermediate_size,
                num_hidden_layers=num_layers,
                num_attention_heads=self.token_embedding_size // 64,
                attention_dropout=0.1,
                # attn_implementation= attn_implementation,
                _flash_attn_2_enabled= True if attn_implementation != 'eager' else False
                # dtype=torch.float16
            )
        )
        # diffuse_action_head.model.embed_tokens.weight


        if attn_implementation != 'eager':
            self.transformer.to(torch.float16)
        if self.dim_align_type != 2:
            self.action_tokenizer = None
            self.time_emb = SinusoidalPosEmb(token_embedding_size)
            self.transformer.model.embed_tokens = None
            
        from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
        from diffusers import DDIMScheduler
        self.scheduler_type = scheduler_type

        print('scheduler_type', scheduler_type)
        if scheduler_type in [1]:
            self.noise_scheduler = DDPMScheduler(prediction_type = prediction_type,)

            
            # FOR INFERENCE
            self.noise_scheduler_eval = DDIMScheduler(prediction_type = prediction_type,)
            # self.noise_scheduler_eval = self.noise_scheduler
            print('noise_scheduler_eval', 'ddim')     
        else:
            self.noise_scheduler = DDPMScheduler(
                beta_schedule = 'squaredcos_cap_v2',
                num_train_timesteps = 100,
                prediction_type = prediction_type,
            )

            self.noise_scheduler_eval = DDIMScheduler(
                beta_schedule = 'squaredcos_cap_v2',
                num_train_timesteps = 100,
                prediction_type = prediction_type,
            )


    @property
    def attention_scores(self):
        """Return attention score. This is for debugging/visualization purpose."""
        return self.atten_scores

    def forward(self, obs: dict, act: dict, obs_tokens=None, act_tokens=None, noisy_action_tokens=None, 
                timesteps=None, num_pred_action=1, ret_feats=False, aug_img_tokens=False, reg_token_nums=0, reused_context_image_tokens=None):
        image = obs["image"]

        b = image.shape[0]
        # t = image.shape[1]
        t = self.time_sequence_length

        wrist_image = obs["wrist_image"][:, :t-num_pred_action+1] if "wrist_image" in obs else None
        depth_image = obs["depth_image"][:, :t-num_pred_action+1] if "depth_image" in obs else None

        if obs_tokens is None:
            # import ipdb;ipdb.set_trace()
            if reused_context_image_tokens is None:
                if 'natural_language_embedding' in obs:
                    context = obs["natural_language_embedding"]

                    context_image_tokens = self.image_tokenizer(image[:, :t-num_pred_action+1], context[:, :t-num_pred_action+1], wrist_image, depth_image)  # [B x T x L x C]
                else:
                    inputs = self.clip_tokenizer(text=obs['language_instruction'], return_tensors="pt", max_length=77, padding="max_length")
                    for key in inputs:
                        inputs[key] = inputs[key].to(image.device)

                    context = self.model.text_model(**inputs)[0].squeeze(0).detach()
                    context = context[:, None,...].repeat(1, image.shape[1], 1, 1)
                    obs['natural_language_embedding'] = context
                    context_image_tokens = self.image_tokenizer(image[:, :t-num_pred_action+1], context[:, :t-num_pred_action+1], wrist_image, depth_image)  # [B x T x L x C]
            else:
                context_image_tokens = reused_context_image_tokens
                context = obs["natural_language_embedding"]
        else:
            raise Exception('should not set obs_token')


        use_obs_poses = False

        if not self.use_action_head_diff:
            if self.dim_align_type == 0:
                # pad
                noisy_action_tokens = F.pad(noisy_action_tokens, (0,self.token_embedding_size -noisy_action_tokens.shape[-1]), mode='constant', value=0)
            elif self.dim_align_type == 1:
                # linear
                noisy_action_tokens = self.linear_dim_aligner(noisy_action_tokens)
            elif self.dim_align_type == 2:
                pass


        if self.use_action_head_diff and noisy_action_tokens is not None and timesteps is not None:
            timestep_tokens = self.time_emb(timesteps)
            timestep_tokens = timestep_tokens[:,None,:]

            null_input = torch.zeros([noisy_action_tokens.shape[0], noisy_action_tokens.shape[1], self.token_embedding_size]).to(timestep_tokens.device)
            if self.use_action_head_diff == 4:
                null_input = torch.empty([0, noisy_action_tokens.shape[1], self.token_embedding_size]).to(timestep_tokens.device)

            if self.use_action_head_diff == 4:
                full_tokens = torch.cat([context_image_tokens[:, :t-num_pred_action+1].flatten(1, 2),  # TODO remove t-num_pred_action+1
                                ], axis=-2)
            else:
                full_tokens = torch.cat([context_image_tokens[:, :t-num_pred_action+1].flatten(1, 2),  # TODO remove t-num_pred_action+1
                                null_input[:, :]], axis=-2)
            context_img_len = context_image_tokens[:, :t-num_pred_action+1].flatten(1, 2).shape[1]
            full_tokens = torch.cat([context[:, 0], full_tokens], dim=1)

            
            output_logits_orig = self.transformer(inputs_embeds=full_tokens)[0]
            output_logits = output_logits_orig[:, context[:, 0].shape[1] :, :]
            output_logits = output_logits[:, context_img_len:]
            
            
            if self.use_action_head_diff in [2]:
                # token diffusion head
                  
                b, t, c = noisy_action_tokens.shape
                noisy_action_tokens1 = F.pad(noisy_action_tokens, (0,self.token_embedding_size -noisy_action_tokens.shape[-1]), mode='constant', value=0)
                output = self.diffuse_action_head(noisy_action_tokens1.flatten(0, 1), output_logits.flatten(0, 1), timestep_tokens.repeat(1, t, 1).flatten(0, 1))
                if ret_feats:
                    return output.view(b, t, c), output_logits
                return output.view(b, t, c)
            else:
                raise Exception("not implement")
        elif noisy_action_tokens is not None and timesteps is not None:
            timestep_tokens = self.time_emb(timesteps)

            timestep_tokens = timestep_tokens[:,None,:]

            obs_tokens = context_image_tokens[:, :t-num_pred_action+1].flatten(1, 2)
            
            context_img_len = obs_tokens.shape[1]
            
            full_tokens = torch.cat([obs_tokens,  # TODO remove t-num_pred_action+1
                                     timestep_tokens, noisy_action_tokens[:, :]], axis=-2)

            full_tokens = torch.cat([context[:, 0], full_tokens], dim=1)

            if self.attn_implementation == 'eager':
                output_logits = self.transformer(inputs_embeds=full_tokens)[0]
            else:
                output_logits = self.transformer(inputs_embeds=full_tokens.to(torch.float16))[0]

            output_logits = output_logits[:, context[:, 0].shape[1] :, :]
            def ret_func(x):
                if ret_feats:
                    return x, context_image_tokens
                else:
                    return x
            if reg_token_nums > 0:
                return ret_func(torch.cat([output_logits[:, context_img_len:context_img_len+reg_token_nums], output_logits[:, context_img_len+reg_token_nums+1:]], dim=1), )
            else:
                return ret_func(output_logits[:, context_img_len+1:])



    # TODO
    @torch.no_grad()
    def inference_withfeats(self, obs: dict, obs_tokens=None, num_pred_action=4, abs_pose=0, horizon=-1, reg_prediction_nums=0, pad_diff_nums=0, cfg=0):
        # for AutoRegressive Inference
        
        wrist_image = obs["wrist_image"] if "wrist_image" in obs else None
        
        obs["image"] = obs["image"][:, -(self.time_sequence_length-num_pred_action+1):]
        image = obs["image"]
        b = image.shape[0]
        t = self.time_sequence_length + pad_diff_nums



        if horizon == -1:
            cond_data = torch.zeros(size=(b, t,  self.trajectory_dim), device=image.device, dtype=image.dtype)
        else:    
            cond_data = torch.zeros(size=(b, t,  self.trajectory_dim), device=image.device, dtype=image.dtype)
        # print(cond_data.shape)

        # import ipdb;ipdb.set_trace()
        import torch.nn.functional as F
        trajectory = torch.randn(
            size=cond_data.shape, 
            dtype=cond_data.dtype,
            device=cond_data.device,
            generator=None)

        num_inference_steps = self.num_inference_steps         
        self.noise_scheduler_eval.set_timesteps(num_inference_steps)
        feats = None


        for t in self.noise_scheduler_eval.timesteps:
            # 1. apply conditioning
            # trajectory[condition_mask] = cond_data[condition_mask]
            t = t.cuda()
            # 2. predict model output
            # import ipdb;ipdb.set_trace()

            if reg_prediction_nums > 0:
                noisy_action_tokens = torch.cat([torch.zeros_like(trajectory[:, :reg_prediction_nums]), trajectory,], dim=1)
            else:
                noisy_action_tokens = trajectory
            
            if feats is None:
                model_output, feats = self(obs, act=None, obs_tokens=None, act_tokens=None, noisy_action_tokens = noisy_action_tokens, timesteps=t.repeat(len(trajectory)), num_pred_action=num_pred_action, ret_feats=True)
            else:
                b, traj_len, c = noisy_action_tokens.shape
                noisy_action_tokens1 = F.pad(noisy_action_tokens, (0,self.token_embedding_size -noisy_action_tokens.shape[-1]), mode='constant', value=0)
                timestep_tokens = self.time_emb(t.repeat(len(trajectory)))
                timestep_tokens = timestep_tokens[:,None,:]
                if self.use_action_head_diff == 5:
                    # import ipdb;ipdb.set_trace()
                    noisy_action_tokens_t = noisy_action_tokens.flatten(1,2)
                    
                    noisy_action_tokens1 = F.pad(noisy_action_tokens_t, (0,self.token_embedding_size -noisy_action_tokens_t.shape[-1]), mode='constant', value=0)
                    model_output = self.diffuse_action_head(noisy_action_tokens1, feats[:, 0], timestep_tokens.repeat(1, traj_len, 1)[:, 0])

                elif self.use_action_head_diff == 4:



                    b, t, c = noisy_action_tokens.shape
                    noisy_action_tokens1 = F.pad(noisy_action_tokens, (0,self.token_embedding_size -noisy_action_tokens.shape[-1]), mode='constant', value=0)
                    # import ipdb;ipdb.set_trace()
                    
                    pad_inp = torch.zeros([noisy_action_tokens.shape[0], noisy_action_tokens.shape[1], self.token_embedding_size]).to(timestep_tokens.device)

                    output_logits = torch.cat([feats, pad_inp], dim=1)
                    output = self.diffuse_action_head(noisy_action_tokens1, output_logits, timestep_tokens.repeat(1, t, 1))
                    output = output[:, -noisy_action_tokens.shape[1]:, :]

                else:
                    # import ipdb;ipdb.set_trace()
                    model_output = self.diffuse_action_head(noisy_action_tokens1.flatten(0, 1), feats.flatten(0, 1), timestep_tokens.repeat(1, traj_len, 1).flatten(0, 1))
                model_output = model_output.view(b, traj_len, c)
            model_output = model_output[:, :]
            if reg_prediction_nums > 0:
                model_output = model_output[:, reg_prediction_nums:]
            
            if cfg:
                noise_pred_uncond, noise_pred_text = model_output.chunk(2)
                noise_pred = noise_pred_uncond + cfg * (noise_pred_text - noise_pred_uncond)
                model_output = torch.cat([noise_pred, noise_pred], dim=0)
            b, num, dim = model_output.shape


            if horizon == -1:
                trajectory = self.noise_scheduler_eval.step(model_output, t, trajectory[...,:,:], generator=None,).prev_sample
            else:
                trajectory = self.noise_scheduler_eval.step(model_output, t, trajectory[..., :,:], generator=None,).prev_sample
        if cfg:
            trajectory = trajectory[:len(trajectory)//2]
        trajectory = trajectory[..., -num_pred_action:, :]

        if self.trajectory_dim == 7:
            # For OXE
            return {'world_vector': trajectory[...,:3],
                    'rotation_delta': trajectory[...,3:6],
                    'gripper_closedness_action': trajectory[...,6:7],
                    'terminate_episode': torch.zeros_like(trajectory[...,6:7])  # useless
            }

        else:
            # if it is quaternion, only for Maniskill2
            return {'world_vector': (trajectory[...,:3] + 1)*0.0768 - 0.0768,
                    'rotation_delta': (trajectory[...,3:7] + 1)*0.0768 - 0.0768,
                    'gripper_closedness_action': trajectory[...,7:8],
                    'terminate_episode': trajectory[...,8:]
            }



    # TODO
    @torch.no_grad()
    def inference(self, obs: dict, obs_tokens=None, num_pred_action=4, abs_pose=0, horizon=-1, reg_prediction_nums=0, pad_diff_nums=0, cfg=0):
        # for AutoRegressive Inference
        
        wrist_image = obs["wrist_image"] if "wrist_image" in obs else None
        
        obs["image"] = obs["image"][:, -(self.time_sequence_length-num_pred_action+1):]
        image = obs["image"]
        b = image.shape[0]
        t = self.time_sequence_length + pad_diff_nums



        if horizon == -1:
            cond_data = torch.zeros(size=(b, t,  self.trajectory_dim), device=image.device, dtype=image.dtype)
        else:    
            cond_data = torch.zeros(size=(b, t,  self.trajectory_dim), device=image.device, dtype=image.dtype)
        # print(cond_data.shape)

        # import ipdb;ipdb.set_trace()
        import torch.nn.functional as F
        trajectory = torch.randn(
            size=cond_data.shape, 
            dtype=cond_data.dtype,
            device=cond_data.device,
            generator=None)

        num_inference_steps = self.num_inference_steps
        if 'DENOISING_STEPS' in os.environ: # This is only for ablation study
            num_inference_steps = int(os.environ['DENOISING_STEPS'])

        self.noise_scheduler_eval.set_timesteps(num_inference_steps)
        img_feats = None

       
        for t in self.noise_scheduler_eval.timesteps:
            # 1. apply conditioning
            t = t.cuda()
            # 2. predict model output
            

            if reg_prediction_nums > 0:
                noisy_action_tokens = torch.cat([torch.zeros_like(trajectory[:, :reg_prediction_nums]), trajectory,], dim=1)
            else:
                noisy_action_tokens = trajectory
            

            if img_feats is None:
                # if 'DEBUG' in os.environ:
                #     import ipdb;ipdb.set_trace()
                model_output, img_feats = self(obs, act=None, obs_tokens=None, act_tokens=None, noisy_action_tokens = noisy_action_tokens,
                                 timesteps=t.repeat(len(trajectory)), num_pred_action=num_pred_action, ret_feats=True)
            else:
                model_output = self(obs, act=None, obs_tokens=None, act_tokens=None, noisy_action_tokens = noisy_action_tokens,
                                 timesteps=t.repeat(len(trajectory)), num_pred_action=num_pred_action, ret_feats=False, reused_context_image_tokens=img_feats)
            if reg_prediction_nums > 0:
                model_output = model_output[:, reg_prediction_nums:]
            
            if cfg:
                noise_pred_uncond, noise_pred_text = model_output.chunk(2)
                noise_pred = noise_pred_uncond + cfg * (noise_pred_text - noise_pred_uncond)
                model_output = torch.cat([noise_pred, noise_pred], dim=0)
            # model_output = network(obs_new, noisy_trajectory=trajectory, timesteps=t)
            b, num, dim = model_output.shape

            if horizon == -1:
                trajectory = self.noise_scheduler_eval.step(model_output, t, trajectory[...,:,:], generator=None,).prev_sample
            else:
                trajectory = self.noise_scheduler_eval.step(model_output, t, trajectory[..., :,:], generator=None,).prev_sample
        
        
        if cfg:
            trajectory = trajectory[:len(trajectory)//2]
        trajectory = trajectory[..., -num_pred_action:, :]

        if self.trajectory_dim == 7:
            # For OXE
            return {'world_vector': trajectory[...,:3],
                    'rotation_delta': trajectory[...,3:6],
                    'gripper_closedness_action': trajectory[...,6:7],
                    'terminate_episode': torch.zeros_like(trajectory[...,6:7])  # useless
            }
        else:
            # if it is quaternion, only for Maniskill2
            return {'world_vector': (trajectory[...,:3] + 1)*0.0768 - 0.0768,
                    'rotation_delta': (trajectory[...,3:7] + 1)*0.0768 - 0.0768,
                    'gripper_closedness_action': trajectory[...,7:8],
                    'terminate_episode': trajectory[...,8:]
            }
