import copy
from typing import Optional

import torch
import torch.nn as nn
from timm.models import create_model
from timm.models.vision_transformer import (
    PatchDropout,
    PatchEmbed,
    VisionTransformer,
    checkpoint_seq,
    resample_abs_pos_embed,
)
from vision_tokenizers.clip import clip_vit_hf
from vision_tokenizers.qformer import QFormer, qformer_base_hf
from vision_tokenizers.token_learner import TokenLearnerModule
from transformers import CLIPModel, CLIPProcessor
import os

ORIGIN_TOKENS = 257


def forward_depth_wrist(self: VisionTransformer, image, wrist_image, depth_image):
    # x = self.patch_embed(image)
    # x = self._pos_embed(x)

    # if depth_image is not None and self.use_depth_img:
    #     depth_image = depth_image.repeat(1, 3, 1, 1)
    #     depth_x = self.patch_embed_depth(depth_image)
    #     depth_x = self._pos_embed(depth_x)
    #     x[:, self.num_prefix_tokens :] = x[:, self.num_prefix_tokens :] + depth_x[:, self.num_prefix_tokens :]

    # if wrist_image is not None and self.use_wrist_img:
    #     wrist_x = self.patch_embed_wrist(wrist_image)
    #     wrist_x = wrist_x + self.pos_embed_wrist
    #     wrist_x = PatchDropout(0.3, num_prefix_tokens=0)(wrist_x)

    #     x = torch.cat([x, wrist_x], dim=1)

    if wrist_image is not None and self.use_wrist_img:
        wrist_image = nn.functional.interpolate(wrist_image, size=image.shape[-2:], mode="bilinear", align_corners=False)
        image = torch.cat([image, wrist_image], dim=-1)
    x = self.patch_embed(image)
    x = self._pos_embed(x)
    if wrist_image is not None and self.use_wrist_img:
        wrist_x_idx = self.num_prefix_tokens + self.patch_embed.num_patches // 2
        wrist_x = x[:, wrist_x_idx:]
        wrist_x = PatchDropout(0.5, num_prefix_tokens=0)(wrist_x)
        x = torch.cat([x[:, :wrist_x_idx], wrist_x], dim=1)

    x = self.patch_drop(x)
    x = self.norm_pre(x)
    if self.grad_checkpointing and not torch.jit.is_scripting():
        x = checkpoint_seq(self.blocks, x)
    else:
        x = self.blocks(x)
    x = self.norm(x)
    return x[:, self.num_prefix_tokens :]

def forward_depth_wrist_v2(self: VisionTransformer, image, wrist_image, depth_image):

    image_x =  self.patch_embed(image)
    image_x = self._pos_embed(image_x)
    image_x = self.patch_drop(image_x)
    image_x = self.norm_pre(image_x)
    if self.grad_checkpointing and not torch.jit.is_scripting():
        image_x = checkpoint_seq(self.blocks, image_x)
    else:
        image_x = self.blocks(image_x)
    image_x = self.norm(image_x)
    
    if wrist_image is not None and self.use_wrist_img:
        wrist_image = nn.functional.interpolate(wrist_image, size=image.shape[-2:], mode="bilinear", align_corners=False)
        wrist_x = self.wrist_image_part.patch_embed(wrist_image)
        wrist_x = self.wrist_image_part._pos_embed(wrist_x)
        wrist_x = PatchDropout(0.5, num_prefix_tokens=self.wrist_image_part.num_prefix_tokens)(wrist_x)
        wrist_x = self.wrist_image_part.patch_drop(wrist_x)
        wrist_x = self.wrist_image_part.norm_pre(wrist_x)
        if self.wrist_image_part.grad_checkpointing and not torch.jit.is_scripting():
            wrist_x = checkpoint_seq(self.wrist_image_part.blocks, wrist_x)
        else:
            wrist_x = self.wrist_image_part.blocks(wrist_x)
        wrist_x = self.wrist_image_part.norm(wrist_x)
        x = torch.cat([image_x[:, self.num_prefix_tokens:], wrist_x[:, self.wrist_image_part.num_prefix_tokens:]], dim = 1)

    else:
        x = image_x[:, self.num_prefix_tokens:]
    return x

class RT1ViTImageTokenizer(nn.Module):
    """Tokenizes based on vocab size."""

    def __init__(
        self,
        embedding_output_dim: int,
        use_qformer: bool = True,
        qformer_depth: int = 4,
        num_tokens: int = 32,
        dropout_rate=0.1,
        freeze_backbone=True,
        use_wrist_img=False,
        use_depth_img=False,
        input_size=None,
        vit_forward_version = None,
    ):
        """Instantiates a RT1ImageTokenizer.

        Args:
          embedding_output_dim: The output size of the tokens.
          use_token_learner: Whether to use token learner.
          num_tokens: Relevant only for token learner - the number of learned tokens.
        """
        super().__init__()
        self.embedding_output_dim = embedding_output_dim

        # self.tokenizer = clip_vit_hf(
        #     freeze=freeze_backbone,
        #     gradient_checkpointing=True,
        #     model_path="/xxx/xxx/share_data/Anonymous/clip-vit-large-patch14/",
        # )
        if input_size is None:
            if use_wrist_img:
                if vit_forward_version == 1:
                    img_size = (224,448) # for forward_v1
                else:
                    img_size = 224 # for forward_v2
                # num_tokens = num_tokens + num_tokens
                num_tokens = num_tokens + num_tokens // 2
            else:
                img_size = 224
        else: #str to tuple
            import ast
            img_size = ast.literal_eval(input_size)
            if use_wrist_img:
                h,w = img_size
                if vit_forward_version == 1:
                    img_size = (h,w*2)
                else:
                    img_size = (h,w)
                num_tokens = num_tokens + num_tokens // 2
            # import ipdb;ipdb.set_trace()
        self.tokenizer = create_model(
            "vit_base_patch14_reg4_dinov2.lvd142m", pretrained=True, img_size=img_size, drop_path_rate=0.1, proj_drop_rate=0.1
        )

        if use_depth_img:
            self.tokenizer.patch_embed_depth = copy.deepcopy(self.tokenizer.patch_embed)

        if use_wrist_img:
            # self.tokenizer.patch_embed_wrist = PatchEmbed(img_size=128, patch_size=16, embed_dim=self.tokenizer.embed_dim)
            # # self.tokenizer.pos_embed_wrist = resample_abs_pos_embed(self.tokenizer.pos_embed, (8, 8), num_prefix_tokens=0)
            # self.tokenizer.pos_embed_wrist = nn.Parameter(
            #     torch.randn(1, self.tokenizer.patch_embed_wrist.num_patches, self.tokenizer.embed_dim) * 0.02
            # )
            if vit_forward_version == 1:
                pass
            else:
                self.tokenizer.wrist_image_part = create_model(
                "vit_base_patch14_reg4_dinov2.lvd142m", pretrained=True, img_size=img_size, drop_path_rate=0.1, proj_drop_rate=0.1
            )
                self.tokenizer.wrist_image_part.set_grad_checkpointing()

        self.tokenizer.forward_depth_wrist = forward_depth_wrist_v2.__get__(self.tokenizer, VisionTransformer)
        if vit_forward_version == 1:
            self.tokenizer.forward_depth_wrist = forward_depth_wrist.__get__(self.tokenizer, VisionTransformer)
        self.tokenizer.use_wrist_img = use_wrist_img
        self.tokenizer.use_depth_img = use_depth_img

        if freeze_backbone:
            for param in self.tokenizer.parameters():
                param.requires_grad = False
        self.tokenizer.set_grad_checkpointing()

        # TODO
        # self.tokenizer_proj = CLIPModel.from_pretrained("/xxx/xxx/share_data/Anonymous/clip-vit-large-patch14/").visual_projection
        # self.tokenizer_proj.requires_grad_(False)
        # self.tokenizer = CLIPModel.from_pretrained("/xxx/xxx/share_data/Anonymous/clip-vit-large-patch14/")

        # self.qformer = qformer_base_hf(
        #     num_queries=num_tokens,
        #     embed_dim=768,
        #     depth=4,
        # )
        self.use_qformer = use_qformer
        if use_qformer == 2:
            from flamingo_pytorch import PerceiverResampler
            self.qformer = PerceiverResampler(
                dim = self.embedding_output_dim,
                depth = qformer_depth,
                dim_head = 64,
                heads = self.embedding_output_dim // 64,
                num_latents = num_tokens,    # the number of latents to shrink your media sequence to, perceiver style
                # num_time_embeds = 1  # say you have 4 images maximum in your dialogue
            )
            self.num_tokens = num_tokens
        elif use_qformer:
            self.qformer = QFormer(
                num_queries=num_tokens,
                embed_dim=self.embedding_output_dim,
                depth=qformer_depth,
                num_heads=self.embedding_output_dim // 64,
                mlp_ratio=4,
                qkv_bias=False,
                norm_layer=nn.LayerNorm,
                dropout_rate=dropout_rate,
                drop_path=0.0,
                use_checkpoint=not freeze_backbone,
                with_film=True,
                cross_dim=self.tokenizer.embed_dim,
            )
            self.num_tokens = num_tokens
        else:
            self.num_tokens = self.tokenizer.patch_embed.num_patches + (0 if not use_wrist_img else self.tokenizer.patch_embed_wrist.num_patches)

    @property
    def tokens_per_context_image(self) -> int:
        return self.num_tokens

    def forward(self, image, context=None, wrist_image=None, depth_image=None):
        """Gets image tokens.

        Args:
          image: Images of shape (b, t, 3, h, w) to tokenize.
          context: An optional context vector (e.g., a natural language embedding).
          training: Whether or not we are in training mode.

        Returns:
          tokens: has shape (batch, t, num_tokens_per_timestep, embedding_dim)
        """
        b, t, c, h, w = image.shape

        # Fold the time axis into the batch axis.
        image = image.flatten(0, 1)
        if wrist_image is not None:
            wrist_image = wrist_image.flatten(0, 1)
            # wrist_image = nn.functional.interpolate(wrist_image, size=(h, w), mode="bilinear", align_corners=False)
            # image = torch.cat([image, wrist_image], dim=-1)
        if depth_image is not None:
            depth_image = depth_image.flatten(0, 1)

        if context is not None:
            # assert context.dim() == 3, "Context tensor rank should be 3"
            # context = context.view(b * t, -1)
            context = context.flatten(0, 1)
            while context.dim() != 2:
                context = context.mean(dim=-2)
        tokens = self.get_image_embeddings(image, wrist_image, depth_image)
        # tokens = self.qformer(tokens)
        # TODO: add context in qformer
        if self.use_qformer:
            if self.use_qformer == 2:
                tokens = self.qformer(tokens[:, None,...])[:, 0]
            else:
                tokens = self.qformer(tokens, context)
        # Unflatten the time axis, which was previously flattened into the batch.
        token_num = tokens.shape[1]
        tokens = tokens.view(b, t, token_num, -1)
        return tokens

    def get_image_embeddings(self, image: torch.Tensor, wrist_image=None, depth_image=None) -> torch.Tensor:
        """Gets embeddings from image.

        Args:
          image: Expected to be float32 in range [0, 1] with shape (b, 3, h, w).
          context: Expected to be float32 with shape (b, embedding_dim)
          training: Whether or not we are in training mode.

        Returns:
          tokens of shape (b, num_tokens, embedding_dim)
        """
        # image_tokens = image_tokens.permute(0, 3, 1, 2) # [b, c, h, w]
        # image_tokens = self.tokenizer(image)[0]  # [B, L, C]

        # image_tokens = self.tokenizer.forward_features(image)[:, self.tokenizer.num_prefix_tokens :]
        # import ipdb;ipdb.set_trace()
        image_tokens = self.tokenizer.forward_depth_wrist(image, wrist_image, depth_image)
        
        # image_tokens = self.tokenizer_proj(image_tokens)
        # image_tokens = self.tokenizer.get_image_features(image)
        return image_tokens


if __name__ == "__main__":

    net = RT1ViTImageTokenizer(embedding_output_dim=768, use_wrist_img = True)
    import ipdb;ipdb.set_trace()
    net = net.cuda()
    image = torch.randn((2, 15, 3, 224, 224), device="cuda")
    language_embedding = torch.randn((2, 15, 77, 768), device="cuda")
    wrist = torch.randn((2,15,3,128,128), device = 'cuda')
    output = net(image, language_embedding, wrist)
    print(output.shape)
