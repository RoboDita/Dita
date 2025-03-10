import torch
import torch.nn as nn
from typing import Optional
from film_efficientnet.pretrained_efficientnet_encoder import EfficientNetEncoder
from tokenizers_rt1.token_learner import TokenLearnerModule 

ORIGIN_TOKENS = 81
class RT1ImageTokenizer(nn.Module):
    """Tokenizes based on vocab size."""

    def __init__(self, 
                 embedding_output_dim: int, 
                 use_token_learner: bool = False, 
                 num_tokens: int = 8):
        """Instantiates a RT1ImageTokenizer.
        
        Args:
          embedding_output_dim: The output size of the tokens.
          use_token_learner: Whether to use token learner.
          num_tokens: Relevant only for token learner - the number of learned tokens.
        """
        super().__init__()
        self.embedding_output_dim = embedding_output_dim

        self.tokenizer = EfficientNetEncoder(pooling=False, early_film=True)

        self.use_token_learner = use_token_learner
        if self.use_token_learner:
            self.num_tokens = num_tokens
            self.token_learner = TokenLearnerModule(self.embedding_output_dim, num_tokens=self.num_tokens)

    @property
    def tokens_per_context_image(self) -> int:
        if self.use_token_learner:
            return self.num_tokens
        else:
            return ORIGIN_TOKENS

    def forward(self, image: torch.Tensor, context:None):
        """Gets image tokens.

        Args:
          image: Images of shape (b, t, 3, h, w) to tokenize.
          context: An optional context vector (e.g., a natural language embedding).
          training: Whether or not we are in training mode.

        Returns:
          tokens: has shape (batch, t, num_tokens_per_timestep, embedding_dim)
        """
        b, t, c, h, w= image.shape

        # Fold the time axis into the batch axis.
        image = image.view(b * t, c, h, w)  
        if context is not None:
            assert context.dim() == 3, "Context tensor rank should be 3"
            context = context.view(b * t, -1)
        tokens = self.get_image_embeddings(image, context)
        if self.use_token_learner:
            tokens = self.token_learner(tokens)
        # Unflatten the time axis, which was previously flattened into the batch.
        token_num = tokens.shape[1]
        tokens = tokens.view(b, t, token_num, -1)
        return tokens

    def get_image_embeddings(self, image: torch.Tensor, context: Optional[torch.Tensor]) -> torch.Tensor:
        """Gets embeddings from image.

        Args:
          image: Expected to be float32 in range [0, 1] with shape (b, 3, h, w).
          context: Expected to be float32 with shape (b, embedding_dim)
          training: Whether or not we are in training mode.

        Returns:
          tokens of shape (b, num_tokens, embedding_dim)
        """
        # image_tokens = image_tokens.permute(0, 3, 1, 2) # [b, c, h, w]
        image_tokens = self.tokenizer(image, context=context)
        image_tokens = image_tokens.reshape(-1, ORIGIN_TOKENS, self.embedding_output_dim)
        return image_tokens
