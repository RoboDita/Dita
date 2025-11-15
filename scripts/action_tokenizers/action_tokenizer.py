from typing import List, Optional

import torch


class DiscreteActionTokenizer:
    """Tokenizes based on vocab size."""

    def __init__(self, action_spec: dict, vocab_size: List[int], action_order: Optional[list] = None):
        """Instantiates an DiscreteActionTokenizer.

        Args:
            action_spec: Tensor spec of the expected action tensor.
            vocab_size: Number of buckets to discretize action to.
            action_order: Order of the action names, used to discern the order of
                tokenized actions to detokenize and assemble back to action tensor
        """
        self._action_spec = action_spec
        self._vocab_size = vocab_size
        if action_order is None:
            self._action_order = list(action_spec.keys())
        else:
            for action in action_order:
                if action not in action_spec.keys():
                    raise ValueError("actions: %s not found in action_spec: %s" % (action, list(action_spec.keys())))
                assert action in action_spec.keys()
            self._action_order = action_order
        self._tokens_per_action = 1  # add [SoA]
        for action in self._action_order:
            action_shape = self._action_spec[action]["tensor"].shape
            if len(action_shape) != 1:
                raise ValueError("Only action shapes with single dimension supported, got %s" % action_shape)
            if self._action_spec[action]["tensor"].dtype == torch.int32:
                # Int32 actions are already assumed to be tokens.
                self._tokens_per_action += 1
            else:
                self._tokens_per_action += action_shape[0]

        # We measure # of action tokens in two different way. One is by checking
        # from action_order (above) and the other is by looping through the
        # action spec (below). We assert the # of action tokens are the same
        # calculated by these two ways. This will assure action_order is correctly
        # configured, otherwise, it will through an error in the assert.
        num_action_token = 1
        for spec in self._action_spec.values():
            if spec["tensor"].dtype == torch.int32:
                num_action_token += 1
            else:
                num_action_token += spec["tensor"].shape[-1]
        assert num_action_token == self._tokens_per_action

    @property
    def tokens_per_action(self) -> int:
        return self._tokens_per_action

    @property
    def action_spec(self) -> dict:
        return self._action_spec

    @property
    def action_order(self) -> list:
        return self._action_order

    def tokenize(self, action: dict) -> torch.Tensor:
        """Tokenizes an action."""
        action_tokens = []
        vocab_idx = 1
        for i, k in enumerate(self._action_order):
            a = action[k]  # a is [batch, actions_size]
            spec = self._action_spec[k]
            if spec["tensor"].dtype == torch.int32:
                # Int32 actions are already assumed to be tokens, assume it is smaller
                # than the vocab size, so all we need to do is pad zeros.
                # assert torch.sum(a, dim=-1) == 1
                # extract the token [batch, 1]
                # argmax: [0, 0, 0] -> 0, [0, 1, 0] -> 1,
                token = torch.argmax(a, dim=-1, keepdim=True)
                assert torch.max(token) < self._vocab_size[i]
                token = token + vocab_idx
                vocab_idx += self._vocab_size[i]
            else:
                a = torch.clamp(a, spec["minimum"], spec["maximum"])
                # Normalize the action [batch, actions_size]
                token = (a - spec["minimum"]) / (spec["maximum"] - spec["minimum"])  # [0, 1]
                # Bucket and discretize the action to vocab_size, [batch, actions_size]
                token = (token * (self._vocab_size[i] - 1)).to(torch.int32) + vocab_idx
                vocab_idx += self._vocab_size[i]
            action_tokens.append(token)
        # Append all actions, [batch, all_actions_size]
        action_tokens = torch.cat(action_tokens, dim=-1)
        # [SoA] token
        action_tokens = torch.cat([action_tokens.new_zeros(action_tokens.shape[:-1] + (1,)), action_tokens], dim=-1)
        return action_tokens

    def detokenize(self, action_tokens: torch.Tensor) -> dict:
        """Detokenizes an action."""
        # action_tokens: [..., L, C], value range: [0, vocab_size)

        action = {}
        flatten_action = []
        vocab_idx = 1
        token_index = 0
        topk_list = [5, 10, 20, 50] # None
        topk_result = {}
        for i, k in enumerate(self._action_order):
            spec = self._action_spec[k]
            action_dim = spec["tensor"].shape[0]
            if spec["tensor"].dtype == torch.int32:
                # Int32 actions are already assumed to be tokens.
                action_pred = action_tokens[..., token_index : token_index + 1, vocab_idx : vocab_idx + self._vocab_size[i]]
                # A poor model may output tokens outside the allowed range, in that case
                # set them to a default value, the 0 token in this case.
                action_pred = torch.argmax(action_pred, dim=-1)
                flatten_action.append(action_pred + vocab_idx)
                action[k] = torch.nn.functional.one_hot(action_pred.squeeze(-1).long(), num_classes=action_dim).to(torch.int32)
                token_index += 1
                vocab_idx += self._vocab_size[i]
            else:
                action_pred = action_tokens[..., token_index : token_index + action_dim, vocab_idx : vocab_idx + self._vocab_size[i]]
                if i < 2 and topk_list is not None:
                    for topk in topk_list:
                        _, indices = torch.topk(action_pred, topk, dim=-1)
                        if topk not in topk_result:
                            topk_result[topk] = indices + vocab_idx
                        else:
                            topk_result[topk] = torch.cat([topk_result[topk], indices+vocab_idx], dim=-2)
                action_pred = torch.argmax(action_pred, dim=-1)
                flatten_action.append(action_pred + vocab_idx)
                action_pred = action_pred / (self._vocab_size[i] - 1)
                action_pred = (action_pred * (spec["maximum"] - spec["minimum"])) + spec["minimum"]
                action[k] = action_pred
                token_index += action_dim
                vocab_idx += self._vocab_size[i]

        return action, torch.cat(flatten_action, dim=-1), topk_result # topk_result[n] (b, len, action_dim, n)


if __name__ == "__main__":
    import numpy as np

    action_spec = {
        "world_vector": {
            "tensor": torch.empty((3,), dtype=torch.float32),
            "minimum": torch.tensor([-1.0], dtype=torch.float32),
            "maximum": torch.tensor([1.0], dtype=torch.float32),
        },
        "rotation_delta": {
            "tensor": torch.empty((3,), dtype=torch.float32),
            "minimum": torch.tensor([-np.pi / 2], dtype=torch.float32),
            "maximum": torch.tensor([np.pi / 2], dtype=torch.float32),
        },
        "gripper_closedness": {
            "tensor": torch.empty((1,), dtype=torch.float32),
            "minimum": torch.tensor([-1.0], dtype=torch.float32),
            "maximum": torch.tensor([1.0], dtype=torch.float32),
        },
        "terminate": {
            "tensor": torch.empty((2,), dtype=torch.int32),
            "minimum": torch.tensor([0], dtype=torch.int32),
            "maximum": torch.tensor([1], dtype=torch.int32),
        },
    }

    tokenizer = DiscreteActionTokenizer(action_spec, vocab_size=1024, action_order=["terminate", "world_vector", "rotation_delta", "gripper_closedness"])

    print(tokenizer.tokens_per_action)  # 8

    n_repeat = 2
    for _ in range(n_repeat):
        action = {
            "world_vector": torch.rand((3,), dtype=torch.float32) * 2.0 - 1.0,
            "rotation_delta": torch.rand((3,), dtype=torch.float32) * np.pi - np.pi / 2.0,
            "gripper_closedness": torch.rand((1,), dtype=torch.float32),
            "terminate": torch.tensor([0, 1], dtype=torch.int32),
        }
        action_tokens = tokenizer.tokenize(action)
        policy_action = tokenizer.detokenize(action_tokens)

        for k in action:
            print(action[k] - policy_action[k])

        # batched_action = {
        #     'world_vector': torch.rand((2, 3), dtype=torch.float32) * 2.0 - 1.0,
        #     'rotation_delta': torch.rand((2, 3), dtype=torch.float32) * np.pi - np.pi / 2.0,
        #     'gripper_closedness': torch.rand((2, 1), dtype=torch.float32),
        #     'terminate': torch.tensor([[0, 1], [1, 0]], dtype=torch.int32)
        # }
        # action_tokens = tokenizer.tokenize(batched_action)
        # policy_action = tokenizer.detokenize(action_tokens)

        # for k in batched_action:
        #     for a, policy_a in zip(batched_action[k], policy_action[k]):
        #         print(a - policy_a)

# Example Usage:
# Instantiate the action spec
# action_spec = {
#     'terminate': {'tensor': torch.empty((1,), dtype=torch.int32),
#                   'minimum': torch.tensor([0], dtype=torch.float32),
#                   'maximum': torch.tensor([1], dtype=torch.float32)},
#     'world_vector': {'tensor': torch.empty((3,), dtype=torch.float32),
#                      'minimum': torch.tensor([-1.0, -1.0, -1.0], dtype=torch.float32),
#                      'maximum': torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32)},
#     'rotation_delta': {'tensor': torch.empty((3,), dtype=torch.float32),
#                        'minimum': torch.tensor([-1.0, -1.0, -1.0], dtype=torch.float32),
#                        'maximum': torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32)},
#     'gripper_closedness': {'tensor': torch.empty((1,), dtype=torch.float32),
#                            'minimum': torch.tensor([0.0], dtype=torch.float32),
#                            'maximum': torch.tensor([1.0], dtype=torch.float32)}
# }
# # Instantiate the tokenizer
# tokenizer = RT1ActionTokenizer(action_spec, vocab_size=10)
# # Example action tensor
# action_tensor = {
#     'terminate': torch.tensor([[0, 1]]),
#     'world_vector': torch.tensor([[0.9, 0.8, -0.3]]),
#     'rotation_delta': torch.tensor([[-0.1, 0.2, 0.6]]),
#     'gripper_closedness': torch.tensor([[0.9]])
# }
# # Tokenize action
# tokenized_action = tokenizer.tokenize(action_tensor)
# print("Tokenized Action:", tokenized_action)
# # Detokenize action
# detokenized_action = tokenizer.detokenize(tokenized_action)
# print("Detokenized Action:", detokenized_action)
