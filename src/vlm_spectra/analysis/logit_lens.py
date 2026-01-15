from __future__ import annotations

from typing import Dict, List, Tuple, Union

import torch


def _normalize_hidden_states(
    hidden_states: Union[torch.Tensor, Tuple[torch.Tensor, ...]]
) -> torch.Tensor:
    if isinstance(hidden_states, tuple):
        hidden_states_list = []
        for hs in hidden_states:
            if hs is not None:
                if hs.dim() == 3:
                    if hs.size(0) > 1:
                        raise ValueError(
                            "Batch size > 1 not supported for logit lens yet"
                        )
                    hs = hs.squeeze(0)
                hidden_states_list.append(hs)
        return torch.stack(hidden_states_list)
    if hidden_states.dim() == 4:
        if hidden_states.size(1) > 1:
            raise ValueError("Batch size > 1 not supported for logit lens yet")
        return hidden_states.squeeze(1)
    return hidden_states


def compute_logit_lens(
    hidden_states: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
    norm: torch.nn.Module,
    lm_head: torch.nn.Module,
    tokenizer,
    top_k: int = 5,
) -> List[List[List[Tuple[str, str]]]]:
    """Compute top-k token predictions at each layer and position."""
    normalized_states = _normalize_hidden_states(hidden_states)

    all_top_tokens: List[List[List[Tuple[str, str]]]] = []
    num_layers = len(normalized_states)

    for layer in range(num_layers):
        layer_hidden_states = normalized_states[layer]
        normalized = norm(layer_hidden_states)
        logits = lm_head(normalized)
        probs = torch.softmax(logits, dim=-1)
        top_values, top_indices = torch.topk(probs, k=top_k, dim=-1)

        layer_top_tokens = []
        for pos in range(layer_hidden_states.size(0)):
            tokens = [tokenizer.decode(idx.item()) for idx in top_indices[pos]]
            token_probs = [f"{prob.item():.4f}" for prob in top_values[pos]]
            layer_top_tokens.append(list(zip(tokens, token_probs)))
        all_top_tokens.append(layer_top_tokens)

    return all_top_tokens
