"""BERT-style transformer architecture for SurvTRACE (vendored + cleaned up).

Adapted from https://github.com/RyanWangZf/SurvTRACE (MIT License).
"""

import math
import inspect
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class BaseModel(nn.Module):
    def __init__(self, config, *inputs, **kwargs):
        super().__init__()
        self.config = config

    def init_weights(self):
        if self.config.get("pruned_heads"):
            self.prune_heads(self.config["pruned_heads"])
        self.apply(self._init_weights)

    def get_head_mask(self, head_mask, num_hidden_layers):
        if head_mask is not None:
            head_mask = self._convert_head_mask_to_5d(head_mask, num_hidden_layers)
        else:
            head_mask = [None] * num_hidden_layers
        return head_mask

    def _convert_head_mask_to_5d(self, head_mask, num_hidden_layers):
        if head_mask.dim() == 1:
            head_mask = (
                head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            )
            head_mask = head_mask.expand(num_hidden_layers, -1, -1, -1, -1)
        elif head_mask.dim() == 2:
            head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
        assert head_mask.dim() == 5
        head_mask = head_mask.to(dtype=self.dtype)
        return head_mask

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(
                mean=0.0, std=self.config["initializer_range"]
            )
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(
                mean=0.0, std=self.config["initializer_range"]
            )
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


class BertEmbeddings(nn.Module):
    """Construct embeddings from categorical (word) and numerical features."""

    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(
            config["vocab_size"], config["hidden_size"]
        )
        self.num_embeddings = nn.Parameter(
            torch.randn(1, config["num_numerical_feature"], config["hidden_size"]),
            requires_grad=True,
        )
        self.num_embeddings.data.normal_(mean=0.0, std=config["initializer_range"])
        self.LayerNorm = nn.LayerNorm(
            config["hidden_size"], eps=config["layer_norm_eps"]
        )
        self.dropout = nn.Dropout(config["hidden_dropout_prob"])

    def forward(self, input_ids=None, input_x_num=None, inputs_embeds=None):
        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        num_embeddings = torch.unsqueeze(input_x_num, 2) * self.num_embeddings
        embeddings = torch.cat([num_embeddings, inputs_embeds], axis=1)
        embeddings = self.dropout(embeddings)
        return embeddings


class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config["hidden_size"] % config["num_attention_heads"] != 0:
            raise ValueError(
                f"hidden_size ({config['hidden_size']}) is not a multiple of "
                f"num_attention_heads ({config['num_attention_heads']})"
            )
        self.num_attention_heads = config["num_attention_heads"]
        self.attention_head_size = int(
            config["hidden_size"] / config["num_attention_heads"]
        )
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config["hidden_size"], self.all_head_size)
        self.key = nn.Linear(config["hidden_size"], self.all_head_size)
        self.value = nn.Linear(config["hidden_size"], self.all_head_size)
        self.dropout = nn.Dropout(config["attention_probs_dropout_prob"])

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        output_attentions=False,
        **kwargs,
    ):
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)

        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (
            (context_layer, attention_probs)
            if output_attentions
            else (context_layer,)
        )
        return outputs


class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config["hidden_size"], config["hidden_size"])
        self.LayerNorm = nn.LayerNorm(
            config["hidden_size"], eps=config["layer_norm_eps"]
        )
        self.dropout = nn.Dropout(config["hidden_dropout_prob"])

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)

    def forward(self, hidden_states, attention_mask=None, head_mask=None, output_attentions=False, **kwargs):
        self_outputs = self.self(
            hidden_states,
            attention_mask=attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
        )
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]
        return outputs


class BertIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config["hidden_size"], config["intermediate_size"])

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = F.gelu(hidden_states)
        return hidden_states


class BertOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config["intermediate_size"], config["hidden_size"])
        self.LayerNorm = nn.LayerNorm(
            config["hidden_size"], eps=config["layer_norm_eps"]
        )
        self.dropout = nn.Dropout(config["hidden_dropout_prob"])

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        output_attentions=False,
        **kwargs,
    ):
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask=attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
        )
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]

        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        outputs = (layer_output,) + outputs
        return outputs


class DenseVanillaBlock(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        batch_norm=True,
        dropout=0.0,
        activation=nn.ReLU,
        w_init_=lambda w: nn.init.kaiming_normal_(w, nonlinearity="relu"),
    ):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias)
        if w_init_:
            w_init_(self.linear.weight.data)
        self.activation = activation()
        self.batch_norm = nn.BatchNorm1d(out_features) if batch_norm else None
        self.dropout = nn.Dropout(dropout) if dropout else None

    def forward(self, input):
        input = self.activation(self.linear(input))
        if self.batch_norm:
            input = self.batch_norm(input)
        if self.dropout:
            input = self.dropout(input)
        return input


class BertCLS(nn.Module):
    """Classification head for single-event survival."""

    def __init__(self, config):
        super().__init__()
        w_init = lambda w: nn.init.kaiming_normal_(w, nonlinearity="relu")
        net = [
            DenseVanillaBlock(
                config["hidden_size"] * config["num_feature"],
                config["intermediate_size"],
                batch_norm=True,
                dropout=config["hidden_dropout_prob"],
                activation=nn.ReLU,
                w_init_=w_init,
            ),
            nn.Linear(config["intermediate_size"], config["out_feature"]),
        ]
        self.net = nn.Sequential(*net)

    def forward(self, hidden_states):
        hidden_states = hidden_states.flatten(start_dim=1)
        return self.net(hidden_states)


class BertCLSMulti(nn.Module):
    """Classification head for competing-risks (one output per event type)."""

    def __init__(self, config):
        super().__init__()
        w_init = lambda w: nn.init.kaiming_normal_(w, nonlinearity="relu")
        self.net = nn.Sequential(
            DenseVanillaBlock(
                config["hidden_size"] * config["num_feature"],
                config["intermediate_size"],
                batch_norm=True,
                dropout=config["hidden_dropout_prob"],
                activation=nn.ReLU,
                w_init_=w_init,
            )
        )
        self.net_out = nn.ModuleList(
            [
                nn.Linear(config["intermediate_size"], config["out_feature"])
                for _ in range(config["num_event"])
            ]
        )

    def forward(self, hidden_states, event=0):
        hidden_states = hidden_states.flatten(start_dim=1)
        hidden_states = self.net(hidden_states)
        return self.net_out[event](hidden_states)


class BertEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList(
            [BertLayer(config) for _ in range(config["num_hidden_layers"])]
        )

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        output_attentions=False,
        output_hidden_states=True,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None
            layer_outputs = layer_module(
                hidden_states,
                attention_mask=attention_mask,
                head_mask=layer_head_mask,
                output_attentions=output_attentions,
            )
            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        return tuple(
            v
            for v in [hidden_states, all_hidden_states, all_self_attentions]
            if v is not None
        )
