#######################################################################################
#                                                                                     #
#   VAE Decoder Robert classes                                                        #
#                                                                                     #
#   This file contains an alternated version of the Roberta Model class. Classes      #
#   that are different from the original, have the prefix VAE_Decoder.                #
#                                                                                     #
#    The main classes:                                                                #
#        - VAE_Decoder_RobertaForCausalLM, consisting of:                             #
#            - VAE_Decoder_RobertaModel, consisting of:                               #
#                - RobertaEmbeddings                                                  #
#                - VAE_Decoder_RobertaEncoder, consisting of:                         #
#                    - VAE_Decoder_RobertaLayer blocks, consisting of:                #
#                        - VAE_DecoderRobertaAttention, consisting of:                #
#                            - VAE_Decoder_RobertaSelfAttention                       #
#                            - RobertaSelfOutput                                      #
#                        - RobertaIntermediate                                        #
#                        - RobertaOutput                                              #
#                - RobertaPooler                                                      #
#            - RobertaLMHead                                                          #
#                                                                                     #
#######################################################################################

# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch RoBERTa model. """

import math
import warnings

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

from transformers.activations import ACT2FN, gelu
from transformers.configuration_roberta import RobertaConfig
from transformers.modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPooling,
    CausalLMOutput,
)
from transformers.modeling_utils import (
    PreTrainedModel,
    apply_chunking_to_forward,
    find_pruneable_heads_and_indices,
    prune_linear_layer,
)
from transformers.utils import logging

logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "RobertaConfig"
_TOKENIZER_FOR_DOC = "RobertaTokenizer"

ROBERTA_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "roberta-base",
    "roberta-large",
    "roberta-large-mnli",
    "distilroberta-base",
    "roberta-base-openai-detector",
    "roberta-large-openai-detector",
    # See all RoBERTa models at https://huggingface.co/models?filter=roberta
]


class RobertaEmbeddings(nn.Module):
    """
    Same as BertEmbeddings with a tiny tweak for positional embeddings indexing.
    """

    # Copied from transformers.modeling_bert.BertEmbeddings.__init__
    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))

        # End copy
        self.padding_idx = config.pad_token_id
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size, padding_idx=self.padding_idx
        )

    def forward(self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None):
        if position_ids is None:
            if input_ids is not None:
                # Create the position ids from the input token ids. Any padded tokens remain padded.
                position_ids = create_position_ids_from_input_ids(input_ids, self.padding_idx).to(input_ids.device)
            else:
                position_ids = self.create_position_ids_from_inputs_embeds(inputs_embeds)

        # Copied from transformers.modeling_bert.BertEmbeddings.forward
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]

        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

    def create_position_ids_from_inputs_embeds(self, inputs_embeds):
        """We are provided embeddings directly. We cannot infer which are padded so just generate
        sequential position ids.
        :param torch.Tensor inputs_embeds:
        :return torch.Tensor:
        """
        input_shape = inputs_embeds.size()[:-1]
        sequence_length = input_shape[1]

        position_ids = torch.arange(
            self.padding_idx + 1, sequence_length + self.padding_idx + 1, dtype=torch.long, device=inputs_embeds.device
        )
        return position_ids.unsqueeze(0).expand(input_shape)


# Copied from transformers.modeling_bert.BertSelfAttention with Bert->Roberta
class VAE_Decoder_RobertaSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads)
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
            self,
            hidden_states,
            attention_mask=None,
            latent_layer_memory_i=None,
            head_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            output_attentions=False,
    ):
        mixed_query_layer = self.query(hidden_states)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        if encoder_hidden_states is not None:

            print("--> Warning: encoder decoder attention is ON, this is NOT what you want.")

            mixed_key_layer = self.key(encoder_hidden_states)
            mixed_value_layer = self.value(encoder_hidden_states)
            attention_mask = encoder_attention_mask
        else:
            mixed_key_layer = self.key(hidden_states)
            mixed_value_layer = self.value(hidden_states)

        # >>>>>> Claartje code
        if latent_layer_memory_i is not None:
            mixed_key_layer = torch.cat((latent_layer_memory_i, mixed_key_layer), dim=1)
            mixed_value_layer = torch.cat((latent_layer_memory_i, mixed_value_layer), dim=1)
        # <<<<<< End Claartje code

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        # batch, n_heads, seq_len, seq_len + 1 (if latent is added)
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        if attention_mask is not None:
            # >>>>>> Claartje code
            if latent_layer_memory_i is not None:
                # If there is a latent vector added to the key, value matrices, the attention mask should
                # not mask attending to this. When tokens are masked their value is set to -1000.0 and
                # when they are not masked they are set to 0.0. We want all tokens to have access to the latents
                # so we will concat series of 0.0 to the attention mask (in front) to not mask the latent.
                batch, _, seq_l, _ = attention_mask.shape
                extension = torch.zeros(batch, 1, seq_l, 1).type_as(attention_mask)
                attention_mask_extended_for_memory = torch.cat((extension,
                                                                attention_mask), dim=3)
                attention_scores = attention_scores + attention_mask_extended_for_memory
            # <<<<<< End Claartje code
            else:
                # Apply the attention mask is (precomputed for all layers in RobertaModel forward() function)
                attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        print("attention_probs.shape", attention_probs.shape)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        return outputs


# Copied from transformers.modeling_bert.BertSelfOutput
class RobertaSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


# Copied from transformers.modeling_bert.BertAttention with Bert->Roberta
class VAE_Decoder_RobertaAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self = VAE_Decoder_RobertaSelfAttention(config)
        self.output = RobertaSelfOutput(config)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
        )

        # Prune linear layers
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
            self,
            hidden_states,
            attention_mask=None,
            latent_layer_memory_i=None,  # >>>>>> Claartje code
            head_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            output_attentions=False,
    ):
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            latent_layer_memory_i,  # >>>>>> Claartje code
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            output_attentions,
        )
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


# Copied from transformers.modeling_bert.BertIntermediate
class RobertaIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


# Copied from transformers.modeling_bert.BertOutput
class RobertaOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


# Copied from transformers.modeling_bert.BertLayer with Bert->Roberta
class VAE_Decoder_RobertaLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = VAE_Decoder_RobertaAttention(config)
        self.is_decoder = config.is_decoder
        self.add_cross_attention = config.add_cross_attention
        if self.add_cross_attention:
            assert self.is_decoder, f"{self} should be used as a decoder model if cross attention is added"
            self.crossattention = VAE_Decoder_RobertaAttention(config)
        self.intermediate = RobertaIntermediate(config)
        self.output = RobertaOutput(config)

    def forward(
            self,
            hidden_states,
            attention_mask=None,
            latent_layer_memory_i=None,
            head_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            output_attentions=False,
    ):
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            latent_layer_memory_i,
            head_mask,
            output_attentions=output_attentions,
        )
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        if self.is_decoder and encoder_hidden_states is not None:
            print("This shouldn't happen (cross-attention)")
            assert hasattr(
                self, "crossattention"
            ), f"If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention layers by setting `config.add_cross_attention=True`"
            cross_attention_outputs = self.crossattention(
                attention_output,
                attention_mask,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                output_attentions,
            )
            attention_output = cross_attention_outputs[0]
            outputs = outputs + cross_attention_outputs[1:]  # add cross attentions if we output attention weights

        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )
        outputs = (layer_output,) + outputs
        return outputs

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


# Copied from transformers.modeling_bert.BertEncoder with Bert->Roberta
class VAE_Decoder_RobertaEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([VAE_Decoder_RobertaLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(
            self,
            hidden_states,
            attention_mask=None,
            latent_layer_memory=None,
            head_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=False,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None

            if getattr(self.config, "gradient_checkpointing", False):
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, output_attentions)

                    return custom_forward

                if latent_layer_memory is not None:
                    latent_layer_memory_i = latent_layer_memory[i]
                else:
                    latent_layer_memory_i = None

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    attention_mask,
                    latent_layer_memory_i,  # TODO: not sure if this works with this checkpointing thing
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                )
            else:
                if latent_layer_memory is not None:
                    latent_layer_memory_i = latent_layer_memory[i]
                else:
                    latent_layer_memory_i = None
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    latent_layer_memory_i,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    output_attentions,
                )
            hidden_states = layer_outputs[0]
            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=all_hidden_states, attentions=all_attentions
        )


# Copied from transformers.modeling_bert.BertPooler
class VAE_Decoder_RobertaPooler(nn.Module):
    def __init__(self, config, latent_size):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size * 2, latent_size * 2)
        # self.activation = nn.Tanh()  # Claartje code

    def forward(self, hidden_states):
        # >>>>>> Claartje code
        first_token_tensor = hidden_states[:, 0]
        last_token_tensor = hidden_states[:, -1]
        first_last = torch.cat((first_token_tensor, last_token_tensor), 1)
        pooled_output = self.dense(first_last)
        # pooled_output = self.activation(pooled_output)
        # <<<<<< End Claartje code
        return pooled_output



class RobertaPreTrainedModel(PreTrainedModel):
    """An abstract class to handle weights initialization and
    a simple interface for downloading and loading pretrained models.
    """

    config_class = RobertaConfig
    base_model_prefix = "roberta"

    # Copied from transformers.modeling_bert.BertPreTrainedModel._init_weights
    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


class VAE_Decoder_RobertaModel(RobertaPreTrainedModel):
    """
    The model can behave as an encoder (with only self-attention) as well
    as a decoder, in which case a layer of cross-attention is added between
    the self-attention layers, following the architecture described in `Attention is all you need`_ by Ashish Vaswani,
    Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.
    To behave as an decoder the model needs to be initialized with the
    :obj:`is_decoder` argument of the configuration set to :obj:`True`.
    To be used in a Seq2Seq model, the model needs to initialized with both :obj:`is_decoder`
    argument and :obj:`add_cross_attention` set to :obj:`True`; an
    :obj:`encoder_hidden_states` is then expected as an input to the forward pass.
    .. _`Attention is all you need`:
        https://arxiv.org/abs/1706.03762
    """

    authorized_missing_keys = [r"position_ids"]

    # Copied from transformers.modeling_bert.BertModel.__init__ with Bert->Roberta
    def __init__(self, config, add_pooling_layer=False):
        super().__init__(config)
        self.config = config

        self.embeddings = RobertaEmbeddings(config)
        self.encoder = VAE_Decoder_RobertaEncoder(config)

        # self.pooler = VAE_Decoder_RobertaPooler(config) if add_pooling_layer else None
        self.pooler = None  # >>>>>> Claartje code

        self.init_weights()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """Prunes heads of the model.
        heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
        See base class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    # Copied from transformers.modeling_bert.BertModel.forward
    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            latent_layer_memory=None,  # >>>>>> Claartje code
            latent_embedding=None,  # >>>>>> Claartje code
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        r"""
        encoder_hidden_states  (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention
            if the model is configured as a decoder.
        encoder_attention_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask
            is used in the cross-attention if the model is configured as a decoder.
            Mask values selected in ``[0, 1]``:
            ``1`` for tokens that are NOT MASKED, ``0`` for MASKED tokens.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)

        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # HERE THE CAUSAL / LEFT-TO-RIGHT / FORWARD LOOKING MASK IS MADE!
        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, device)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds
        )

        # >>>>>> Claartje code
        if latent_embedding is not None:
            embedding_output += latent_embedding.unsqueeze(1)
        # <<<<<< End Claartje code

        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            latent_layer_memory=latent_layer_memory,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]

        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class VAE_Decoder_RobertaForCausalLM(RobertaPreTrainedModel):
    authorized_missing_keys = [r"position_ids", r"predictions.decoder.bias"]
    authorized_unexpected_keys = [r"pooler"]

    def __init__(self, config):
        super().__init__(config)

        self.config = config

        # >>>>>> Claartje code
        config.is_decoder = True
        config.add_cross_attention = False
        config.return_dict = True
        # <<<<<< End Claartje code

        self.roberta = VAE_Decoder_RobertaModel(config, add_pooling_layer=False)
        self.lm_head = RobertaLMHead(config)

        # >>>>>> Claartje code
        self.latent_to_memory_projection = None
        self.latent_to_embedding_projection = None
        # <<<<<< End Claartje code

        self.init_weights()

    def add_latent_projection_layers(self, latent_size, hidden_size, n_layers,
                                     add_latent_via_memory=True, add_latent_via_embeddings=True):
        # >>>>>> Claartje code
        # TODO: In the Optimus they have bias=False, but can't find good reason for that?
        if add_latent_via_memory:
            self.latent_to_memory_projection = nn.Linear(latent_size, hidden_size * n_layers)
            self.latent_to_memory_projection.weight.data.normal_(mean=0.0, std=self.config.initializer_range)

        if add_latent_via_embeddings:
            self.latent_to_embedding_projection = nn.Linear(latent_size, hidden_size)
            self.latent_to_embedding_projection.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        # <<<<<< End Claartje code

    def get_output_embeddings(self):
        return self.lm_head.decoder

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            latent_z=None,
            add_latent_via_embeddings=True,
            add_latent_via_memory=True,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            return_exact_match_acc=False,
            return_predictions=False,
            return_cross_entropy=True,
            reduce_loss=True
    ):
        r"""
        encoder_hidden_states  (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention
            if the model is configured as a decoder.
        encoder_attention_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask
            is used in the cross-attention if the model is configured as a decoder.
            Mask values selected in ``[0, 1]``:
            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the left-to-right language modeling loss (next word prediction).
            Indices should be in ``[-100, 0, ..., config.vocab_size]`` (see ``input_ids`` docstring)
            Tokens with indices set to ``-100`` are ignored (masked), the loss is only computed for the tokens with
            labels in ``[0, ..., config.vocab_size]``
        Returns:
        Example::
            from transformers import RobertaTokenizer, RobertaForCausalLM, RobertaConfig
            import torch
            tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
            config = RobertaConfig.from_pretrained("roberta-base", return_dict=True)
            config.is_decoder = True
            model = RobertaForCausalLM.from_pretrained('roberta-base', config=config)
            inputs = tokenizer(Hello, my dog is cute", return_tensors="pt")
            outputs = model(**inputs)
            prediction_logits = outputs.logits
        """

        print("Test upload yes")

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # >>>>>> Claartje code
        if add_latent_via_memory:
            latent_layer_memory = self.latent_to_memory_projection(latent_z)
            # Makes tuple of equally sized tensors of (batch x 1 x hidden_size)
            latent_layer_memory = torch.split(latent_layer_memory.unsqueeze(1), self.config.hidden_size, dim=2)
        else:
            latent_layer_memory = None

        if add_latent_via_embeddings:
            latent_embedding = self.latent_to_embedding_projection(latent_z)
        else:
            latent_embedding = None
        # <<<<<< End Claartje code

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            latent_layer_memory=latent_layer_memory,  # >>>>>> Claartje code
            latent_embedding=latent_embedding,  # >>>>>> Claartje code
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=True,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # GET OUTPUTS
        sequence_output = outputs[0]

        print("len(sequence_output)", len(sequence_output))

        prediction_scores = self.lm_head(sequence_output)

        # SHIFT SOME THINGS TO MAKE SURE WE ARE COMPARING THE RIGHT THINGS
        # we are doing next-token prediction; shift prediction scores and input ids by one
        logits = prediction_scores[:, :-1, :].contiguous()

        if return_cross_entropy:
            if labels is not None:
                labels = labels[:, 1:].contiguous()
                labels = labels.reshape(-1)
            else:
                print("Can't return CE loss if no labels are provided! Quitting..."); quit()

        # Get rid of the sequence dimension
        batch_size, seq_len, vocab_size = logits.shape
        logits = logits.reshape(-1, vocab_size)

        predictions = None
        exact_match_acc = None
        cross_entropy = None
        probs = None

        if return_cross_entropy or return_predictions:
            probs = torch.nn.functional.softmax(logits, dim=-1)
            log_probs = torch.log(probs)

            if labels is not None:
                reduction = 'mean' if reduce_loss else 'none'
                cross_entropy = torch.nn.functional.nll_loss(log_probs, labels, reduction=reduction)

                # still reduce along sequence dimension
                if reduction == 'none':
                    cross_entropy = cross_entropy.reshape(batch_size, seq_len).mean(dim=1)

            if return_predictions or return_exact_match_acc:
                predictions = probs.argmax(-1)


                if return_exact_match_acc and labels is not None:

                    correct = (labels == predictions).float()
                    correct = correct.reshape(batch_size, seq_len)
                    correct = correct.mean(dim=-1)  # per sequence
                    exact_match_acc = correct.mean(dim=0)

                predictions = predictions.reshape((batch_size, seq_len))

        if logits is not None:
            logits = logits.reshape((batch_size, seq_len, vocab_size))

        return_dict = {
            "cross_entropy": cross_entropy,
            "predictions": predictions,
            "exact_match_acc": exact_match_acc,
            "logits": logits,
            "hidden_states": outputs.hidden_states,
            "attentions": outputs.attentions
        }

        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((cross_entropy,) + output) if cross_entropy is not None else output

        return return_dict

        # Used to be:
        # return CausalLMOutput(
        #     loss=cr,
        #     logits=prediction_scores,
        #     hidden_states=outputs.hidden_states,
        #     attentions=outputs.attentions,
        # )

    def prepare_inputs_for_generation(self, input_ids, attention_mask=None, **model_kwargs):
        input_shape = input_ids.shape

        # if model is used as a decoder in encoder-decoder model, the decoder attention mask is created on the fly
        if attention_mask is None:
            attention_mask = input_ids.new_ones(input_shape)

        return {"input_ids": input_ids, "attention_mask": attention_mask}



class RobertaLMHead(nn.Module):
    """Roberta Head for masked language modeling."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        self.decoder.bias = self.bias

    def forward(self, features, **kwargs):
        x = self.dense(features)
        x = gelu(x)
        x = self.layer_norm(x)

        # project back to size of vocabulary with bias
        x = self.decoder(x)

        return x


def create_position_ids_from_input_ids(input_ids, padding_idx):
    """Replace non-padding symbols with their position numbers. Position numbers begin at
    padding_idx+1. Padding symbols are ignored. This is modified from fairseq's
    `utils.make_positions`.
    :param torch.Tensor x:
    :return torch.Tensor:
    """
    # The series of casts and type-conversions here are carefully balanced to both work with ONNX export and XLA.
    mask = input_ids.ne(padding_idx).int()
    incremental_indices = torch.cumsum(mask, dim=1).type_as(mask) * mask
    return incremental_indices.long() + padding_idx