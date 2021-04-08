# coding=utf-8
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

#######################################################################################
#                                                                                     #
#   VAE Decoder Roberta classes                                                       #
#                                                                                     #
#   This file contains an alternated version of the Roberta Model class. Classes      #
#   that are different from the original, have the prefix VaeDecoder.                 #
#                                                                                     #
#    The main classes:                                                                #
#        - VaeDecoderRobertaForCausalLM, consisting of:                               #
#            - VaeDecoderRobertaModel, consisting of:                                 #
#                - RobertaEmbeddings                                                  #
#                - VaeDecoderRobertaEncoder, consisting of:                           #
#                    - VaeDecoderRobertaLayer blocks, consisting of:                  #
#                        - VaeDecoderRobertaAttention, consisting of:                 #
#                            - VAE_Decoder_RobertaSelfAttention                       #
#                            - RobertaSelfOutput                                      #
#                        - RobertaIntermediate                                        #
#                        - RobertaOutput                                              #
#                - VaeDecoderRobertaPooler                                            #
#            - RobertaLMHead                                                          #
#                                                                                     #
#######################################################################################

import math

import torch
import torch.nn as nn
import torch.utils.checkpoint
from torch.nn import CrossEntropyLoss, MSELoss
from transformers import top_k_top_p_filtering

from transformers.activations import ACT2FN, gelu

from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutputWithPoolingAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
)
from transformers.modeling_utils import (
    PreTrainedModel,
    apply_chunking_to_forward,
    find_pruneable_heads_and_indices,
    prune_linear_layer,
)
from transformers.utils import logging
from transformers.models.roberta.configuration_roberta import RobertaConfig

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

    # Copied from transformers.models.bert.modeling_bert.BertEmbeddings.__init__
    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.drop_inputs_decoder = None
        self.drop_inputs_decoder_prob = None

        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")

        # End copy
        self.padding_idx = config.pad_token_id
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size, padding_idx=self.padding_idx
        )

    def forward(
            self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None, past_key_values_length=0
    ):
        if position_ids is None:
            if input_ids is not None:
                # Create the position ids from the input token ids. Any padded tokens remain padded.
                position_ids = create_position_ids_from_input_ids(
                    input_ids, self.padding_idx, past_key_values_length
                ).to(input_ids.device)
            else:
                position_ids = self.create_position_ids_from_inputs_embeds(inputs_embeds)

        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        # Claartje: Input embedding drop-out
        # This function drops full embeddings according to drop-out probability < drop_inputs_decoder_prob >
        # it scales the rest of the embeddings by:  * ( 1 / (1 - dropout_prob) )
        # so if the drop-out probability is 0.2, then the other weights are scaled by 1 / 0.8 = 1.25
        # this is to keep the expected magnitude of the embeddings the same.
        if self.drop_inputs_decoder and self.training:
            # inputs_embeds = torch.nn.functional.dropout2d(inputs_embeds, p=self.drop_inputs_decoder_prob,
            #                                               training=self.training, inplace=False)
            #print("DROPOUT: self.drop_inputs_decoder_prob",self.drop_inputs_decoder_prob)
            batch_size, seq_len = input_ids.shape
            mask = (torch.rand(size=(batch_size, seq_len)) > (self.drop_inputs_decoder_prob)).float().unsqueeze(2).to(input_ids.device)
            inputs_embeds = inputs_embeds * mask
        """"
        torch.cuda.FloatTensor(10, 10).uniform_() > 0.8
        a  = torch.randn((2, 3, 3))
        mask = torch.randint(low=0, high=2, size=(2, 3)).unsqueeze(2)
        print(a)
        print(mask)
        print(a * mask)
        """

        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + token_type_embeddings
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

    def create_position_ids_from_inputs_embeds(self, inputs_embeds):
        """
        We are provided embeddings directly. We cannot infer which are padded so just generate sequential position ids.
        Args:
            inputs_embeds: torch.Tensor
        Returns: torch.Tensor
        """
        input_shape = inputs_embeds.size()[:-1]
        sequence_length = input_shape[1]

        position_ids = torch.arange(
            self.padding_idx + 1, sequence_length + self.padding_idx + 1, dtype=torch.long, device=inputs_embeds.device
        )
        return position_ids.unsqueeze(0).expand(input_shape)


# Copied from transformers.models.bert.modeling_bert.BertSelfAttention with Bert->Roberta
class VaeDecoderRobertaSelfAttention(nn.Module):
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
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")

        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = nn.Embedding(2 * config.max_position_embeddings - 1, self.attention_head_size)

        self.is_decoder = config.is_decoder

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
            self,
            hidden_states,
            attention_mask=None,
            latent_layer_memory_i=None,  # >>>> Claartje code
            latent_layer_cross_i=None,  # >>>> Claartje code
            gating=False, # >>>> Claartje code
            head_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            past_key_value=None,
            output_attentions=False,
    ):

        GATING = gating

        """

        Shapes:
            hidden_states:          [Batch, Seq_len, Hidden_size]
            mixed_query_layer:      [Batch, Seq_len, N_heads x Head_size], which is [Batch, Seq_len, Hidden_size]
            mixed_key_layer:        ,,
            mixed_value_layer:      ,,
            latent_layer_memory_i:  [Batch, 1, Hidden_size]
            latent_layer_cross_i:   ,,
            query_layer:            [Batch, N_heads, Seq_len, Head_size]
            key_layer:              ,,
            value_layer:            ,,
            attention_probs:        [Batch, ]
        """

        mixed_query_layer = self.query(hidden_states)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        is_cross_attention = encoder_hidden_states is not None

        # Cross attention with cache
        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_layer = past_key_value[0]
            value_layer = past_key_value[1]
            attention_mask = encoder_attention_mask

        # Cross attention no cache
        elif is_cross_attention:
            key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
            value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
            attention_mask = encoder_attention_mask

        # >>>>>>> Claartje code
        # Cross latent attention or Latent Decoder attention
        elif latent_layer_cross_i is not None:
            # latent_layer_cross_i should be of shape (batch x 1 x hidden_size)
            mixed_key_layer = self.key(hidden_states)
            mixed_value_layer = self.value(hidden_states)

            # This should now be: [Batch, N_heads, Seq_len, Head_size]
            key_layer = self.transpose_for_scores(mixed_key_layer)
            value_layer = self.transpose_for_scores(mixed_value_layer)

            # This should now be: [Batch, N_heads, 1, Head_size]
            key_latent_layer = self.transpose_for_scores(latent_layer_cross_i)
            value_latent_layer = self.transpose_for_scores(latent_layer_cross_i)

            # This should now be: [Batch, N_heads, Seq_len + 1, Head_size]
            key_layer = torch.cat([key_latent_layer, key_layer], dim=2)
            value_layer = torch.cat([value_latent_layer, value_layer], dim=2)

        # Normal attention with cache (use_cache is True and time_step > 0)
        elif past_key_value is not None:
            mixed_key_layer = self.key(hidden_states)
            mixed_value_layer = self.value(hidden_states)

            key_layer = self.transpose_for_scores(mixed_key_layer)
            value_layer = self.transpose_for_scores(mixed_value_layer)

            # First cat the cache
            key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
            value_layer = torch.cat([past_key_value[1], value_layer], dim=2)

            # Then the latent (if given)
            if latent_layer_memory_i is not None:
                key_latent_layer = self.transpose_for_scores(latent_layer_memory_i)
                value_latent_layer = self.transpose_for_scores(latent_layer_memory_i)
                key_layer = torch.cat([key_latent_layer, key_layer], dim=2)

            if GATING is False:
                # print("key layer with latent and cache", key_layer.shape)
                value_layer = torch.cat([value_latent_layer, value_layer], dim=2)

        # Normal attention no cache or time_step is 0 with using cache
        else:
            mixed_key_layer = self.key(hidden_states)
            mixed_value_layer = self.value(hidden_states)

            # >>>> Claartje code
            if latent_layer_memory_i is not None and GATING is False:
                mixed_key_layer = torch.cat((latent_layer_memory_i, mixed_key_layer), dim=1)
                mixed_value_layer = torch.cat((latent_layer_memory_i, mixed_value_layer), dim=1)

            elif latent_layer_memory_i is not None and GATING is True:
                mixed_key_layer = torch.cat((latent_layer_memory_i, mixed_key_layer), dim=1)

            key_layer = self.transpose_for_scores(mixed_key_layer)
            value_layer = self.transpose_for_scores(mixed_value_layer)
            # <<<< Claartje code

        query_layer = self.transpose_for_scores(mixed_query_layer)

        #if self.is_decoder: Claartje: for me this is always the case, might as well remove the if
        # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
        # Further calls to cross_attention layer can then reuse all cross-attention
        # key/value_states (first "if" case)
        # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
        # all previous decoder key/value_states. Further calls to uni-directional self-attention
        # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
        # if encoder bi-directional self-attention `past_key_value` is always `None`
        present_key_value = (key_layer, value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            seq_length = hidden_states.size()[1]
            position_ids_l = torch.arange(seq_length, dtype=torch.long, device=hidden_states.device).view(-1, 1)
            position_ids_r = torch.arange(seq_length, dtype=torch.long, device=hidden_states.device).view(1, -1)
            distance = position_ids_l - position_ids_r
            positional_embedding = self.distance_embedding(distance + self.max_position_embeddings - 1)
            positional_embedding = positional_embedding.to(dtype=query_layer.dtype)  # fp16 compatibility

            if self.position_embedding_type == "relative_key":
                relative_position_scores = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores
            elif self.position_embedding_type == "relative_key_query":
                relative_position_scores_query = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                relative_position_scores_key = torch.einsum("bhrd,lrd->bhlr", key_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores_query + relative_position_scores_key

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        if attention_mask is not None:
            gating = False
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
                # print("-"*80)

            elif latent_layer_cross_i is not None:
                batch_size, seq_len, _ = hidden_states.shape

                # First make a mask that has zero on the diagonal and -1000.0 off diagonal of shape seq_len x seq_len
                # so that all the hidden states may attend to themselves (0.0 mask)
                square_mask = torch.ones(seq_len, seq_len) * -10000.0
                square_mask = square_mask * (1.0 - torch.eye(seq_len, seq_len))
                # then add the extension for attention to the latent (which must be allowed) -> 0.0 mask

                zero_vec_extension = torch.zeros((seq_len, 1))

                # add the extension for the latent
                cross_attention_mask = torch.cat((zero_vec_extension, square_mask), dim=1)

                # then replicate to match the correct shape for attention scores
                # final shape should be: [Batch, 1, Q_dim, K_dim] = [Batch, 1, Seq_len, Seq_len+1] (+1 for the latent)
                cross_attention_mask = cross_attention_mask.unsqueeze(0).unsqueeze(0)
                cross_attention_mask = cross_attention_mask.repeat(batch_size, 1, 1, 1).type_as(attention_mask)

                attention_scores = attention_scores + cross_attention_mask

            else:
                # Apply the attention mask is (precomputed for all layers in RobertaModel forward() function)
                attention_scores = attention_scores + attention_mask
            # <<<<<< End Claartje code

        """
        Description of shapes and the gating mechanism:
        
        If gating is used:
            - query vectors are formed from hidden states
            - key vectors are formed from latent + hidden states
            - value vevtors are formed from hidden states
            - attention matrix will thus be of shape: [batch, n_heads, seq_len query, seq_len key]
                -> which is thus: [batch, n_heads, seq_len, seq_len+1]
            - then the scores (not probabilities) from hidden states to latent are cut off forming a vector which records
            the dot product between the query vectors and the latent key
            - then the resulting scores (normal attention matrix) is transformed to probabilities:
                -> this has the normal shape: [batch, n_heads, seq_len, seq_len]
            - then the normal contextual layer is build
            - then the latent is added by a gate to all contextual vectors according to the cut-off attention_scores:
                - sigmoid is applied to those scores which define the gating weights
                - then the contextual layer will be linearly combined with the latent_value_layer:
                (1-weight) * context_layer + w * latent_value layer
                    -> where the weights are specific for sequence positions and heads
        """


        if GATING is True:
            latent_attention_scores = attention_scores[:, :, :, 0]
            latent_attention_probs = nn.Sigmoid()(latent_attention_scores).unsqueeze(-1)
            latent_value_layer = self.transpose_for_scores(latent_layer_memory_i)
            attention_scores = attention_scores[:, :, :, 1:]

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        if GATING is True:
            # Linear combination between the contextual layer and the latent_value_layer
            context_layer = ((1.0 - latent_attention_probs) * context_layer) + (latent_attention_probs * latent_value_layer)


        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        # cat back on to return
        attention_probs = torch.cat([latent_attention_probs, attention_probs], dim=-1)

        # Claartje:
        # on cache: only return if actual self-attention, if cross attention for latent, no cache needs to be returned
        out_dict = {
            "context_layer": context_layer,
            "attention_probs": attention_probs if output_attentions else None,
            "present_key_value": present_key_value if latent_layer_cross_i is None else None,
        }

        return out_dict


# Copied from transformers.models.bert.modeling_bert.BertSelfOutput
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


# Copied from transformers.models.bert.modeling_bert.BertAttention with Bert->Roberta
class VaeDecoderRobertaAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self = VaeDecoderRobertaSelfAttention(config)
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
            latent_layer_memory_i=None,  # >>>> Claartje code
            latent_layer_cross_i=None,  # >>>> Claartje code
            gating=False,
            head_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            past_key_value=None,
            output_attentions=False,
    ):


        self_outputs = self.self(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            latent_layer_memory_i=latent_layer_memory_i,  # >>>> Claartje code
            latent_layer_cross_i=latent_layer_cross_i,  # >>>> Claartje code
            gating=gating,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
        )
        attention_output = self.output(self_outputs['context_layer'], hidden_states) # residual

        out_dict = {
            'attention_output': attention_output, # context layer -> attention output
            'attention_probs': self_outputs['attention_probs'], # forward whatever comes from selfattention module
            'present_key_value': self_outputs['present_key_value'] # forward whatever comes from selfattention module
        }

        return out_dict


# Copied from transformers.models.bert.modeling_bert.BertIntermediate
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


# Copied from transformers.models.bert.modeling_bert.BertOutput
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


# Copied from transformers.models.bert.modeling_bert.BertLayer with Bert->Roberta
class VaeDecoderRobertaLayer(nn.Module):
    def __init__(self, config, add_latent_via_cross_attention=False):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = VaeDecoderRobertaAttention(config)
        self.is_decoder = config.is_decoder
        self.add_cross_attention = config.add_cross_attention
        if self.add_cross_attention:
            assert self.is_decoder, f"{self} should be used as a decoder model if cross attention is added"
            self.crossattention = VaeDecoderRobertaAttention(config)

        if add_latent_via_cross_attention:
            self.crossattention = VaeDecoderRobertaAttention(config)

        self.intermediate = RobertaIntermediate(config)
        self.output = RobertaOutput(config)

    def forward(
            self,
            hidden_states,
            attention_mask=None,
            latent_memory_i=None,
            latent_cross_i=None,
            gating=False,
            head_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            past_key_value=None,
            output_attentions=False,
    ):

        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        self_attention_out_dict = self.attention(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            latent_layer_memory_i=latent_memory_i,  # <<<< Claartje code
            latent_layer_cross_i=None,  # <<< just to explicify that this is not cross attention
            gating=gating,
            encoder_attention_mask=None,  # <<< just to explicify that this is not cross attention
            encoder_hidden_states=None,  # <<< just to explicify that this is not cross attention
            head_mask=head_mask,
            output_attentions=output_attentions,
            past_key_value=self_attn_past_key_value,
        )


        attention_output = self_attention_out_dict['attention_output'] # attention_output (new hidden states)

        # Here we use the cross attention module to add information from the latent
        cross_attention_probs = None
        if latent_cross_i is not None:
            cross_attention_out_dict = self.crossattention(
                hidden_states=attention_output,
                attention_mask=attention_mask,
                head_mask=head_mask,  # head mask: not needed
                latent_layer_memory_i=None,
                latent_layer_cross_i=latent_cross_i,
                encoder_hidden_states=None,
                encoder_attention_mask=None,
                past_key_value=None,
                output_attentions=output_attentions  # cross_attn_past_key_value no cache needed
            )
            cross_attention_probs = cross_attention_out_dict["attention_probs"]
            attention_output = cross_attention_out_dict["attention_output"] # overwrite attention_output if cross attention is applied

        # Feed forward output
        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )

        out_dict = {
            "layer_output": layer_output,
            "self_attention_probs": self_attention_out_dict["attention_probs"] if output_attentions else None,
            "cross_attention_probs": cross_attention_probs if output_attentions else None,
            "present_key_value": self_attention_out_dict["present_key_value"],
        }

        return out_dict

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


# Copied from transformers.models.bert.modeling_bert.BertEncoder with Bert->Roberta
class VaeDecoderRobertaEncoder(nn.Module):
    def __init__(self, config, add_latent_via_cross_attention=False):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([VaeDecoderRobertaLayer(config,
                                                           add_latent_via_cross_attention=add_latent_via_cross_attention) \
                                    for _ in range(config.num_hidden_layers)])

    def forward(
            self,
            hidden_states,
            attention_mask=None,
            latent_layer_memory=None,  # <<<< Claartje code
            head_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            latent_layer_cross=None,  # <<<< Claartje code
            gating=False,
            past_key_values=None,
            use_cache=None,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions else None

        next_decoder_cache = () if use_cache else None
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None

            # <<<< Claartje code
            if latent_layer_memory is not None:
                latent_layer_memory_i = latent_layer_memory[i]
            else:
                latent_layer_memory_i = None
            # >>>> Claartje code

            if latent_layer_cross is not None:
                latent_layer_cross_i = latent_layer_cross[i]
            else:
                latent_layer_cross_i = None

            # TODO: mind you, gradient_checkpointing is broken
            if getattr(self.config, "gradient_checkpointing", False) and self.training:

                if use_cache:
                    logger.warn(
                        "`use_cache=True` is incompatible with `config.gradient_checkpointing=True`. Setting "
                        "`use_cache=False`..."
                    )
                    use_cache = False

                if latent_layer_memory_i:
                    print("Claartje: Using gradient checkpointing in combination with memory mechanism. Not sure"
                          "if that works...")

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, past_key_value, output_attentions)

                    return custom_forward

                layer_out_dict = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    attention_mask,
                    latent_layer_memory_i,  # Claartje: not sure if this works with gradient checkpointing
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                )
            else:
                layer_out_dict = layer_module(
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    latent_memory_i=latent_layer_memory_i,  # <<<< Claartje code
                    latent_cross_i=latent_layer_cross_i, # <<<<< Claartje code
                    gating=gating,
                    head_mask=layer_head_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                )
                # dict contains the following keys "layer_output", "self_attention_probs",
                # "cross_attention_probs", "present_key_value"

            hidden_states = layer_out_dict["layer_output"]
            if use_cache:
                next_decoder_cache += (layer_out_dict["present_key_value"],)

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_out_dict["self_attention_probs"],)

                if layer_out_dict["cross_attention_probs"] is not None:
                    all_cross_attentions = all_cross_attentions + (layer_out_dict["cross_attention_probs"],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_decoder_cache,
                    all_hidden_states,
                    all_self_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )


# Copied from transformers.models.bert.modeling_bert.BertPooler
class RobertaPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class RobertaPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = RobertaConfig
    base_model_prefix = "roberta"

    # Copied from transformers.models.bert.modeling_bert.BertPreTrainedModel._init_weights
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


class VaeDecoderRobertaModel(RobertaPreTrainedModel):
    """
    The model can behave as an encoder (with only self-attention) as well as a decoder, in which case a layer of
    cross-attention is added between the self-attention layers, following the architecture described in `Attention is
    all you need`_ by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz
    Kaiser and Illia Polosukhin.
    To behave as an decoder the model needs to be initialized with the :obj:`is_decoder` argument of the configuration
    set to :obj:`True`. To be used in a Seq2Seq model, the model needs to initialized with both :obj:`is_decoder`
    argument and :obj:`add_cross_attention` set to :obj:`True`; an :obj:`encoder_hidden_states` is then expected as an
    input to the forward pass.
    .. _`Attention is all you need`: https://arxiv.org/abs/1706.03762
    """

    _keys_to_ignore_on_load_missing = [r"position_ids"]

    # Copied from transformers.models.bert.modeling_bert.BertModel.__init__ with Bert->Roberta
    def __init__(self, config, add_pooling_layer=True, add_latent_via_cross_attention=False):
        super().__init__(config)
        self.config = config

        self.embeddings = RobertaEmbeddings(config)
        self.encoder = VaeDecoderRobertaEncoder(config, add_latent_via_cross_attention=add_latent_via_cross_attention)

        # self.pooler = VAE_Decoder_RobertaPooler(config) if add_pooling_layer else None  <<<< disabled by Claartje
        self.pooler = None  # >>>> Claartje code

        self.init_weights()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    # Copied from transformers.models.bert.modeling_bert.BertModel.forward
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
            latent_layer_cross=None,  # >>>>>> Claartje code
            gating=False,
            past_key_values=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        r"""
        encoder_hidden_states  (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
            the model is configured as a decoder.
        encoder_attention_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
            the cross-attention if the model is configured as a decoder. Mask values selected in ``[0, 1]``: ``1`` for
            tokens that are NOT MASKED, ``0`` for MASKED tokens.
        past_key_values (:obj:`tuple(tuple(torch.FloatTensor))` of length :obj:`config.n_layers` with each tuple having 4 tensors of shape :obj:`(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
            Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.
            If :obj:`past_key_values` are used, the user can optionally input only the last :obj:`decoder_input_ids`
            (those that don't have their past key value states given to this model) of shape :obj:`(batch_size, 1)`
            instead of all :obj:`decoder_input_ids` of shape :obj:`(batch_size, sequence_length)`.
        use_cache (:obj:`bool`, `optional`):
            If set to :obj:`True`, :obj:`past_key_values` key value states are returned and can be used to speed up
            decoding (see :obj:`past_key_values`).
        """

        #print("input_ids.shape", input_ids.shape)

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if not self.config.is_decoder:
            use_cache = False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            batch_size, seq_length = input_shape
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size, seq_length = input_shape
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length + past_key_values_length)), device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # Claartje: HERE THE CAUSAL / LEFT-TO-RIGHT / FORWARD LOOKING MASK IS MADE!
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
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
        )

        # >>>> Claartje code
        # If latent is added via embeddings it is simply summed with the input embeddings / initial hidden states
        if latent_embedding is not None:
            embedding_output += latent_embedding.unsqueeze(1)
        # <<<< Claartje code

        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            latent_layer_memory=latent_layer_memory,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            latent_layer_cross=latent_layer_cross,
            gating=gating,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]


        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )


class VaeDecoderRobertaForCausalLM(RobertaPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids", r"lm_head.decoder.bias"]
    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config):
        super().__init__(config)

        self.config = config

        # >>>> Claartje code
        config.is_decoder = True
        config.add_cross_attention = False
        config.return_dict = True
        # <<<< Claartje code

        # add the cross attention modules, potentially delete again later
        self.roberta = VaeDecoderRobertaModel(config, add_pooling_layer=False,
                                              add_latent_via_cross_attention=True)
        self.lm_head = RobertaLMHead(config)

        # >>>> Claartje code
        self.latent_to_memory_projection = None
        self.latent_to_embedding_projection = None
        # <<<< End Claartje code

        self.init_weights()

    def add_latent_projection_layers(self, latent_size, hidden_size, n_layers,
                                     add_latent_via_memory=True, add_latent_via_embeddings=True):
        # >>>> Claartje code
        # TODO: In the Optimus they have bias=False, but can't find good reason for that?
        if add_latent_via_memory:
            self.latent_to_memory_projection = nn.Linear(latent_size, hidden_size * n_layers)
            self.latent_to_memory_projection.weight.data.normal_(mean=0.0, std=self.config.initializer_range)

        if add_latent_via_embeddings:
            self.latent_to_embedding_projection = nn.Linear(latent_size, hidden_size)
            self.latent_to_embedding_projection.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        # <<<< End Claartje code

    def get_output_embeddings(self):
        return self.lm_head.decoder

    def set_output_embeddings(self, new_embeddings):
        self.lm_head.decoder = new_embeddings

    def forward(
            self,
            input_ids=None,
            attention_mask=None,

            labels=None,

            past_key_values=None,
            use_cache=None,

            output_attentions=None,
            output_hidden_states=None,

            # Claartje: I do not use these things now
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            encoder_hidden_states=None,  # >>>>>> Claartje: this should never enter
            encoder_attention_mask=None,  # >>>>>> Claartje: this should never enter

            # >>>> Claartje code
            return_attention_probs=None,
            return_attention_to_latent=None,

            latent_to_decoder_output=None,
            gating=False,
            return_predictions=False,
            return_probabilities=False,
            return_logits=False,

            return_exact_match=False,
            return_cross_entropy=True,
            return_reconstruction_loss=True,

            reduce_seq_dim_exact_match="mean",
            reduce_batch_dim_exact_match="mean",
            reduce_seq_dim_ce="sum",
            reduce_batch_dim_ce="mean",

            return_hidden_states=None,
            return_last_hidden_state=False,
            return_output_embeddings=False,

            return_log_probs=False,

            nucleus_sampling=False,
            top_k=0,
            top_p=0.9,
            # <<<< Claartje code

    ):

        # >>>> Claartje code
        # Latent to decoder

        add_latent_via_cross, add_latent_via_mem = False, False
        if latent_to_decoder_output is not None:
            latent_layer_memory = latent_to_decoder_output['latent_to_memory']
            latent_embedding = latent_to_decoder_output['latent_to_embeddings']
            latent_layer_cross = latent_to_decoder_output['latent_to_cross']
            if latent_layer_cross is not None:
                add_latent_via_cross = True
            # if latent_embedding is not None:
            #     add_latent_via_embed = True
            if latent_layer_memory is not None:
                add_latent_via_mem = True
        else:
            latent_layer_memory, latent_embedding, latent_layer_cross = None, None, None
        # <<<< Claartje code

        output_attentions = True if (return_attention_to_latent or return_attention_probs) else False

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,  # >>>>>> Claartje: abusing these for my attention to latent
            encoder_attention_mask=encoder_attention_mask,  # >>>>>> Claartje: abusing these for my attention to latent
            latent_layer_cross=latent_layer_cross,
            gating=gating,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,  # >>>>>> Claartje code
            latent_layer_memory=latent_layer_memory,  # >>>>>> Claartje code
            latent_embedding=latent_embedding,  # >>>>>> Claartje code
        )

        # >>>> Claartje code
        # Get outputs and shift the predictions and labels so they align
        logits, output_embeddings = self.lm_head(outputs.last_hidden_state)
        # we don't care about the last prediction, because that is the prediction at </s>
        logits = logits[:, :-1, :].contiguous()

        cross_entropy, predictions, exact_match, cross_entropy_per_word, reconstruction_loss, \
        cross_entropy, probs, attention_probs, attention_to_latent, hidden_states, log_probs = None, None, \
        None, None, None, None, None, None, None, None, None

        # Reformat labels and create a mask (based on pad token id = 1)
        if labels is not None:
            labels = labels[:, 1:].contiguous()  # skip <s> token
            # pad token is int 1
            label_mask = (labels != 1).float()
            labels = labels.reshape(-1) # make into one big vector (no seq dimension)

            # -> from now on the sequence length is thus 63 instead of 64
            # logits are [<w1>' <w2>' ... <w62> </s>'] and labels [<w1> <w2> ... <w62> </s>]

        # Attention probabilities
        # This is a tuple of 12 layers of size [batch, 12, seq_len, seq_len + 1]
        attention_probs = outputs.attentions if (return_attention_probs or return_attention_to_latent) else None
        attention_to_latent = None

        if return_attention_probs:
            attention_probs = {
                "self_attention_probs": outputs.attentions,
                "cross_attention_probs": outputs.cross_attentions
            }
        else:
            attention_probs = None

        self_attention_to_latent, cross_attention_to_latent = None, None
        if return_attention_to_latent:
            if add_latent_via_mem:
                # stack all layers into one big tensor
                # dimensions are: layers, batch, n_heads, query_dim, key_value_dim
                attention_probs = torch.stack(outputs.attentions)

                # batch, n_heads, n_layers, seq_len_query, seq_len_val
                attention_probs = attention_probs.permute(1, 2, 0, 3, 4)

                # Get attention to latent only, for all tokens except end token
                self_attention_to_latent = attention_probs[:, :, :, :-1, 0]

            elif add_latent_via_cross:
                cross_atts = torch.stack(outputs.cross_attentions, dim=1)
                # to key = 0 (latent), and without the last query (eos token)
                cross_attention_to_latent = cross_atts[:, :, :, :-1, 0]

        # Get rid of the sequence dimension (merge into batch dimension)
        batch_size, seq_len, vocab_size = logits.shape
        logits = logits.reshape(-1, vocab_size)

        # ONLY CROSS ENTROPY (at train time)
        if (return_cross_entropy or return_reconstruction_loss) and not (
                return_predictions or return_probabilities or return_exact_match or nucleus_sampling or return_log_probs):
            # cross entropy = log_softmax + nll_loss, but use cross_entropy function for stability
            cross_entropy = torch.nn.functional.cross_entropy(logits, labels, reduction='none')
            cross_entropy = cross_entropy.reshape(batch_size, seq_len)  # bring back the sequence dimension
            cross_entropy = cross_entropy * label_mask
            reconstruction_loss = cross_entropy.sum(dim=-1).mean()

        # ALSO PREDICT (at inference or validation time)
        elif return_predictions or return_probabilities or return_exact_match or nucleus_sampling or return_log_probs:
            if return_cross_entropy or return_reconstruction_loss:
                # use this as it is numerically stable, not efficient together with softmax
                # do cross entropy with normal logits, not filtered logits
                cross_entropy = torch.nn.functional.cross_entropy(logits, labels, reduction='none')
                cross_entropy = cross_entropy.reshape(batch_size, seq_len)  # bring back the sequence dimension
                reconstruction_loss = (cross_entropy * label_mask).sum(dim=-1).mean()

            if nucleus_sampling:
                # logits are overwritten now
                logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)

            probs = torch.nn.functional.softmax(logits, dim=-1)

            if return_predictions or return_exact_match or nucleus_sampling or return_log_probs:
                if nucleus_sampling:
                    predictions = torch.multinomial(probs, num_samples=1)
                else:
                    predictions = probs.argmax(-1)

                if return_log_probs:
                    log_probs = torch.distributions.Categorical(probs=probs).log_prob(predictions.squeeze(1))
                    log_probs = log_probs.reshape(batch_size, seq_len)

                if return_exact_match:
                    exact_match = (labels == predictions.squeeze()).float()
                    exact_match = exact_match.reshape(batch_size, seq_len)

        # REDUCE CORRECTLY
        if cross_entropy is not None:
            cross_entropy = cross_entropy * label_mask
            cross_entropy_per_word = (cross_entropy.sum(dim=-1) / label_mask.sum(dim=-1)).mean()
            cross_entropy = self.reduce_correct(cross_entropy, reduce_seq_dim_ce, -1, label_mask)  # seq dim
            cross_entropy = self.reduce_correct(cross_entropy, reduce_batch_dim_ce, 0,
                                                label_mask)  # batch dim <- always mean

        if exact_match is not None:
            exact_match = exact_match * label_mask
            exact_match = self.reduce_correct(exact_match, reduce_seq_dim_exact_match, -1, label_mask)  # seq dim
            exact_match = self.reduce_correct(exact_match, reduce_batch_dim_exact_match, 0, label_mask)  # batch dim

        if return_hidden_states:
            hidden_states = torch.stack(outputs.hidden_states, dim=0)
        else:
            hidden_states = None

        if return_probabilities:
            probs = probs.reshape((batch_size, seq_len, -1))

        if return_predictions:
            predictions = predictions.reshape((batch_size, seq_len))

        if logits is not None:
            logits = logits.reshape((batch_size, seq_len, vocab_size))

        # print("use cache in decoder_roberta_new", use_cache)
        # print("len(outputs.past_key_values)", len(outputs.past_key_values))
        # print("len(outputs.past_key_values[0])", len(outputs.past_key_values[0]))
        # print("outputs.past_key_values[0][0].shape", outputs.past_key_values[0][0].shape)

        # -----------------------------------------------------
        # cross_entropy VS log_probs
        #
        # Cross entropy has the negative log likelihood of the inputs (so reconstruction neg. log likelihood)
        # Log_probs has the log likelihood of the predictions (so the log prob assigned to the prediction)
        #   where the prediction might be sampled or taken to be argmax, depending on nucleus_sampling arg.
        # -----------------------------------------------------

        return_dict = {
            "cross_entropy": cross_entropy,  # CE reduced following the input settings
            "cross_entropy_per_word": cross_entropy_per_word,  # CE per word (mean over seq and batch)
            "predictions": predictions if return_predictions else None,
            "exact_match": exact_match if return_exact_match else None,
            "attention_probs": attention_probs if return_attention_probs else None,
            "self_attention_to_latent": self_attention_to_latent if (return_attention_to_latent and add_latent_via_mem) else None,
            "cross_attention_to_latent": cross_attention_to_latent if (return_attention_to_latent and add_latent_via_cross) else None,
            "hidden_states": hidden_states if return_hidden_states else None,
            "probabilities": probs if return_probabilities else None,
            "last_hidden_state": outputs.last_hidden_state if return_last_hidden_state else None,
            "logits": logits if return_logits else None,
            "output_embeddings": output_embeddings if return_output_embeddings else None,
            "past_key_values": outputs.past_key_values if use_cache else None,
            "cross_attentions": outputs.cross_attentions if return_attention_probs else None,
            "log_probs": log_probs if return_log_probs else None,
            "reconstruction_loss": reconstruction_loss if return_reconstruction_loss else None
        }
        return return_dict
        # <<<< Claartje code

    # >>>> Claartje code
    @staticmethod
    def reduce_correct(some_tensor, reduction_type, reduction_dim, label_mask):
        # Here we assume the label mask has already been applied to 'some_tensor'

        # Average the sequence dimension: summed loss / sequence lengths
        if reduction_type == "mean" and reduction_dim == -1:
            label_mask = label_mask.sum(dim=reduction_dim)
            some_tensor = some_tensor.sum(dim=reduction_dim)
            some_tensor = some_tensor / label_mask

        # Average batch dimension: just take the average over the batch dim
        elif reduction_type == "mean" and reduction_dim == 0:
            some_tensor = some_tensor.mean(dim=reduction_dim)

        # Sum over either sequence or batch dimension, just sum
        elif reduction_type == "sum":
            some_tensor = some_tensor.sum(dim=reduction_dim)

        return some_tensor
    # <<<< Claartje code

    def prepare_inputs_for_generation(self, input_ids, past=None, attention_mask=None, **model_kwargs):
        input_shape = input_ids.shape
        # if model is used as a decoder in encoder-decoder model, the decoder attention mask is created on the fly
        if attention_mask is None:
            attention_mask = input_ids.new_ones(input_shape)

        # cut decoder_input_ids if past is used
        if past is not None:
            input_ids = input_ids[:, -1:]

        return {"input_ids": input_ids, "attention_mask": attention_mask, "past_key_values": past}

    def _reorder_cache(self, past, beam_idx):
        reordered_past = ()
        for layer_past in past:
            reordered_past += (tuple(past_state.index_select(0, beam_idx) for past_state in layer_past),)
        return reordered_past


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
        output_embedding = self.layer_norm(x)

        # project back to size of vocabulary with bias
        logits = self.decoder(output_embedding)

        return logits, output_embedding


def create_position_ids_from_input_ids(input_ids, padding_idx, past_key_values_length=0):
    """
    Replace non-padding symbols with their position numbers. Position numbers begin at padding_idx+1. Padding symbols
    are ignored. This is modified from fairseq's `utils.make_positions`.
    Args:
        x: torch.Tensor x:
    Returns: torch.Tensor
    """
    # The series of casts and type-conversions here are carefully balanced to both work with ONNX export and XLA.
    mask = input_ids.ne(padding_idx).int()
    incremental_indices = (torch.cumsum(mask, dim=1).type_as(mask) + past_key_values_length) * mask
    return incremental_indices.long() + padding_idx
