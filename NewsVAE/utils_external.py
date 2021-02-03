# type: ignore
# Ignore type checking in this file as it contains copied code which throws a lot of errors I can't really fix
from typing import List
import torch.nn as nn

"""
This file contains functions from external sources, such as the Pytorch Library.
"""


# This function is copied from the PreTrainedModel definition :
# https://huggingface.co/transformers/_modules/transformers/modeling_utils.html#PreTrainedModel
def tie_weights(encoder, decoder, base_model_prefix):
    uninitialized_encoder_weights: List[str] = []

    # print("decoder.__class__, encoder.__class__", decoder.__class__, encoder.__class__)

    if decoder.__class__ != encoder.__class__:
        print(
            f"{decoder.__class__} and {encoder.__class__} are not equal. In this case make sure that all encoder "
            f"weights are correctly initialized. "
        )

    def tie_encoder_to_decoder_recursively(
            decoder_pointer: nn.Module,
            encoder_pointer: nn.Module,
            module_name: str,
            uninitialized_encoder_weights: List[str],
            depth=0,
    ):
        assert isinstance(decoder_pointer, nn.Module) and isinstance(
            encoder_pointer, nn.Module
        ), f"{decoder_pointer} and {encoder_pointer} have to be of type torch.nn.Module"
        if hasattr(decoder_pointer, "weight"):
            assert hasattr(encoder_pointer, "weight")
            encoder_pointer.weight = decoder_pointer.weight
            if hasattr(decoder_pointer, "bias"):
                assert hasattr(encoder_pointer, "bias")
                encoder_pointer.bias = decoder_pointer.bias
            return

        encoder_modules = encoder_pointer._modules
        decoder_modules = decoder_pointer._modules

        # print("Encoder modules", " ".join([n for n in encoder_modules.keys()]))
        # print("Decoder modules", " ".join([n for n in decoder_modules.keys()]))

        if len(decoder_modules) > 0:
            # print("len(decoder_modules)", len(decoder_modules))
            assert (
                    len(encoder_modules) > 0
            ), f"Encoder module {encoder_pointer} does not match decoder module {decoder_pointer}"

            all_encoder_weights = set([module_name + "/" + sub_name for sub_name in encoder_modules.keys()])
            encoder_layer_pos = 0
            for name, module in decoder_modules.items():
                if name.isdigit():
                    # print("name is digit", name)
                    encoder_name = str(int(name) + encoder_layer_pos)
                    decoder_name = name

                    # print("encoder_name", encoder_name)
                    # print("decoder_name", decoder_name)

                    if not isinstance(decoder_modules[decoder_name],
                                      type(encoder_modules[encoder_name])) and len(
                        encoder_modules
                    ) != len(decoder_modules):
                        # this can happen if the name corresponds to the position in a list module list of layers
                        # in this case the decoder has added a cross-attention that the encoder does not have
                        # thus skip this step and substract one layer pos from encoder
                        encoder_layer_pos -= 1
                        continue
                elif name not in encoder_modules:
                    continue
                elif depth > 500:
                    raise ValueError(
                        "Max depth of recursive function `tie_encoder_to_decoder` reached. It seems that there is a "
                        "circular dependency between two or more `nn.Modules` of your model. "
                    )
                else:
                    # print("else")
                    decoder_name = encoder_name = name

                    # print("decoder_name = encoder_name = name")
                    # print("decoder_name:", decoder_name)

                tie_encoder_to_decoder_recursively(
                    decoder_modules[decoder_name],
                    encoder_modules[encoder_name],
                    module_name + "/" + name,
                    uninitialized_encoder_weights,
                    depth=depth + 1,
                )
                all_encoder_weights.remove(module_name + "/" + encoder_name)

            uninitialized_encoder_weights += list(all_encoder_weights)

    # tie weights recursively
    tie_encoder_to_decoder_recursively(decoder, encoder, base_model_prefix, uninitialized_encoder_weights)

    if len(uninitialized_encoder_weights) > 0:
        print(
            f"The following encoder weights were not tied to the decoder {uninitialized_encoder_weights}"
        )
