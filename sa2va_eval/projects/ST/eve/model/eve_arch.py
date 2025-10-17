from abc import ABC, abstractmethod

import torch

from projects.ST.eve.constants import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                           DEFAULT_IMAGE_PATCH_TOKEN, IGNORE_INDEX,
                           IMAGE_TOKEN_INDEX)

from .multimodal_encoder.vision_tokenizer import VisionTokenizer


class EVEMetaModel:

    def __init__(self, config):
        super(EVEMetaModel, self).__init__(config)

        if hasattr(config, "mm_vision_tower"):
            self.vision_tower = VisionTokenizer(config.mm_hidden_size, 
                                                config.hidden_size,
                                                config.mm_vision_tower)

    def get_vision_tower(self):
        vision_tower = getattr(self, 'vision_tower', None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower

    def initialize_vision_modules(self, model_args, fsdp=None):
        vision_tower = model_args.vision_tower
        self.config.mm_vision_tower = vision_tower

        if self.get_vision_tower() is None:
            vision_tower = VisionTokenizer(model_args.vision_tower_hidden_size,
                                           self.config.hidden_size,
                                           model_args.vision_tower)
            if fsdp is not None and len(fsdp) > 0:
                self.vision_tower = [vision_tower]
            else:
                self.vision_tower = vision_tower
        else:
            if fsdp is not None and len(fsdp) > 0:
                vision_tower = self.vision_tower[0]
            else:
                vision_tower = self.vision_tower

        self.config.mm_hidden_size = vision_tower.hidden_size


class EVEMetaForCausalLM(ABC):

    @abstractmethod
    def get_model(self):
        pass

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()

    def get_transfer_tensor(self, tmp_tensor):
        return tmp_tensor.to(dtype=self.get_model().dtype, device=self.get_model().device)

    def prepare_inputs_labels_for_multimodal(
        self, input_ids, position_ids, attention_mask, past_key_values, labels, images, visual_prompt_indexes=None
    ):

        vp_ids = torch.Tensor(self.vp_token_idxs).to(input_ids.device)
        # print(vp_token_indices[vp_token_indices>=0])
        vp_embeds = self.model.embed_tokens(vp_ids.to(torch.int64))
        vp_embeds = torch.cat(
            [
                vp_embeds[:-1],
                torch.zeros(1, self.config.hidden_size).to(
                    input_ids.device
                ),  # add a dummy token
            ],
        )  # (n_patches + 1, hidden_size)

        vision_tower = self.get_vision_tower()
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            if past_key_values is not None and vision_tower is not None and images is not None and input_ids.shape[
                1] == 1:
                target_shape = past_key_values[-1][-1].shape[-2] + 1
                attention_mask = torch.cat((attention_mask, torch.ones(
                    (attention_mask.shape[0], target_shape - attention_mask.shape[1]),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device
                )), dim=1)
                position_ids = torch.sum(attention_mask, dim=1).unsqueeze(-1) - 1
            visual_token_mask = self.get_transfer_tensor(torch.zeros_like(input_ids).unsqueeze(-1))
            return input_ids, position_ids, attention_mask, past_key_values, None, labels, visual_token_mask

        image_features = self.get_vision_tower()(images)

        # Let's just add dummy tensors if they do not exist,
        # it is a headache to deal with None all the time.
        # But it is not ideal, and if you have a better idea,
        # please open an issue / submit a PR, thanks.
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        input_ids_temp = input_ids # points to the actual input_ids tensor

        # remove the padding using attention_mask -- TODO: double check
        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in
                     zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

        # -- TODO: better implementation?
        # replace IMAGE_TOKEN_INDEX(-200) with 0 to be compatible with repetition penalty
        input_ids_temp[input_ids_temp == IMAGE_TOKEN_INDEX] = 0

        new_labels = []
        new_input_embeds = []
        visual_token_mask = []
        cur_image_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            # print(cur_input_ids, visual_prompt_indexes)
            if num_images == 0:
                cur_image_features = image_features[cur_image_idx]
                if visual_prompt_indexes is not None:
                    assert len(visual_prompt_indexes[cur_image_idx]) == 0, visual_prompt_indexes[cur_image_idx]
                cur_image_features = cur_image_features + vp_embeds[-1:].to(cur_image_features.dtype)
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                visual_token_mask.append(self.get_transfer_tensor(torch.zeros_like(cur_input_ids).unsqueeze(-1)))
                continue

            image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [
                cur_input_ids.shape[0]]
            cur_input_ids_noim = []
            cur_labels = labels[batch_idx]
            cur_labels_noim = []
            for i in range(len(image_token_indices) - 1):
                cur_input_ids_noim.append(cur_input_ids[image_token_indices[i] + 1:image_token_indices[i + 1]])
                cur_labels_noim.append(cur_labels[image_token_indices[i] + 1:image_token_indices[i + 1]])
            split_sizes = [x.shape[0] for x in cur_labels_noim]
            cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_noim))
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
            cur_new_input_embeds = []
            cur_new_labels = []
            cur_visual_mask = []
            for i in range(num_images + 1):
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                cur_new_labels.append(cur_labels_noim[i])
                cur_visual_mask.append(
                    self.get_transfer_tensor(torch.zeros_like(cur_labels_noim[i]).unsqueeze(-1)))
                if i < num_images:
                    cur_image_features = image_features[cur_image_idx]
                    if self.use_vp and visual_prompt_indexes is not None:
                        _vp_embed = vp_embeds[visual_prompt_indexes[cur_image_idx]]
                    else:
                        _vp_embed = vp_embeds[-1:]
                    cur_image_features = cur_image_features + _vp_embed.to(cur_image_features.dtype)
                    cur_image_idx += 1
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_labels.append(
                        torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=cur_labels.device,
                                   dtype=cur_labels.dtype))
                    cur_visual_mask.append(
                        self.get_transfer_tensor(torch.ones(cur_image_features.shape[0], 1)))

            new_input_embeds.append(torch.cat(cur_new_input_embeds))
            new_labels.append(torch.cat(cur_new_labels))
            visual_token_mask.append(torch.cat(cur_visual_mask))
        
        # Truncate sequences to max length as image embeddings can make the sequence longer
        tokenizer_model_max_length = getattr(self.config, 'tokenizer_model_max_length', None)
        if tokenizer_model_max_length is not None:
            new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
            new_labels = [x[:tokenizer_model_max_length] for x in new_labels]
            visual_token_mask = [x[:tokenizer_model_max_length] for x in visual_token_mask]

        # Combine them
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype,
                                       device=new_labels[0].device)
        visual_token_mask_padded = self.get_transfer_tensor(torch.zeros_like(new_labels_padded).unsqueeze(-1))
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)

        for i, (cur_new_embed, cur_new_label, cur_new_mask) in enumerate(zip(new_input_embeds, new_labels, visual_token_mask)):
            cur_len = cur_new_embed.shape[0]
            new_input_embeds_padded.append(torch.cat((
                cur_new_embed,
                torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype,
                            device=cur_new_embed.device)
            ), dim=0))
            if cur_len > 0:
                new_labels_padded[i, :cur_len] = cur_new_label
                visual_token_mask_padded[i, :cur_len] = cur_new_mask
                attention_mask[i, :cur_len] = True
                position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype,
                                                            device=position_ids.device)

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None

        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels, visual_token_mask_padded
