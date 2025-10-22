from collections import OrderedDict
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from transformers import GPT2Model, ElectraModel, AutoModel
from transformers.models.electra.modeling_electra import ElectraEmbeddings, ElectraEncoder  
from transformers.models.gpt2.modeling_gpt2 import *

# class Baseline(nn.Module):
#     def __init__(self,
#                  model_link='beomi/KcELECTRA-base-v2022',
#                  class_num=2):
#         super(Baseline, self).__init__()
#         self.electra = AutoModel.from_pretrained(model_link)
        
#         self.classifier = nn.Sequential(OrderedDict([
#             ('dense',nn.Linear(768, 768)),
#             ('dropout', nn.Dropout(0.1)),
#             ('out_proj', nn.Linear(768, 2)),
#         ]))
        
#     def encode(self, input_ids, att_mask, token_type_ids):
#         output = self.electra(input_ids, att_mask, token_type_ids)
#         last_hidden_state = output.last_hidden_state
        
#         # cls = torch.mean(last_hidden_state, dim=1)
#         cls = last_hidden_state[:, 0, :]
#         return cls
        
#     def forward(self, input_ids, att_mask, token_type_ids):
#         output = self.electra(input_ids, att_mask, token_type_ids)
#         last_hidden_state = output.last_hidden_state
        
#         # cls = torch.mean(last_hidden_state, dim=1)
#         cls = last_hidden_state[:, 0, :]
#         logit = self.classifier(cls)
#         return logit
    
class Baseline(ElectraModel):
    def __init__(self, config):
        super(Baseline, self).__init__(config)
        self.config = config
        
        self.embeddings = ElectraEmbeddings(config)
        self.speaker_embeddings = nn.Embedding(4, config.embedding_size)

        if config.embedding_size != config.hidden_size:
            self.embeddings_project = nn.Linear(config.embedding_size, config.hidden_size)

        self.encoder = ElectraEncoder(config)
        self.classifier = nn.Sequential(OrderedDict([
            ('dense',nn.Linear(768, 768)),
            ('dropout', nn.Dropout(0.1)),
            ('out_proj', nn.Linear(768, 2)),
        ]))

        # Initialize weights and apply final processing
        self.post_init()
    
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        speaker_type_ids: Optional[torch.Tensor] = None,
        
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,):
        
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            # self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            if hasattr(self.embeddings, "token_type_ids"):
                buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape)

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

        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        hidden_states = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
        )
        speaker_type_embeddings = self.speaker_embeddings(speaker_type_ids)
        hidden_states += speaker_type_embeddings
        
        if hasattr(self, "embeddings_project"):
            hidden_states = self.embeddings_project(hidden_states)
        
        hidden_states = self.encoder(
            hidden_states,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        last_hidden_state = hidden_states.last_hidden_state
        
        # cls = torch.mean(last_hidden_state, dim=1)
        cls = last_hidden_state[:, 0, :]
        logit = self.classifier(cls)
        return logit

class GPT_Baseline(GPT2Model):
    def __init__(self, config, class_num=2, pad_token_id=0,
                 cls_token_id=2, sep_token_id=3, 
                 seq_split_id_list=[42000, 42001, 42002, 42003], 
                 sentence_ps='None', window_size=3):
        super().__init__(config)

        self.embed_dim = config.hidden_size
        self.pad_token_id = pad_token_id
        self.cls_token_id = cls_token_id
        self.sep_token_id = sep_token_id
        # self.is_avgpool = is_avgpool
        self.sentence_ps = sentence_ps
        self.window_size = window_size
        self.seq_split_id_list = seq_split_id_list

        self.wte = nn.Embedding(config.vocab_size, self.embed_dim)
        self.wpe = nn.Embedding(config.max_position_embeddings, self.embed_dim)

        self.speaker_embeddings = nn.Embedding(4, self.embed_dim)

        self.drop = nn.Dropout(config.embd_pdrop)
        self.h = nn.ModuleList([GPT2Block(config, layer_idx=i) for i in range(config.num_hidden_layers)])
        self.ln_f = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_epsilon)

        # Model parallel
        self.model_parallel = False
        self.device_map = None
        self.gradient_checkpointing = False
        self._attn_implementation = config._attn_implementation

        self.classifier = nn.Sequential(OrderedDict([
            ('dense',nn.Linear(self.embed_dim, self.embed_dim)),
            ('dropout', nn.Dropout(0.1)),
            ('out_proj', nn.Linear(self.embed_dim, class_num)),
        ]))

        # Initialize weights and apply final processing
        self.post_init()
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        speaker_type_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        is_inference: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPastAndCrossAttentions]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
            batch_size = input_ids.shape[0]
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size = inputs_embeds.shape[0]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, input_shape[-1])

        if past_key_values is None:
            past_length = 0
            past_key_values = tuple([None] * len(self.h))
        else:
            past_length = past_key_values[0][0].size(-2)
        if position_ids is None:
            position_ids = torch.arange(past_length, input_shape[-1] + past_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0)

        # Attention mask.
        if attention_mask is not None:
            attention_mask = attention_mask.view(batch_size, -1)
            if self._attn_implementation == "flash_attention_2":
                attention_mask = attention_mask if 0 in attention_mask else None
            else:
                # We create a 3D attention mask from a 2D tensor mask.
                # Sizes are [batch_size, 1, 1, to_seq_length]
                # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
                # this attention mask is more simple than the triangular masking of causal attention
                # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
                attention_mask = attention_mask[:, None, None, :]

                # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
                # masked positions, this operation will create a tensor which is 0.0 for
                # positions we want to attend and the dtype's smallest value for masked positions.
                # Since we are adding it to the raw scores before the softmax, this is
                # effectively the same as removing these entirely.
                attention_mask = attention_mask.to(dtype=self.dtype)  # fp16 compatibility
                attention_mask = (1.0 - attention_mask) * torch.finfo(self.dtype).min

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.add_cross_attention and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            if self._attn_implementation != "flash_attention_2":
                encoder_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # head_mask has shape n_layer x batch x n_heads x N x N
        head_mask = self.get_head_mask(head_mask, self.config.n_layer)

        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids)
        
        hidden_states = inputs_embeds + position_embeds

        if token_type_ids is not None:
            token_type_embeds = self.wte(token_type_ids)
            hidden_states = hidden_states + token_type_embeds

        if speaker_type_ids is not None:
            speaker_embeds = self.speaker_embeddings(speaker_type_ids)
            hidden_states = hidden_states + speaker_embeds

        hidden_states = self.drop(hidden_states)

        output_shape = (-1,) + input_shape[1:] + (hidden_states.size(-1),)

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        presents = () if use_cache else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None
        all_hidden_states = () if output_hidden_states else None
        for i, (block, layer_past) in enumerate(zip(self.h, past_key_values)):
            # Model parallel
            if self.model_parallel:
                torch.cuda.set_device(hidden_states.device)
                # Ensure layer_past is on same device as hidden_states (might not be correct)
                if layer_past is not None:
                    layer_past = tuple(past_state.to(hidden_states.device) for past_state in layer_past)
                # Ensure that attention_mask is always on the same device as hidden_states
                if attention_mask is not None:
                    attention_mask = attention_mask.to(hidden_states.device)
                if isinstance(head_mask, torch.Tensor):
                    head_mask = head_mask.to(hidden_states.device)
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:
                outputs = self._gradient_checkpointing_func(
                    block.__call__,
                    hidden_states,
                    None,
                    attention_mask,
                    head_mask[i],
                    encoder_hidden_states,
                    encoder_attention_mask,
                    use_cache,
                    output_attentions,
                )
            else:
                outputs = block(
                    hidden_states,
                    layer_past=layer_past,
                    attention_mask=attention_mask,
                    head_mask=head_mask[i],
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                )

            hidden_states = outputs[0]
            if use_cache is True:
                presents = presents + (outputs[1],)

            if output_attentions:
                all_self_attentions = all_self_attentions + (outputs[2 if use_cache else 1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (outputs[3 if use_cache else 2],)

            # Model Parallel: If it's the last layer for that device, put things on the next device
            if self.model_parallel:
                for k, v in self.device_map.items():
                    if i == v[-1] and "cuda:" + str(k) != self.last_device:
                        hidden_states = hidden_states.to("cuda:" + str(k + 1))

        hidden_states = self.ln_f(hidden_states)

        hidden_states = hidden_states.view(output_shape)
        # Add last hidden state
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        batch_size = input_ids.shape[0]
        device = hidden_states.device
        logits = self.classifier(hidden_states) # batch, seq_len, dim -> batch, seq_len, class_num
        
        # sequence_lengths = torch.eq(input_ids, self.pad_token_id).int().argmax(-1) - 1
        # sequence_lengths = sequence_lengths % input_ids.shape[-1]
        # sequence_lengths = sequence_lengths.to(device)
        
        # if self.is_avgpool:
        #     logit = []
        #     for i in range(batch_size):
        #         logit.append(torch.mean(logits[i], dim=0))
        #     logit = torch.stack(logit, dim=0)
        #     return logits, logit
        # else:
        #     logit = logits[torch.arange(batch_size, device=device), sequence_lengths]
        #     return logits, logit
        return self.post_processing_module(logits, input_ids, device, self.sentence_ps, self.window_size) # logits, logit, seq_logits


    def post_processing_module(self, logits, input_ids, device, sentence_ps='None', window_size=3):
        batch_size = input_ids.shape[0]
        # 여러 패딩 토큰 ID를 고려한 불리언 마스크 생성
        mask = torch.isin(input_ids, torch.tensor(self.seq_split_id_list, device=device))
        # True 값의 인덱스를 리스트 형태로 추출
        true_indices = [torch.where(mask[b])[0].tolist() for b in range(batch_size)]
        # 유효한 시퀀스 길이를 구함
        sequence_lengths = (torch.eq(input_ids, self.pad_token_id).int().argmax(-1) - 1) % input_ids.shape[-1]


        def segment_average_torch(input_tensor, true_indices, sequence_lengths):
                batch_size, seq_len, class_num = input_tensor.shape
                segment_avg_list = []

                for b in range(batch_size):
                    start_idx = 0
                    segments = true_indices[b]
                    
                    batch_segment_avg = []
                    for end_idx in segments:
                        if start_idx < end_idx:  # Ensure there is a valid segment
                            segment = input_tensor[b, start_idx:end_idx, :]  # 문장 단위
                            # Calculate segment average for the segment
                            avg_value = segment.mean(dim=0)
                            batch_segment_avg.append(avg_value)
                            start_idx = end_idx  # Move to the next segment
                        
                    if end_idx < sequence_lengths[b]:
                        segment = input_tensor[b, start_idx:sequence_lengths[b]+1, :]
                        avg_value = segment.mean(dim=0)
                        batch_segment_avg.append(avg_value)
                    segment_avg_list.append(torch.stack(batch_segment_avg))
                
                return segment_avg_list
        # def segment_average_torch(input_tensor, true_indices, sequence_lengths):
        #     batch_size, seq_len, class_num = input_tensor.shape
        #     segment_avg_list = []

        #     for b in range(batch_size):
        #         start_idx = 0
        #         segments = true_indices[b]
                
        #         batch_segment_avg = []
        #         for end_idx in segments:

        #             if start_idx < end_idx:  # Ensure there is a valid segment
        #                 segment = input_tensor[b, start_idx:end_idx, :]  # 문장 단위
                        
        #                 # Calculate segment average for the segment
        #                 avg_value = segment.mean(dim=0)

        #                 batch_segment_avg.append(avg_value)
                    
        #             start_idx = end_idx  # Move to the next segment
                
        #             if end_idx < sequence_lengths[b]:
        #                 segment = input_tensor[b, start_idx:sequence_lengths[b]+1, :]
        #                 avg_value = segment.mean(dim=0)
        #                 batch_segment_avg.append(avg_value)

        #         segment_avg_list.append(torch.stack(batch_segment_avg))
            
        #     return segment_avg_list

        def moving_average_segments(segment_avg_logits, window_size):
            moving_avg_segments = []

            for batch in segment_avg_logits:
                moving_avg_batch = []
                num_segments = batch.shape[0]
                
                for i in range(num_segments):
                    start = max(0, i - window_size + 1)
                    end = i + 1
                    window_avg = batch[start:end].mean(dim=0)
                    moving_avg_batch.append(window_avg)
                
                moving_avg_segments.append(torch.stack(moving_avg_batch))
            
            return moving_avg_segments

        if sentence_ps == 'None':
            # 세그먼트 평균 계산
            seq_logits = segment_average_torch(logits, true_indices, sequence_lengths)

        elif sentence_ps == 'moving_average':
            # 세그먼트 평균 계산
            segment_avg_logits = segment_average_torch(logits, true_indices, sequence_lengths)
            # 세그먼트 이동 평균 계산
            seq_logits = moving_average_segments(segment_avg_logits, window_size)

        logit = []
        for batch in seq_logits:
            v, _ = torch.max(batch, dim=0)
            logit.append(v)
        logit = torch.stack(logit, dim=0)

        return logits, logit, seq_logits

def get_Model(class_name):
    try:
        Myclass = eval(class_name)()
        return Myclass
    except NameError as e:
        print("Class [{}] is not defined".format(class_name))

def main():
    pass

if __name__ == "__main__":
    main()