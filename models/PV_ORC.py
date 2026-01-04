from math import sqrt
from transformers import PretrainedConfig, BertConfig, BertModel, BertTokenizer
from transformers import PreTrainedModel
from typing import List
from layers.Embed import PatchEmbedding, PatchEmbedding2Oter
from layers.M_encoder import Mul_Block
from layers.StandardNorm import Normalize
import torch.nn as nn
import torch
from models.P_encoder import PromptEncoder
from transformers.activations import ACT2FN
import torch.nn.functional as F

class MyConfig(PretrainedConfig):
    model_type = "MYMODEL"
    def __init__(
        self,
        args=None,
        block_type="bottleneck",
        layers: list[int] = [3, 4, 6, 3],
        num_classes: int = 24,
        input_channels: int = 96,
        cardinality: int = 1,
        base_width: int = 64,
        stem_width: int = 64,
        stem_type: str = "",
        avg_down: bool = False,
        **kwargs,
    ):
        if block_type not in ["basic", "bottleneck"]:
            raise ValueError(f"`block_type` must be 'basic' or bottleneck', got {block_type}.")
        if stem_type not in ["", "deep", "deep-tiered"]:
            raise ValueError(f"`stem_type` must be '', 'deep' or 'deep-tiered', got {stem_type}.")
        self.block_type = block_type
        self.layers = layers
        self.num_classes = num_classes
        self.input_channels = input_channels
        self.cardinality = cardinality
        self.base_width = base_width
        self.stem_width = stem_width
        self.stem_type = stem_type
        self.avg_down = avg_down
        super().__init__(**kwargs)
        if args != None:
            self.pred_len = args.pred_len
            self.seq_len = args.seq_len
            self.d_ff = args.d_ff
            self.llm_dim = args.llm_dim
            self.patch_len = args.patch_len
            self.stride = args.stride
            self.llm_layers = args.llm_layers
            self.prompt_domain = args.prompt_domain
            self.dropout = args.dropout
            self.d_model = args.d_model
            self.n_heads = args.n_heads
            self.enc_in = args.enc_in
            self.dec_in = args.dec_in
            self.c_out = args.c_out
            self.oenc_dim = args.oenc_dim
            self.freq =args.freq
        else:
            self.pred_len = 96
            self.seq_len = 96
            self.d_ff = 32
            self.llm_dim = 768
            self.patch_len = 16
            self.stride = 8
            self.llm_layers = 6
            self.prompt_domain = 0
            self.dropout = 0.1
            self.d_model = 16
            self.n_heads = 8 
            self.enc_in = 8
            self.dec_in = 8
            self.c_out = 8
            self.oenc_dim = 128
            self.freq = 'h'

class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x


class MyModel(PreTrainedModel):
    config_class = MyConfig

    def __init__(self, config):
        super().__init__(config)
        self.llm = LLM(config)
        # self.l1 = nn.Linear(config.input_channels, config.num_classes)

    def forward(self, seq_x,seq_x_mark,**kwargs):
        x = self.llm(seq_x, seq_x_mark)
        # x = self.l1(seq_x)
        # return x
        return {"logits": x} 



class LLM(nn.Module):
    def __init__(self, configs, patch_len=16, stride=8):
        super().__init__()
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.d_ff = configs.d_ff
        self.top_k = 5
        self.d_llm = configs.llm_dim
        self.patch_len = configs.patch_len
        self.stride = configs.stride

        self.bert_config = BertConfig.from_pretrained('/BERT/PATH') #"input your BERT model path"
        
        self.bert_config.num_hidden_layers = configs.llm_layers
        self.bert_config.output_attentions = True
        self.bert_config.output_hidden_states = True

        self.bert_config.vocab_size = 30523
        self.llm_model = BertModel(self.bert_config)

        try:
            self.tokenizer = BertTokenizer.from_pretrained(
                '/BERT/PATH',
                trust_remote_code=True,
                local_files_only=True
            )
        except EnvironmentError:  # downloads the tokenizer from HF if not already done
            print("Local tokenizer files not found. Atempting to download them..")
            self.tokenizer = BertTokenizer.from_pretrained(
                '/BERT/PATH',
                trust_remote_code=True,
                local_files_only=False
            )

        if self.tokenizer.eos_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            pad_token = '[PAD]'
            self.tokenizer.add_special_tokens({'pad_token': pad_token})
            self.tokenizer.pad_token = pad_token

        for param in self.llm_model.parameters():
            param.requires_grad = False

        if configs.prompt_domain:
            self.description = configs.content
        else:
            self.description = 'The Electricity Transformer Temperature (ETT) is a crucial indicator in the electric power long-term deployment.'

        '''
        tokenizer 添加P-turing元素        
        '''
        if "[P]" not in self.tokenizer.get_vocab():
            self.tokenizer.add_special_tokens({"additional_special_tokens": ["[P]"]})
####################################--train--######################################
        # emb = self.llm_model.get_input_embeddings()
        # if emb.num_embeddings != len(self.tokenizer):
        #     self.llm_model.resize_token_embeddings(len(self.tokenizer))
###############################################################################
        self.P_ids = self.tokenizer.get_vocab()["[P]"]
        self.p_encoder = PromptEncoder(prompt_len=4,embed_dim=self.d_llm)


        self.dropout = nn.Dropout(configs.dropout)

        self.patch_embedding = PatchEmbedding(
            configs.d_model, self.patch_len, self.stride, configs.dropout)
        
        # for other data
        self.oter_enc = Mul_Block(configs=configs, freature_flag='TRANSFORMER')
        
        self.act = nn.ReLU()
        self.o_pred_lab = nn.Linear(self.seq_len, self.pred_len)
        self.layer_norm = nn.LayerNorm(self.pred_len)
        self.dim2llm_d = nn.Linear(configs.oenc_dim, 1)
        # self.patch_embedding2Ot = PatchEmbedding2Oter(configs.d_model, self.patch_len, self.stride, configs.dropout)

        self.word_embeddings = self.llm_model.get_input_embeddings().weight
        self.vocab_size = self.word_embeddings.shape[0]
        self.num_tokens = 1000
        self.mapping_layer = nn.Linear(self.vocab_size, self.num_tokens)

        self.reprogramming_layer = ReprogrammingLayer(configs.d_model, configs.n_heads, self.d_ff, self.d_llm)

        self.patch_nums = int((configs.seq_len - self.patch_len) / self.stride + 2)
        # self.patch_nums = 200
        self.head_nf = self.d_ff * self.patch_nums

        self.output_projection = FlattenHead(configs.enc_in, self.head_nf, self.pred_len,
                                                 head_dropout=configs.dropout)

        self.normalize_layers = Normalize(configs.enc_in, affine=False)
        self.onomalize_layers_o = Normalize(configs.enc_in, affine=False)

        self.project = nn.Linear(self.pred_len*2, self.pred_len)
        self.layer_norm2p = nn.LayerNorm(self.pred_len*2)

        #Moe
        self.moe = TimeMoeSparseExpertsLayer(hidden_size=configs.pred_len*2)

    def forward(self, x_enc, x_mark_enc,mask=None,labels=None):
        dec_out = self.forecast(x_enc, x_mark_enc)
        return dec_out[:, -self.pred_len:, :]

    def forecast(self, x_enc, x_mark_enc):
        x_o_enc = x_enc[:,:,:-1]
        x_enc = x_enc[:,:,-1].unsqueeze(-1)
        x_enc = self.normalize_layers(x_enc, 'norm')
        x_o_enc = self.onomalize_layers_o(x_o_enc, 'norm').to(torch.float32)

        B, T, N = x_enc.size()
        x_enc = x_enc.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)

        min_values = torch.min(x_enc, dim=1)[0]
        max_values = torch.max(x_enc, dim=1)[0]
        medians = torch.median(x_enc, dim=1).values
        lags = self.calcute_lags(x_enc)
        trends = x_enc.diff(dim=1).sum(dim=1)

        prompt = []
        for b in range(x_enc.shape[0]):
            min_values_str = str(min_values[b].tolist()[0])
            max_values_str = str(max_values[b].tolist()[0])
            median_values_str = str(medians[b].tolist()[0])
            lags_values_str = str(lags[b].tolist())
            prompt_ = (
                f"[P] [P] [P] [P]"
                f"<|start_prompt|>Dataset description: {self.description}"
                f"Task description: forecast the next {str(self.pred_len)} steps given the previous {str(self.seq_len)} steps information;"
                "Input statistics: "
                f"min value {min_values_str}, "
                f"max value {max_values_str}, "
                f"median value {median_values_str}, "
                f"the trend of input is {'upward' if trends[b] > 0 else 'downward'}, "
                f"top 5 lags are : {lags_values_str}<|<end_prompt>|>"
            )

            prompt.append(prompt_)

        x_enc = x_enc.reshape(B, N, T).permute(0, 2, 1).contiguous()

        prompt = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=2048).input_ids
        
        #处理P-turing元素
        
        P_location = prompt.eq(self.P_ids).nonzero()
        P_prompt = self.p_encoder(P_location.shape[0]//4)
        prompt[(prompt==self.P_ids)] = self.tokenizer.pad_token_id
        prompt_embeddings = self.llm_model.get_input_embeddings()(prompt.to(x_enc.device))  # (batch, prompt_token, dim)

        for b,i in P_location:
            prompt_embeddings[b,i,:] = P_prompt[b,i-1,:]
        pass 


        source_embeddings = self.mapping_layer(self.word_embeddings.permute(1, 0)).permute(1, 0)

        x_enc = x_enc.permute(0, 2, 1).contiguous()
        enc_out, n_vars = self.patch_embedding(x_enc.to(torch.float32)) #x_enc.to(torch.bfloat16)

        # other data
        o_enc_out = self.oter_enc(x_o_enc, x_mark_enc)
        o_enc_out = self.act(self.o_pred_lab(o_enc_out.permute(0,2,1)))
        o_enc_out = self.layer_norm(o_enc_out).permute(0,2,1)
        o_enc_out = self.dim2llm_d(o_enc_out)

        enc_out = self.reprogramming_layer(enc_out, source_embeddings, source_embeddings)
        llama_enc_out = torch.cat([prompt_embeddings, enc_out], dim=1)
        dec_out = self.llm_model(inputs_embeds=llama_enc_out).last_hidden_state
        dec_out = dec_out[:, :, :self.d_ff]
        # dec_out = torch.cat([dec_out, o_enc_out], dim=1)


        dec_out = torch.reshape(
            dec_out, (-1, n_vars, dec_out.shape[-2], dec_out.shape[-1]))
        dec_out = dec_out.permute(0, 1, 3, 2).contiguous()

        dec_out = self.output_projection(dec_out[:, :, :, -self.patch_nums:])
        dec_out = dec_out.permute(0, 2, 1).contiguous()

        dec_out = torch.cat([dec_out, o_enc_out], dim=1).permute(0, 2, 1)

        dec_out = self.moe(dec_out)


        dec_out = self.layer_norm2p(dec_out[0])
        dec_out = self.project(dec_out).permute(0, 2, 1)
        dec_out = self.normalize_layers(dec_out, 'denorm')

        return dec_out

    def calcute_lags(self, x_enc):
        q_fft = torch.fft.rfft(x_enc.permute(0, 2, 1).contiguous(), dim=-1)
        k_fft = torch.fft.rfft(x_enc.permute(0, 2, 1).contiguous(), dim=-1)
        res = q_fft * torch.conj(k_fft)
        corr = torch.fft.irfft(res, dim=-1)
        mean_value = torch.mean(corr, dim=1)
        _, lags = torch.topk(mean_value, self.top_k, dim=-1)
        return lags


class ReprogrammingLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_keys=None, d_llm=None, attention_dropout=0.1):
        super(ReprogrammingLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)

        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.value_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.out_projection = nn.Linear(d_keys * n_heads, d_llm)
        self.n_heads = n_heads
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, target_embedding, source_embedding, value_embedding):
        B, L, _ = target_embedding.shape
        S, _ = source_embedding.shape
        H = self.n_heads

        target_embedding = self.query_projection(target_embedding).view(B, L, H, -1)
        source_embedding = self.key_projection(source_embedding).view(S, H, -1)
        value_embedding = self.value_projection(value_embedding).view(S, H, -1)

        out = self.reprogramming(target_embedding, source_embedding, value_embedding)

        out = out.reshape(B, L, -1)

        return self.out_projection(out)

    def reprogramming(self, target_embedding, source_embedding, value_embedding):
        B, L, H, E = target_embedding.shape

        scale = 1. / sqrt(E)

        scores = torch.einsum("blhe,she->bhls", target_embedding, source_embedding)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        reprogramming_embedding = torch.einsum("bhls,she->blhe", A, value_embedding)

        return reprogramming_embedding


class TimeMoeTemporalBlock(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, hidden_act: str):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[hidden_act]

    def forward(self, hidden_state):
        return self.down_proj(self.act_fn(self.gate_proj(hidden_state)) * self.up_proj(hidden_state))

class TimeMoeSparseExpertsLayer(nn.Module):
    def __init__(self, top_k=2, hidden_size=96, num_experts=8, norm_topk_prob=False):
        super().__init__()
        self.top_k = top_k
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.norm_topk_prob = False
        self.intermediate_size = 22016
        moe_intermediate_size = 22016 // self.top_k

        # gating
        self.gate = nn.Linear(hidden_size, self.num_experts, bias=False)
        self.experts = nn.ModuleList(
            [TimeMoeTemporalBlock(
                hidden_size=self.hidden_size,
                intermediate_size=moe_intermediate_size,
                hidden_act="silu",
            ) for _ in range(self.num_experts)]
        )

        self.shared_expert = TimeMoeTemporalBlock(
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            hidden_act='silu',
        )
        self.shared_expert_gate = torch.nn.Linear(hidden_size, 1, bias=False)

    def forward(self, hidden_states: torch.Tensor):
        """ """
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        # router_logits -> (batch * sequence_length, n_experts)
        router_logits = self.gate(hidden_states)

        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        if self.norm_topk_prob:
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        # we cast back to the input dtype
        routing_weights = routing_weights.to(hidden_states.dtype)

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
        )

        # One hot encode the selected experts to create an expert mask
        # this will be used to easily index which expert is going to be sollicitated
        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)

        # Loop over all available experts in the model and perform the computation on each expert
        for expert_idx in range(self.num_experts):
            expert_layer = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx])

            # Index the correct hidden states and compute the expert hidden state for
            # the current expert. We need to make sure to multiply the output hidden
            # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
            current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
            current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx, None]

            # However `index_add_` only support torch tensors for indexing so we'll use
            # the `top_x` tensor here.
            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))

        shared_expert_output = self.shared_expert(hidden_states)
        shared_expert_output = F.sigmoid(self.shared_expert_gate(hidden_states)) * shared_expert_output

        final_hidden_states = final_hidden_states + shared_expert_output

        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        return final_hidden_states, router_logits
