import torch
import torch.nn as nn
import torch.nn.functional as F

from core.common.constants import *

INVERSION_DICT = {
    'M': {8: '6', 5: '6/4'},
    'm': {9: '6', 5: '6/4'},
    'd': {9: '6', 6: '6/4'},
    '7': {8: '6/5', 5: '4/3', 2: '2'},
    'M7': {8: '6/5', 5: '4/3', 1: '2'},
    'm7': {9: '6/5', 5: '4/3', 2: '2'},
    'd7': {9: '6/5', 6: '4/3', 3: '2'}
}

PITCH_TO_DEGREE = {
    0: 'i',
    1: 'ii',
    2: 'ii',
    3: 'iii',
    4: 'iii',
    5: 'iv',
    6: 'iv',
    7: 'v',
    8: 'vi',
    9: 'vi',
    10: 'vii',
    11: 'vii'
}


def get_activation_fn(config):
    if config.activation_fn.lower() == 'tanh':
        activation_fn = nn.Tanh()
    elif config.activation_fn.lower() == 'relu':
        activation_fn = nn.ReLU()
    elif config.activation_fn.lower() == 'lrelu':
        activation_fn = nn.LeakyReLU()
    elif config.activation_fn.lower() == 'mish':
        activation_fn = nn.Mish()
    else:
        raise NotImplementedError
    return activation_fn


def masked_softmax(x, mask, dim, eps=1e-13):
    # Reference: https://github.com/allenai/allennlp/blob/main/allennlp/nn/util.py
    assert x.size() == mask.size()
    mask = mask.float()
    result = F.softmax(x * mask, dim=dim)
    result = result * mask
    result = result / (result.sum(dim=dim, keepdim=True) + eps)
    return result


class ChromaEncoder(nn.Module):
    def __init__(self, config):
        super(ChromaEncoder, self).__init__()
        self.device = config.device
        self.dropout_p = config.dropout_p
        self.no_shift = config.no_shift
        self.num_roots = 12
        self.chroma_dim = 12
        self.mlp_hidden_size = config.mlp_hidden_size
        self.activation_fn = get_activation_fn(config)
        self.local_observation_encoder = nn.Sequential(
            nn.Linear(self.chroma_dim, self.chroma_dim),
            self.activation_fn,
            nn.Dropout(p=self.dropout_p)
        )

        self.num_encoder_rnn_layers = config.num_encoder_rnn_layers
        self.lstm = nn.LSTM(
            input_size=self.chroma_dim,
            hidden_size=self.chroma_dim,
            num_layers=self.num_encoder_rnn_layers,
            dropout=self.dropout_p
        )
        self.attn_layer = nn.Sequential(
            nn.Linear(self.chroma_dim + 1, self.mlp_hidden_size),
            self.activation_fn,
            nn.Dropout(p=self.dropout_p),
            nn.Linear(self.mlp_hidden_size, 1)
        )

    def forward(self, x, lengths):
        # x (B x R, L, C)
        batch_size, max_sequence_length, _ = x.size()
        sequence_mask = torch.arange(max_sequence_length).unsqueeze(0).expand(
            batch_size, max_sequence_length).to(self.device) < lengths.unsqueeze(-1)

        # rnn encoder
        xf_local_enc = self.local_observation_encoder(x)  # (B, L, C)
        xf_local_enc = xf_local_enc.transpose(0, 1).float()  # (L, B or B x R, C)
        xf_rnn_enc, _ = self.lstm(xf_local_enc)  # (L, B, C)

        xf_rnn_enc = xf_rnn_enc.transpose(0, 1)  # (B, L, C)
        # xb_rnn_enc = xb_rnn_enc.transpose(0, 1)  # (B, L, C)
        sequence_mask_expand = sequence_mask.unsqueeze(-1).expand(
            batch_size, max_sequence_length, self.chroma_dim)  # (B, L, C)

        # sequence mask
        xf_rnn_enc_out = torch.zeros_like(xf_rnn_enc)
        xf_rnn_enc_out[sequence_mask_expand] = xf_rnn_enc.masked_select(sequence_mask_expand)  # (B or B x R, L, C)

        # concat and attention
        t_ratios = torch.clamp_max(torch.arange(max_sequence_length, device=self.device).unsqueeze(0).expand(
            batch_size, max_sequence_length) / lengths.unsqueeze(-1), max=1.0)  # (B, L)
        attn = F.softmax(self.attn_layer(
            torch.cat([
                t_ratios.unsqueeze(-1),
                xf_rnn_enc_out
            ], dim=-1)
        ), dim=1).squeeze(-1)  # (B, L, 1) -> (B, L)
        x_enc = torch.einsum('bl,blc->bc', attn, xf_rnn_enc_out)
        return x_enc


class NeuralHSMM(nn.Module):
    def __init__(self, config, eps=1e-13):
        super().__init__()
        self._eps = eps
        self._logit_eps = 1e-6
        self.device = config.device
        self.dropout_p = config.dropout_p
        self.max_residential_time = config.max_residential_time
        self.no_shift = config.no_shift

        self.activation_fn = get_activation_fn(config)
        self.mlp_hidden_size = config.mlp_hidden_size

        self._quality_templates = [
            [0, 4, 7],  # major
            [0, 3, 7],  # minor
            [0, 3, 6],  # diminish
            [0, 4, 7, 10],  # dominant seventh
            [0, 4, 7, 11],  # major seventh
            [0, 3, 7, 10],  # minor seventh
            [0, 3, 6, 9]  # diminish seventh
        ]
        self._quality_names = [
            'M',
            'm',
            'd',
            '7',
            'M7',
            'm7',
            'd7',
            'Rest'
        ]
        self.num_qualities = len(self._quality_templates)
        self.quality_magnification = config.quality_magnification

        self.num_modes = config.num_modes
        self.num_roots = 12
        self.num_tonics = 12
        self.chroma_dim = 12

        # encoder
        self.chroma_encoder = ChromaEncoder(config)

        # mode
        self.mode_emb_rnn_size = self.chroma_dim
        self.mode_emb_rnn = nn.LSTMCell(self.mode_emb_rnn_size, self.mode_emb_rnn_size)

        # key
        self.shift_layer = nn.Sequential(
            nn.Linear(self.chroma_dim * 2, self.mlp_hidden_size),
            self.activation_fn,
            nn.Dropout(p=self.dropout_p),
            nn.Linear(self.mlp_hidden_size, self.chroma_dim)
        )

        # initial
        self.initial_root_layer = nn.Sequential(
            nn.Linear(self.chroma_dim * 2, self.mlp_hidden_size),
            self.activation_fn,
            nn.Dropout(p=self.dropout_p),
            nn.Linear(self.mlp_hidden_size, 1)
        )

        # transition
        self.root_transition_layer = nn.Sequential(
            nn.Linear(self.chroma_dim * 3, self.mlp_hidden_size),
            self.activation_fn,
            nn.Dropout(p=self.dropout_p),
            nn.Linear(self.mlp_hidden_size, 1)
        )
        self.inter_key_transition_limit = config.inter_key_transition_limit
        self.inter_key_transition_ratio_logits = nn.Parameter(torch.randn(1))

        # emission
        self.mode_quality_layer = nn.Linear(self.chroma_dim, self.chroma_dim * self.num_roots, bias=False)

        # residence
        self.master_residence_logits = nn.Parameter(torch.randn(self.max_residential_time))  # (D,)

    def get_mode_embeddings(self):
        hx = torch.zeros([1, self.mode_emb_rnn_size], device=self.device)
        cx = torch.zeros([1, self.mode_emb_rnn_size], device=self.device)
        output = [torch.zeros([1, self.chroma_dim], device=self.device)]
        for d in range(self.num_modes):
            hx, cx = self.mode_emb_rnn(output[-1], (hx, cx))
            output.append(hx)
        mode_embeddings = torch.cat(output[1:], dim=0)
        return mode_embeddings  # (M, C)

    def get_emission_logits(self, mode_embeddings):
        emission_rest_logits = torch.ones(self.chroma_dim, device=self.device) * (-self.quality_magnification)
        emission_rest_logits = emission_rest_logits.unsqueeze(0).unsqueeze(1).repeat(mode_embeddings.size(0), 1, 1)  # (M, 1, C)

        qc_logit = torch.ones((self.num_qualities, self.chroma_dim), device=self.device) * (-self.quality_magnification)
        for quality in range(self.num_qualities):
            for pitch in self._quality_templates[quality]:
                qc_logit[quality, pitch] = self.quality_magnification
        emission_chroma_logits = [qc_logit]
        for root in range(1, self.num_roots):
            emission_chroma_logits.append(torch.roll(qc_logit, shifts=root, dims=-1))
        emission_chroma_logits = torch.stack(emission_chroma_logits, dim=0)  # (R=12, Q, C=12)

        mode_root_embeddings = self.mode_quality_layer(mode_embeddings).view(
            self.num_modes, self.num_roots, self.chroma_dim)  # (M, R, C)

        emission_quality_logits = torch.einsum(
            'mrc,rqc->mrq',
            mode_root_embeddings,
            emission_chroma_logits)

        marginal_emission_chroma_logits = torch.einsum(
            'mrq,rqc->mrc', F.softmax(emission_quality_logits, dim=-1), emission_chroma_logits)  # (M, R, C)

        return emission_rest_logits, emission_quality_logits, emission_chroma_logits, marginal_emission_chroma_logits

    def get_initial_root_distribution(
            self,
            mode_embeddings,
            marginal_emission_logits,
            emission_rest_logits
    ):
        # initial root distribution
        initial_root_logits = self.initial_root_layer(
            torch.cat([
                mode_embeddings.unsqueeze(1).repeat(1, self.num_roots, 1),
                marginal_emission_logits
            ], dim=-1)
        ).squeeze(-1)  # (M, R)

        initial_rest_logits = self.initial_root_layer(
            torch.cat(
                [mode_embeddings.unsqueeze(1), emission_rest_logits],
                dim=-1)
        ).squeeze(-1)  # (M, 1)

        initial_root_distribution = torch.softmax(
            torch.cat(
                [initial_root_logits, initial_rest_logits], dim=-1),
            dim=-1
        )  # (M, R+1)
        if not self.no_shift:
            initial_root_distribution_core, initial_root_distribution_rest = torch.split(
                initial_root_distribution, [self.num_roots, 1], dim=-1)  # (M, R), (M, 1)
            initial_root_distribution_core_list = []
            for i in range(self.num_roots):
                initial_root_distribution_core_list.append(
                    torch.roll(initial_root_distribution_core, shifts=i, dims=-1))
            initial_root_distribution_core = torch.stack(
                initial_root_distribution_core_list, dim=1).view(-1, self.num_roots)  # (M, KR, R) -> (K, R)
            initial_root_distribution_rest = initial_root_distribution_rest.unsqueeze(1).expand(
                self.num_modes, self.num_roots, 1).reshape(-1, 1)  # (M, 1) -> (M, KR, 1) -> (K, 1)
            initial_root_distribution = torch.cat([
                initial_root_distribution_core,
                initial_root_distribution_rest
            ], dim=-1)  # (K, R+1)

        return initial_root_distribution  # (K, R+1)

    def get_key_distribution(self, mode_embeddings, x_enc):
        batch_size = x_enc.size(0)
        mode_distribution = F.softmax(
            torch.einsum('bc,mc->bm', x_enc, mode_embeddings), dim=-1)  # (B, M)
        if self.no_shift:
            key_distribution = mode_distribution
        else:
            shift_logits = self.shift_layer(
                torch.cat([
                    mode_embeddings.unsqueeze(0).expand(batch_size, self.num_modes, self.chroma_dim),
                    x_enc.unsqueeze(1).expand(batch_size, self.num_modes, self.chroma_dim)
                ], dim=-1)  # (B, M, 2C)
            )  # (B, M, KR)
            shift_distribution = F.softmax(shift_logits, dim=-1)  # (B, M, KR)
            key_distribution = mode_distribution.unsqueeze(-1) * shift_distribution  # (B, M, KR)
            key_distribution = key_distribution.view(-1, self.num_modes * self.num_roots)  # (B, K)
        return key_distribution  # (B, K)

    def get_emission_distribution(
            self,
            emission_quality_logits,
            emission_chroma_logits,
            emission_rest_logits
    ):
        # emission
        # Let REST be the last dimension of QUALITY.
        # Therefore, from root(0~12), the probability of quality corresponding to rest is set to 0,
        # and from the state corresponding to rest, conversely,
        # the probability of quality corresponding to rest is set to 1 and the other is set to 0.

        # chroma
        emission_chroma_logits = emission_chroma_logits.unsqueeze(0).expand(
            self.num_modes, self.num_roots, self.num_qualities, self.chroma_dim)  # (M, R, Q, C)
        emission_chroma_distribution = torch.sigmoid(emission_chroma_logits)  # (M, R, Q, C)
        emission_quality_distribution = torch.softmax(emission_quality_logits, dim=-1)  # (M, R, Q)
        emission_rest_distribution = torch.sigmoid(emission_rest_logits)  # (M, 1, C)
        if not self.no_shift:
            ex_emission_chroma_distribution_list = []
            for i in range(self.num_roots):
                ex_emission_chroma_distribution_list.append(
                    torch.roll(emission_chroma_distribution, shifts=(i, i), dims=(1, -1)))
            emission_chroma_distribution = torch.stack(ex_emission_chroma_distribution_list, dim=1).view(
                -1, self.num_roots, self.num_qualities, self.chroma_dim)  # (M, KR, R, Q, C) -> (K, R, Q, C)
            emission_rest_distribution = emission_rest_distribution.unsqueeze(1).expand(
                self.num_modes, self.num_roots, 1, self.chroma_dim
            ).reshape(-1, 1, self.chroma_dim)  # (M, KR, 1, C) -> (K, 1, C)
            emission_rest_distribution_ex1 = emission_rest_distribution.unsqueeze(1).expand(
                    self.num_roots * self.num_modes, self.num_roots, 1, self.chroma_dim)  # (K, R, 1, C)
            emission_rest_distribution_ex2 = emission_rest_distribution.unsqueeze(2).expand(
                    self.num_roots * self.num_modes, 1, self.num_qualities + 1, self.chroma_dim)  # (K, 1, Q+1, C)
        else:
            emission_rest_distribution_ex1 = emission_rest_distribution.unsqueeze(1).expand(
                self.num_modes, self.num_roots, 1, self.chroma_dim)  # (M, R, 1, C)
            emission_rest_distribution_ex2 = emission_rest_distribution.unsqueeze(2).expand(
                self.num_modes, 1, self.num_qualities + 1, self.chroma_dim)  # (M, 1, Q+1, C)
        emission_chroma_distribution = torch.cat([
            emission_chroma_distribution,
            emission_rest_distribution_ex1
        ], dim=-2)  # (K, R, Q+1, C)
        emission_chroma_distribution = torch.cat([
            emission_chroma_distribution,
            emission_rest_distribution_ex2
        ], dim=-3)  # (K, R+1, Q+1, C)

        # quality
        state_rest_to_quality = torch.cat([
            torch.zeros((emission_quality_distribution.size(0), 1, self.num_qualities), device=self.device),
            torch.ones((emission_quality_distribution.size(0), 1, 1), device=self.device),
        ], dim=-1)  # (M, 1, Q+1)
        if not self.no_shift:
            ex_emission_quality_distribution_list = []
            for i in range(self.num_roots):
                ex_emission_quality_distribution_list.append(
                    torch.roll(emission_quality_distribution, shifts=i, dims=1))
            emission_quality_distribution = torch.stack(ex_emission_quality_distribution_list, dim=1).view(
                self.num_modes * self.num_roots, self.num_roots, self.num_qualities
            )  # (M, KR, R, Q) -> (K, R, Q)
            state_rest_to_quality = state_rest_to_quality.unsqueeze(1).expand(
                self.num_modes, self.num_roots, 1, self.num_qualities + 1).reshape(
                -1, 1, self.num_qualities + 1)  # (M, KR, 1, Q+1) -> (K, 1, Q+1)
        emission_quality_distribution = torch.cat([
            emission_quality_distribution,
            torch.zeros(
                tuple(list(emission_quality_distribution.size()[:emission_quality_distribution.dim() - 1]) + [1]),
                device=self.device)
        ], dim=-1)  # (K, R, Q+1)
        emission_quality_distribution = torch.cat([
            emission_quality_distribution,
            state_rest_to_quality
        ], dim=-2)  # (K, R+1, Q+1)
        return emission_quality_distribution, emission_chroma_distribution

    def get_root_transition_distribution(
            self,
            mode_embeddings,
            marginal_emission_logits,
            emission_rest_logits,
    ):
        # mode_embeddings: (M, C)
        # marginal_emission_logits: (M, R, C)
        # emission_rest_logits: (M, 1, C)

        # root transition
        root_transition_logits = self.root_transition_layer(
            torch.cat([
                mode_embeddings.unsqueeze(1).unsqueeze(2).expand(self.num_modes, self.num_roots, self.num_roots, self.chroma_dim),
                marginal_emission_logits.unsqueeze(2).expand(self.num_modes, self.num_roots, self.num_roots, self.chroma_dim),
                marginal_emission_logits.unsqueeze(1).expand(self.num_modes, self.num_roots, self.num_roots, self.chroma_dim)
            ], dim=-1)
        ).squeeze(-1)  # (M, R, R)

        transition_to_rest_logits = self.root_transition_layer(
            torch.cat([
                mode_embeddings.unsqueeze(1).expand(self.num_modes, self.num_roots, self.chroma_dim),
                marginal_emission_logits,
                emission_rest_logits.expand(self.num_modes, self.num_roots, self.chroma_dim)
            ], dim=-1)
        ).squeeze(-1)  # (M, R)

        transition_from_rest_logits = self.root_transition_layer(
            torch.cat([
                mode_embeddings.unsqueeze(1).expand(self.num_modes, self.num_roots, self.chroma_dim),
                emission_rest_logits.expand(self.num_modes, self.num_roots, self.chroma_dim),
                marginal_emission_logits
            ], dim=-1)
        ).squeeze(-1)  # (M, R)

        cat_root_transition_logits = torch.cat([
            root_transition_logits,
            transition_to_rest_logits.unsqueeze(-1)
        ], dim=-1)  # (M, R, R+1)
        cat_transition_from_rest_logits = torch.cat([
            transition_from_rest_logits,
            torch.zeros(tuple(list(transition_from_rest_logits.size()[:-1]) + [1]), device=self.device)
        ], dim=-1).unsqueeze(-2)  # (M, 1, R+1)
        cat_root_transition_logits = torch.cat([
            cat_root_transition_logits,
            cat_transition_from_rest_logits
        ], dim=-2)  # (M, R+1, R+1)
        tmat_mask = (
            ~torch.eye(self.num_roots + 1, dtype=torch.bool, device=self.device)
        ).unsqueeze(0).expand(self.num_modes, self.num_roots + 1, self.num_roots + 1)  # (M, R+1, R+1)
        root_transition_distribution = masked_softmax(
            cat_root_transition_logits,
            mask=tmat_mask,
            dim=-1
        )  # (M, R+1, R+1)

        if not self.no_shift:
            root_transition_distribution, transition_from_rest_distribution = torch.split(
                root_transition_distribution, [self.num_roots, 1], dim=1
            )  # (M, R, R+1), (M, 1, R+1)
            transition_from_rest_distribution = transition_from_rest_distribution[:, :, :-1]  # (M, 1, R)
            root_transition_distribution, transition_to_rest_distribution = torch.split(
                root_transition_distribution, [self.num_roots, 1], dim=-1
            )  # (M, R, R), (M, R, 1)
            root_transition_distribution_list = []
            transition_to_rest_distribution_list = []
            transition_from_rest_distribution_list = []
            for i in range(self.num_roots):
                root_transition_distribution_list.append(
                    torch.roll(root_transition_distribution, shifts=(i, i), dims=(1, 2)))
                transition_to_rest_distribution_list.append(
                    torch.roll(transition_to_rest_distribution, shifts=i, dims=1))
                transition_from_rest_distribution_list.append(
                    torch.roll(transition_from_rest_distribution, shifts=i, dims=2))
            root_transition_distribution = torch.stack(root_transition_distribution_list, dim=1).view(
                -1, self.num_roots, self.num_roots)  # (M, KR, R, R) -> (K, R, R)
            transition_to_rest_distribution = torch.stack(transition_to_rest_distribution_list, dim=1).view(
                -1, self.num_roots, 1)  # (M, KR, R, 1) -> (K, R, 1)
            transition_from_rest_distribution = torch.stack(transition_from_rest_distribution_list, dim=1).view(
                -1, 1, self.num_roots)  # (M, KR, 1, R) -> (K, 1, R)
            transition_from_rest_distribution = torch.cat([
                transition_from_rest_distribution,
                torch.zeros((self.num_modes * self.num_roots, 1, 1), device=self.device)
            ], dim=-1)  # (K, 1, R+1)
            root_transition_distribution = torch.cat([
                root_transition_distribution,
                transition_to_rest_distribution
            ], dim=-1)  # (K, R, R+1)
            root_transition_distribution = torch.cat([
                root_transition_distribution,
                transition_from_rest_distribution
            ], dim=1)  # (K, R+1, R+1)

        return root_transition_distribution  # (K, R+1, R+1)

    def merge_transition_distribution(
            self,
            initial_root_distribution,
            root_transition_distribution,
            key_distribution
    ):
        batch_size, num_keys = key_distribution.size()
        # Key transitions are simplified to be independent of the previous key,
        # but cannot be transferred to the same key.
        self_key_transition_mask = torch.eye(num_keys, dtype=torch.bool, device=self.device).unsqueeze(0).expand(
            batch_size, num_keys, num_keys)  # (B, K, K)
        if 2 < num_keys:
            key_transition_logits = torch.log(
                key_distribution.unsqueeze(1).expand(batch_size, num_keys, num_keys))  # (B, K, K)
            key_transition_distribution = masked_softmax(
                key_transition_logits,
                mask=~self_key_transition_mask,
                dim=-1)  # (B, K, K)
        else:
            key_transition_distribution = (1.0 - torch.eye(num_keys, device=self.device)).unsqueeze(0).expand(
                batch_size, num_keys, num_keys)  # (B, K, K)
        inter_key_transition_ratio = torch.clamp(
            torch.torch.sigmoid(self.inter_key_transition_ratio_logits),
            min=0.0,
            max=self.inter_key_transition_limit
        )  # (1,)
        key_transition_distribution = torch.eye(
            num_keys, device=self.device
        ).unsqueeze(0).expand(batch_size, num_keys, num_keys) * (1.0 - inter_key_transition_ratio) + (
                                              key_transition_distribution * inter_key_transition_ratio)  # (B, K, K)
        self_key_transition_mask = self_key_transition_mask.unsqueeze(3).unsqueeze(4).expand(
            batch_size, num_keys, num_keys, self.num_roots + 1, self.num_roots + 1)  # (B, K, K', R+1, R+1)
        expand_key_transition_distribution = key_transition_distribution.unsqueeze(3).unsqueeze(4).expand(
            batch_size, num_keys, num_keys, self.num_roots + 1, self.num_roots + 1)  # (B, K, K', R+1, R+1')
        expand_initial_root_distribution = initial_root_distribution.unsqueeze(0).unsqueeze(1).unsqueeze(3).expand(
            batch_size, num_keys, num_keys, self.num_roots + 1, self.num_roots + 1
        ) * expand_key_transition_distribution  # (B, K, K', R+1, R+1')
        expand_root_transition_distribution = root_transition_distribution.unsqueeze(0).unsqueeze(1).expand(
            batch_size, num_keys, num_keys, self.num_roots + 1, self.num_roots + 1
        ) * expand_key_transition_distribution  # (B, K, K', R+1, R+1')
        transition_distribution = torch.where(
            self_key_transition_mask,
            expand_root_transition_distribution,
            expand_initial_root_distribution
        ).transpose(2, 3)  # (B, K, K' R+1, R+1') -> (B, K, R+1, K', R+1')
        transition_distribution = transition_distribution.reshape(
            batch_size, num_keys * (self.num_roots + 1), num_keys * (self.num_roots + 1))  # (S=K*(R+1), S=K*(R+1))
        return transition_distribution

    def calc_bernoulli_prob(self, x, emission_quality_distribution, emission_chroma_distribution, out_qx=False):
        batch_size = x.size(0)
        num_keys = emission_chroma_distribution.size(0)
        x = x.unsqueeze(1).unsqueeze(2).unsqueeze(3).expand(
            batch_size, num_keys, self.num_roots + 1, self.num_qualities + 1, self.chroma_dim)  # (B, K, R+1, Q+1, C)
        emission_chroma_distribution = emission_chroma_distribution.unsqueeze(0)
        prob_qx = torch.where(
            x.long() == 0,
            1.0 - emission_chroma_distribution,
            emission_chroma_distribution
        )  # (B, K, R+1, Q+1, C)
        prob_qx = torch.prod(prob_qx, dim=-1)  # (B, K, R+1, Q+1)
        prob = emission_quality_distribution.unsqueeze(0) * prob_qx  # (B, K, R+1, Q+1)
        if out_qx:
            return prob
        else:
            prob = prob.sum(dim=-1)  # (B, K, R+1)
            return prob  # (B, K, R+1)

    def viterbi_step(self, transition_distribution_t, omega):
        num_states = transition_distribution_t.size(-1)
        batch_size = omega.size(0)
        omega = omega.unsqueeze(-1).expand(batch_size, num_states, num_states)  # (B, S, S)
        temp = torch.log(transition_distribution_t) + omega  # (B, S, S)
        omega_temp, omega_arg = temp.max(dim=1)
        return omega_temp, omega_arg

    def forward(self, batch, viterbi=False):
        lengths = batch['sequence_length'].long()
        batch_size = lengths.size(0)
        max_length = lengths.max().item()
        x_chroma = batch['observation_chroma'][:, :max_length].float()  # (B, L, C=12)

        # encoder
        x_enc = self.chroma_encoder(x_chroma, lengths)  # (B, C)

        # mode embeddings
        mode_embeddings = self.get_mode_embeddings()  # (M, C)

        # key
        key_distribution = self.get_key_distribution(
            mode_embeddings=mode_embeddings,
            x_enc=x_enc
        )  # (B, K)
        num_keys = key_distribution.size(1)

        # emission logis
        (
            m_emission_rest_logits,
            m_emission_quality_logits,
            m_emission_chroma_logits,
            m_marginal_emission_chroma_logits
         ) = self.get_emission_logits(mode_embeddings)

        # emission distribution
        emission_quality_distribution, emission_chroma_distribution = self.get_emission_distribution(
            emission_quality_logits=m_emission_quality_logits,
            emission_chroma_logits=m_emission_chroma_logits,
            emission_rest_logits=m_emission_rest_logits,
        )  # (K, R+1, Q+1), (K, R+1, Q+1, C)

        # initial distribution
        initial_root_distribution = self.get_initial_root_distribution(
            mode_embeddings=mode_embeddings,
            marginal_emission_logits=m_marginal_emission_chroma_logits,
            emission_rest_logits=m_emission_rest_logits,
        )  # (K, R+1)

        initial_distribution = (key_distribution.unsqueeze(-1) * initial_root_distribution.unsqueeze(0)).view(
            batch_size, num_keys, self.num_roots + 1).view(batch_size, -1)  # (B, K, R+1) -> (B, S = K x (R+1))

        # transition distribution
        root_transition_distribution = self.get_root_transition_distribution(
            mode_embeddings=mode_embeddings,
            marginal_emission_logits=m_marginal_emission_chroma_logits,
            emission_rest_logits=m_emission_rest_logits,
        )  # (K, R+1, R+1)

        transition_distribution = self.merge_transition_distribution(
            initial_root_distribution=initial_root_distribution,
            root_transition_distribution=root_transition_distribution,
            key_distribution=key_distribution
        )  # (B, S=K*(R+1), S=K*(R+1))

        # residence
        num_states = num_keys * (self.num_roots + 1)
        residence_distribution = torch.softmax(self.master_residence_logits, dim=-1)  # (D,)
        residence_distribution = residence_distribution.unsqueeze(0).unsqueeze(1).unsqueeze(2).expand(
            batch_size, num_keys, self.num_roots + 1, self.max_residential_time)  # (B, K, R+1, D)
        residence_distribution = residence_distribution.view(batch_size, -1, self.max_residential_time)  # (B, S, D)

        chroma_emission_probs = self.calc_bernoulli_prob(
            x=x_chroma[:, 0],
            emission_quality_distribution=emission_quality_distribution,
            emission_chroma_distribution=emission_chroma_distribution
        ).view(batch_size, num_states)  # (B, K, R+1) -> (B, S)

        chroma_alpha_t = initial_distribution.unsqueeze(-1) * chroma_emission_probs.unsqueeze(
            -1) * residence_distribution  # (B, S, D)
        chroma_c_t = chroma_alpha_t.sum(dim=-1).sum(dim=-1)
        chroma_alpha_t = chroma_alpha_t / chroma_c_t.unsqueeze(-1).unsqueeze(-1)
        chroma_alpha_prev = chroma_alpha_t
        chroma_accum_logc = torch.log(chroma_c_t)

        if viterbi:
            chroma_omega_t = torch.log(initial_distribution).unsqueeze(-1) + torch.log(chroma_emission_probs).unsqueeze(-1) + torch.log(residence_distribution)
            chroma_omega_args_t = torch.zeros((batch_size, num_states, self.max_residential_time)).long().to(self.device)
            chroma_omega_prev = chroma_omega_t
            chroma_omega_args = [chroma_omega_args_t]  # list of (B, S, D)

        # forward
        for t in range(1, max_length):
            mask = torch.tensor([t]).to(self.device) < lengths
            chroma_emission_probs = torch.where(
                mask.unsqueeze(-1).repeat(1, num_states),
                self.calc_bernoulli_prob(
                    x=x_chroma[:, t],
                    emission_quality_distribution=emission_quality_distribution,
                    emission_chroma_distribution=emission_chroma_distribution
                ).view(batch_size, num_states),
                torch.zeros((batch_size, num_states), device=self.device)
            )  # (B, S)

            chroma_transition_forward = torch.einsum(
                'bs,bst->bt', chroma_alpha_prev[:, :, 0], transition_distribution)  # (B, S)
            chroma_alpha_t_transition = chroma_transition_forward.unsqueeze(
                -1) * residence_distribution * chroma_emission_probs.unsqueeze(-1)  # (B, S, D)
            chroma_alpha_t_residence = torch.zeros((batch_size, num_states, 1)).to(self.device)
            chroma_alpha_t_residence = torch.cat(
                [chroma_alpha_prev[:, :, 1:], chroma_alpha_t_residence], dim=-1
            ) * chroma_emission_probs.unsqueeze(-1)  # (B, S, D)
            chroma_alpha_t = chroma_alpha_t_transition + chroma_alpha_t_residence  # (B, S, D)

            chroma_c_t = chroma_alpha_t.sum(dim=-1).sum(dim=-1)
            chroma_c_t = torch.where(self._eps < chroma_c_t, chroma_c_t, torch.ones_like(chroma_c_t))  # avoid zero division
            chroma_alpha_t = chroma_alpha_t / chroma_c_t.unsqueeze(-1).unsqueeze(-1)
            chroma_alpha_prev = chroma_alpha_t

            chroma_accum_logc += torch.log(chroma_c_t)

            if viterbi:
                _chroma_omega_t_transition_max, _chroma_omega_t_transition_argmax = self.viterbi_step(
                    transition_distribution, chroma_omega_prev[:, :, 0])  # (B, S)
                chroma_omega_t_transition_max = _chroma_omega_t_transition_max.unsqueeze(-1) + torch.log(
                    residence_distribution) + torch.log(chroma_emission_probs).unsqueeze(-1)  # (B, S, D)
                chroma_omega_t_transition_argmax = _chroma_omega_t_transition_argmax.unsqueeze(-1).repeat(
                    1, 1, self.max_residential_time)  # (B, S, D)

                # r < max_residential_time
                # Note: If r < max_residential_time, there are two possibilities,
                # residence and transition, so check which is larger.
                # (Note that in the forward algorithm, the two are added together for marginalization,
                # but here we choose the larger of the two.)
                chroma_omega_t_residence = chroma_omega_prev[:, :, 1:] + torch.log(chroma_emission_probs).unsqueeze(
                    -1)  # (B, S, D-1)
                chroma_omega_t_residence_args = torch.ones_like(chroma_omega_t_residence).long().to(
                    self.device) * RESID_STATE  # (B, S, D-1)
                chroma_omega_t_cat = torch.cat(
                    [chroma_omega_t_transition_max[:, :, :-1].unsqueeze(-1),
                     chroma_omega_t_residence.unsqueeze(-1)],
                    dim=-1)  # (B, S, D-1, 2)
                chroma_omega_t_argmax_cat = torch.cat(
                    [chroma_omega_t_transition_argmax[:, :, :-1].unsqueeze(-1),
                     chroma_omega_t_residence_args.unsqueeze(-1)],
                    dim=-1)  # (B, S, D-1, 2)
                chroma_omega_t, chroma_gather_index = chroma_omega_t_cat.max(dim=-1)  # (B, S, D-1)
                chroma_omega_t_argmax = chroma_omega_t_argmax_cat.gather(
                    -1, chroma_gather_index.unsqueeze(-1)).squeeze(-1)  # (B, S, D-1)

                # cat r == max_residential_time
                chroma_omega_t = torch.cat([
                    chroma_omega_t,
                    chroma_omega_t_transition_max[:, :, -1].unsqueeze(-1)],
                    dim=-1)  # (B, S, D)
                chroma_omega_t_argmax = torch.cat([
                    chroma_omega_t_argmax,
                    chroma_omega_t_transition_argmax[:, :, -1].unsqueeze(-1)],
                    dim=-1)  # (B, S, D)

                chroma_omega_prev = torch.where(
                    mask.unsqueeze(1).unsqueeze(2).expand(batch_size, num_states, self.max_residential_time),
                    chroma_omega_t,
                    chroma_omega_prev
                )  # (B, S, D)

                chroma_omega_args.append(chroma_omega_t_argmax)

        # chroma likelihood
        chroma_logliks = chroma_accum_logc

        # loss
        loss = (-chroma_logliks).sum()

        if viterbi:
            # finish
            chroma_omega_args = torch.stack(chroma_omega_args, dim=-1)  # (B, S, D, L)
            chroma_joint_probs = chroma_omega_prev
            chroma_joint_probs = chroma_joint_probs[:, :, 0]  # (B, S), no residential-time remaining (0)
            chroma_best_probs, chroma_best_args = chroma_joint_probs.max(dim=1)  # (B,)

            # reconstruction
            batch_states = [[chroma_best_arg] for chroma_best_arg in chroma_best_args.tolist()]
            batch_residence = [[0] for _ in range(len(batch_states))]
            for t in range(0, max_length - 1)[::-1]:
                for ib in range(batch_size):
                    if t < (lengths[ib] - 1):
                        state = batch_states[ib][-1]
                        res = batch_residence[ib][-1]
                        state_from = chroma_omega_args[ib, state, res, t + 1].item()
                        if state_from == RESID_STATE:
                            batch_states[ib].append(state)
                            batch_residence[ib].append(res + 1)
                        else:
                            batch_states[ib].append(state_from)
                            batch_residence[ib].append(0)

            for ib in range(batch_size):
                batch_states[ib] = batch_states[ib][::-1]
                batch_residence[ib] = batch_residence[ib][::-1]

            # quality
            qualities = [[] for _ in range(batch_size)]
            batch_states_tensor = torch.nn.utils.rnn.pad_sequence(
                [torch.tensor(b, device=self.device) for b in batch_states],
                batch_first=True
            )  # (B, L)
            for t in range(max_length):
                qx = self.calc_bernoulli_prob(
                    x=x_chroma[:, t],
                    emission_quality_distribution=emission_quality_distribution,
                    emission_chroma_distribution=emission_chroma_distribution,
                    out_qx=True
                ).view(batch_size, num_states, self.num_qualities + 1)  # (B, S, Q+1)
                q_id = torch.argmax(qx, dim=-1)  # (B, S)
                for ib in range(batch_size):
                    if t < lengths[ib]:
                        qualities[ib].append(
                            self._quality_names[q_id[ib, batch_states_tensor[ib, t]].item()])

        results = {
            BATCH_SIZE: batch_size,
            LOG_LIKELIHOOD: chroma_logliks,
            LOCAL_LOSS: loss,
        }
        if viterbi:
            results[STATES] = batch_states
            results[RESIDENCES] = batch_residence
            results['qualities'] = qualities
        return results

    def get_hsmm_params(self, eps=1e-5):
        org_self_training = self.training
        self.eval()
        with torch.no_grad():
            org_no_shift = self.no_shift
            self.no_shift = True
            # mode embeddings
            mode_embeddings = self.get_mode_embeddings()  # (M, C)

            # residence
            residence_distribution = torch.softmax(self.master_residence_logits, dim=-1)  # (D,)
            residence_distribution = residence_distribution.unsqueeze(0).expand(
                self.num_roots + 1, self.max_residential_time)

            # emission logis
            (
                m_emission_rest_logits,
                m_emission_quality_logits,
                m_emission_chroma_logits,
                m_marginal_emission_chroma_logits
            ) = self.get_emission_logits(mode_embeddings)
            marginal_emission_distribution = F.sigmoid(m_marginal_emission_chroma_logits)  # (M, R, C)

            # initial distribution
            initial_root_distribution = self.get_initial_root_distribution(
                mode_embeddings=mode_embeddings,
                marginal_emission_logits=m_marginal_emission_chroma_logits,
                emission_rest_logits=m_emission_rest_logits,
            )  # (K, R+1)

            # emission distribution
            emission_quality_distribution, emission_chroma_distribution = self.get_emission_distribution(
                emission_quality_logits=m_emission_quality_logits,
                emission_chroma_logits=m_emission_chroma_logits,
                emission_rest_logits=m_emission_rest_logits,
            )  # (K, R+1, Q+1), (K, R+1, Q+1, C)

            # transition distribution
            root_transition_distribution = self.get_root_transition_distribution(
                mode_embeddings=mode_embeddings,
                marginal_emission_logits=m_marginal_emission_chroma_logits,
                emission_rest_logits=m_emission_rest_logits,
            )  # (K, R+1, R+1)

            # stationary transition distribution
            transition_stationary = []
            for m in range(self.num_modes):
                transition = root_transition_distribution[m]  # (R+1, R+1)
                avr_residence = (
                        torch.arange(1, self.max_residential_time + 1).unsqueeze(0).repeat(self.num_roots + 1, 1) * residence_distribution
                ).sum(dim=-1)
                prob_transition = 1.0 / avr_residence
                prob_residence = 1.0 - prob_transition
                self_transition = torch.eye(self.num_roots + 1) * prob_residence
                transition = prob_transition * transition + self_transition
                L, V = torch.linalg.eig(torch.t(transition))
                assert abs(L[0] - 1.0) < eps, abs(L[0] - 1.0)
                m_transition_stationary = V[:, 0]
                assert torch.max(m_transition_stationary.imag**2).item() < eps
                m_transition_stationary = m_transition_stationary.real
                if m_transition_stationary.sum() < 0.0:
                    m_transition_stationary = (-1.0) * m_transition_stationary
                transition_stationary.append(m_transition_stationary)
            transition_stationary = torch.stack(transition_stationary, dim=0)
            transition_stationary /= transition_stationary.sum(dim=-1, keepdim=True)
            tonic_ids = transition_stationary.argmax(dim=-1)

            tonic_qualities = []
            for m in range(self.num_modes):
                major_probs = 0.0
                minor_probs = 0.0
                dim_probs = 0.0
                for qi in range(self.num_qualities):
                    if 'm' in self._quality_names[qi]:
                        minor_probs += emission_quality_distribution[m, tonic_ids[m], qi]
                    elif 'd' in self._quality_names[qi]:
                        dim_probs += emission_quality_distribution[m, tonic_ids[m], qi]
                    else:
                        assert 'M' in self._quality_names[qi] or '7' == self._quality_names[qi]
                        major_probs += emission_quality_distribution[m, tonic_ids[m], qi]
                max_quality_id = torch.argmax(torch.tensor([major_probs, minor_probs, dim_probs], device=self.device))
                tonic_qualities.append(['M', 'm', 'd'][max_quality_id])

            # unigram distribution
            unigram_distribution = torch.einsum(
                'mr,mrc->mc',
                transition_stationary[:, :-1],
                marginal_emission_distribution
            )  # (M, C)
            self.no_shift = org_no_shift
        if org_self_training:
            self.train()
        return {
            'mode_emb': mode_embeddings,
            'emission_chroma': emission_chroma_distribution,
            'emission_quality': emission_quality_distribution,
            'marginal_emission_chroma': marginal_emission_distribution,
            'residence': residence_distribution,
            'initial_root': initial_root_distribution,
            'transition_root': root_transition_distribution,
            'transition_stationary': transition_stationary,
            'unigram': unigram_distribution,
            'tonic_ids': tonic_ids,
            'tonic_qualities': tonic_qualities
        }
