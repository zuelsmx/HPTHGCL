import copy
import random

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch_geometric.nn.conv import GCNConv

from GTLayer import GraphTransformerLayer
from hyperbolic_utils import HyperbolicAugmentor
from mf import SVD

EPS = 1e-15


def dropout_feat(x, drop_prob):
    drop_mask = torch.empty(
        (x.size(1),),
        dtype=torch.float32,
        device=x.device,
    ).uniform_(0, 1) < drop_prob
    x = x.clone()
    x[:, drop_mask] = 0
    return x


class EMA:
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


def update_moving_average(ema_updater, ma_model, current_model):
    for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
        old_weight, up_weight = ma_params.data, current_params.data
        ma_params.data = ema_updater.update_average(old_weight, up_weight)


def set_requires_grad(model, val):
    for p in model.parameters():
        p.requires_grad = val


class Encoder(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hid_dim: int,
        num_nodes: int,
        num_relations: int = 3,
        num_layers: int = 1,
        activation=torch.relu,
        heads=4,
        dropout=0.4,
    ):
        self.num_nodes = num_nodes
        super().__init__()
        self.local_conv = GCNConv(in_dim, hid_dim)
        self.global_attn = GraphTransformerLayer(
            in_dim=hid_dim,
            hid_dim=hid_dim,
            num_edge_types=1,
            heads=heads,
            dropout=dropout,
        )
        self.svds = nn.ModuleList([SVD(N=num_nodes, hidden_dim=hid_dim) for _ in range(num_relations)])
        self.activation = activation
        self.ln = nn.LayerNorm(hid_dim)
        self.gate = nn.Parameter(torch.tensor(0.0))

    def reset_parameters(self):
        self.local_conv.reset_parameters()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, rel_id: int = 0):
        local_x = self.activation(self.local_conv(x, edge_index))

        from torch_geometric.utils import degree

        deg = degree(edge_index[0], num_nodes=self.num_nodes)
        deg = torch.clamp(deg, min=1e-5)

        a_sparse = torch.sparse_coo_tensor(
            edge_index,
            torch.ones(edge_index.size(1), device=x.device),
            (self.num_nodes, self.num_nodes),
        )
        a_dense = a_sparse.coalesce().to_dense()
        identity = torch.eye(a_dense.size(0), device=a_dense.device)
        a_dense = torch.clamp(a_dense + identity, max=1.0)

        loss_topo, pe_q, pe_k = self.svds[rel_id](a_dense)
        z = self.global_attn(local_x, pe_q.detach(), pe_k.detach(), deg, a_dense)
        z = self.ln(z)

        g = torch.sigmoid(self.gate)
        final_z = g * local_x + (1 - g) * z

        return self.activation(final_z), loss_topo


class HPTHGCLModel(nn.Module):
    def __init__(self, encoder, hid_dim, num_relations, tau=0.5, alpha=0.5, moving_average_decay=0.8):
        super().__init__()

        self.loss_weights = nn.Parameter(torch.ones(4), requires_grad=True)
        self.online_encoder = encoder
        self.target_encoder1 = copy.deepcopy(self.online_encoder)
        self.target_encoder2 = copy.deepcopy(self.online_encoder)

        set_requires_grad(self.target_encoder1, False)
        set_requires_grad(self.target_encoder2, False)

        self.target_ema_updater = EMA(moving_average_decay)
        self.hid_dim = hid_dim
        self.num_relations = num_relations
        self.tau = tau
        self.alpha = alpha
        self.hyp_augmentor = HyperbolicAugmentor(c=1.0, radial_std=0.03, angular_std=0.03)

        self.local_projector = nn.Sequential(nn.Linear(hid_dim, hid_dim), nn.PReLU(), nn.Linear(hid_dim, hid_dim))
        self.global_projector = nn.Sequential(nn.Linear(hid_dim, hid_dim), nn.PReLU(), nn.Linear(hid_dim, hid_dim))
        self.weight = nn.Parameter(torch.Tensor(hid_dim, hid_dim), requires_grad=True)

        self.reset_parameters()

    def update_ma(self):
        update_moving_average(self.target_ema_updater, self.target_encoder1, self.online_encoder)
        update_moving_average(self.target_ema_updater, self.target_encoder2, self.online_encoder)

    def reset_parameters(self):
        nn.init.xavier_normal_(self.weight, gain=1.414)
        self.online_encoder.reset_parameters()
        for model in self.local_projector:
            if isinstance(model, nn.Linear):
                nn.init.xavier_normal_(model.weight, gain=1.414)
        for model in self.global_projector:
            if isinstance(model, nn.Linear):
                nn.init.xavier_normal_(model.weight, gain=1.414)

    def forward(self, x, edge_indices, combine=None):
        if combine is not None:
            zs = [self.online_encoder(x, edge_index, rel_id=i)[0] for i, edge_index in enumerate(edge_indices)]
            if combine == 'concat':
                embeddings = torch.concat(zs, dim=-1)
            elif combine == 'mean':
                embeddings = torch.stack(zs).mean(dim=0)
            return embeddings

    def loss(self, x, edge_indices):
        loss = 0.0
        num_contrasts = 0
        for i in range(self.num_relations):
            for j in range(i, self.num_relations):
                loss += self.contrast(x, edge_indices[i], edge_indices[j], rel_i=i, rel_j=j)
                num_contrasts += 1
        return loss / num_contrasts

    def contrast(self, x, edge_index_1, edge_index_2, rel_i, rel_j):
        from torch_geometric.utils import dropout_edge

        edge_index_1_drop, _ = dropout_edge(edge_index_1, p=0.2)
        edge_index_2_drop, _ = dropout_edge(edge_index_2, p=0.2)
        edge_index_0_drop = random.choice([edge_index_1_drop, edge_index_2_drop])

        x_0 = dropout_feat(x, 0.3)
        x_1 = dropout_feat(x, 0.3)
        x_2 = dropout_feat(x, 0.3)

        z0, loss_topo0 = self.online_encoder(x_0, edge_index_0_drop, rel_id=0)
        z1_orig, loss_topo1 = self.online_encoder(x_1, edge_index_1_drop, rel_id=rel_i)
        z2_orig, loss_topo2 = self.online_encoder(x_2, edge_index_2_drop, rel_id=rel_j)
        total_topo_loss = (loss_topo0 + loss_topo1 + loss_topo2) / 3.0

        _, z1_aug = self.hyp_augmentor(z1_orig)
        _, z2_aug = self.hyp_augmentor(z2_orig)

        with torch.no_grad():
            target_z0, _ = self.target_encoder1(x, edge_index_0_drop, rel_id=0)
            target_z1, _ = self.target_encoder1(x, edge_index_1_drop, rel_id=rel_i)
            target_z2, _ = self.target_encoder2(x, edge_index_2_drop, rel_id=rel_j)

        structure_sim = (z1_orig @ z1_orig.T) + (target_z0 @ target_z0.T) + (target_z2 @ target_z2.T)
        structure_sim = F.normalize(structure_sim, p=2, dim=-1, eps=1e-8)

        num_nodes = x.size(0)
        adj_dense = torch.sparse_coo_tensor(
            edge_index_0_drop.to(x.device),
            torch.ones(edge_index_0_drop.size(1), device=x.device),
            torch.Size([num_nodes, num_nodes]),
        ).to_dense()

        recon_structure_loss = torch.mean((adj_dense - structure_sim) ** 2)

        l_cn_1 = self.alpha * self.compute_loss(z1_aug, target_z0.detach(), loss_type='agg') + (
            1.0 - self.alpha
        ) * self.compute_loss(z1_aug, target_z2.detach(), loss_type='agg')
        l_cn_2 = self.alpha * self.compute_loss(z2_aug, target_z1.detach(), loss_type='agg') + (
            1.0 - self.alpha
        ) * self.compute_loss(z2_aug, target_z0.detach(), loss_type='agg')
        l_cn = (l_cn_1 + l_cn_2) / 2

        l_cv_0 = self.compute_loss(z0, z1_aug, loss_type='inter')
        l_cv_1 = self.compute_loss(z1_aug, z2_aug, loss_type='inter')
        l_cv_2 = self.compute_loss(z2_aug, z0, loss_type='inter')
        l_cv = (l_cv_0 + l_cv_1 + l_cv_2) / 3

        l_gl = (self.global_loss(z1_aug, z2_aug) + self.global_loss(z2_aug, z1_aug)) / 2

        log_weights = F.softmax(self.loss_weights, dim=0)
        loss = (
            log_weights[0] * recon_structure_loss
            + log_weights[1] * l_cn
            + log_weights[2] * l_cv
            + log_weights[3] * l_gl
        )

        return loss + total_topo_loss * 0.001

    def _sim(self, z1: torch.Tensor, z2: torch.Tensor):
        z1 = F.normalize(z1, eps=1e-8)
        z2 = F.normalize(z2, eps=1e-8)
        return torch.mm(z1, z2.t())

    def _similarity(self, z1, z2):
        sim = self._sim(z1, z2) / self.tau
        sim = torch.clamp(sim, max=88.0)
        return torch.exp(sim)

    def agg_loss(self, z1, z2):
        sim_matrix = self._similarity(z1, z2)
        diag_sim = sim_matrix.diag()
        ratio = diag_sim / (sim_matrix.sum(dim=-1) + 1e-8)
        ratio = torch.clamp(ratio, min=1e-7, max=0.9999)
        loss = -torch.log(-torch.log(ratio))
        return loss.mean()

    def inter_loss(self, z1, z2):
        sim_intra = self._similarity(z1, z1)
        sim_inter = self._similarity(z1, z2)
        diag_sim = sim_inter.diag()
        denominator = sim_intra.sum(dim=-1) + sim_inter.sum(dim=-1) - sim_intra.diag()
        ratio = diag_sim / (denominator + 1e-8)
        ratio = torch.clamp(ratio, min=1e-7, max=0.9999)
        loss = -torch.log(ratio)
        return loss.mean()

    def compute_loss(self, z1, z2, loss_type='agg'):
        h1, h2 = self.local_projector(z1), self.local_projector(z2)
        if loss_type == 'agg':
            return self.agg_loss(h1, h2)
        if loss_type == 'inter':
            return self.inter_loss(h1, h2)
        return None

    def readout(self, z):
        return z.mean(dim=0)

    def discriminate(self, z, summary, sigmoid=True):
        summary = torch.matmul(self.weight, summary)
        value = torch.matmul(z, summary)
        return torch.sigmoid(value) if sigmoid else value

    def global_loss(self, pos_z: torch.Tensor, neg_z: torch.Tensor):
        s = self.readout(pos_z)
        h = self.global_projector(s)
        pos_score = self.discriminate(pos_z, h, sigmoid=True)
        neg_score = self.discriminate(neg_z, h, sigmoid=True)
        pos_loss = -torch.log(pos_score + 1e-8).mean()
        neg_loss = -torch.log(1 - neg_score + 1e-8).mean()
        return (pos_loss + neg_loss) * 0.5
