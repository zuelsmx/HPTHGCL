import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SVD(nn.Module):
    def __init__(self, N, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.U = nn.Parameter(torch.randn(N, hidden_dim) / np.sqrt(N), requires_grad=True)
        self.S = nn.Parameter(torch.ones(hidden_dim, dtype=torch.float32), requires_grad=True)
        self.V = nn.Parameter(torch.randn(hidden_dim, N) / np.sqrt(hidden_dim), requires_grad=True)

    def forward(self, A):
        new_A = torch.matmul(self.U, torch.diag(self.S))
        new_A = torch.matmul(new_A, self.V) / self.hidden_dim

        identity = torch.eye(n=self.hidden_dim, device=A.device)
        regular_U = F.mse_loss(torch.matmul(self.U.t(), self.U), identity)
        regular_V = F.mse_loss(torch.matmul(self.V, self.V.t()), identity)

        loss = F.mse_loss(A, new_A) + regular_U + regular_V

        s_safe = F.softplus(self.S) + 1e-6
        U = torch.matmul(self.U, torch.sqrt(torch.diag(s_safe)))
        V = torch.matmul(torch.sqrt(torch.diag(s_safe)), self.V)

        pe_Q = U
        pe_K = V.t()
        return loss, pe_Q, pe_K


class get_all_pe(nn.Module):
    def __init__(self, num_nodes, hidden_dim, num_edge_types):
        super().__init__()
        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim
        self.num_edge_types = num_edge_types

        self.svd_list = nn.Sequential()
        for i in range(num_edge_types):
            self.svd_list.add_module(name=f'svd_{i}', module=SVD(N=num_nodes, hidden_dim=hidden_dim))

    def forward(self, original_A):
        pe_Q_list = []
        pe_K_list = []
        loss_topo = 0
        for k, block in enumerate(self.svd_list):
            loss, pe_Q, pe_K = block(original_A[k])
            pe_Q_list.append(pe_Q)
            pe_K_list.append(pe_K)
            loss_topo += loss
        all_pe_Q = torch.cat(pe_Q_list, dim=-1)
        all_pe_K = torch.cat(pe_K_list, dim=-1)
        return loss_topo / self.num_edge_types, all_pe_Q, all_pe_K
