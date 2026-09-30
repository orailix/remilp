"""Normalization and the feasibility-preserving reformulations."""

import math

import torch
from torch_geometric.data import HeteroData
from torch_geometric.transforms import BaseTransform, Compose
from torch_geometric.utils import scatter

from remilp.data.dataset import C2V, V2C

TRANSVECTION = ("variables", "transvection", "variables")


class MILPTransform(BaseTransform):
    @staticmethod
    def _extend_edge_stores(data, new_var, new_con, new_attr):
        v2c, c2v = data[V2C], data[C2V]
        v2c.edge_index = torch.cat(
            [v2c.edge_index, torch.stack([new_var, new_con])], dim=1
        )
        v2c.edge_attr = torch.cat([v2c.edge_attr, new_attr])
        c2v.edge_index = torch.cat(
            [c2v.edge_index, torch.stack([new_con, new_var])], dim=1
        )
        c2v.edge_attr = torch.cat([c2v.edge_attr, new_attr])


class NormalizeRows(MILPTransform):
    """Scale every row so its largest coefficient or bound has magnitude 1."""

    def forward(self, data: HeteroData) -> HeteroData:
        cons = data["constraints"]
        v2c, c2v = data[V2C], data[C2V]
        cons_index = v2c.edge_index[1]
        max_coeffs = scatter(
            v2c.edge_attr.abs(), cons_index, dim_size=cons.num_nodes, reduce="max"
        )
        stacked = torch.cat((cons.lhs_val, cons.rhs_val, max_coeffs), dim=1)
        row_scale = stacked.abs().amax(dim=1, keepdim=True).clamp(min=1e-12)
        cons.lhs_val /= row_scale
        cons.rhs_val /= row_scale
        v2c.edge_attr /= row_scale[cons_index]
        c2v.edge_attr /= row_scale[cons_index]
        return data


class NormalizeObjective(MILPTransform):
    """Scale objective coefficients to magnitude at most 1, tracking the scale."""

    def forward(self, data: HeteroData) -> HeteroData:
        raw_max = data["variables"].obj_coeffs.abs().max()
        data["variables"].obj_coeffs /= raw_max.clamp(min=1e-12)
        data.scaling_coeff *= raw_max
        return data


normalize = Compose([NormalizeRows(), NormalizeObjective()])


class AddRedundantConstraints(MILPTransform):
    """Append round(k_frac * n_cons) rows (at least 1), each a convex combination of r
    rows with its bounds loosened towards the normalized limit."""

    def __init__(self, k_frac: float, r: int):
        self.k_frac = k_frac
        self.r = r

    def forward(self, data: HeteroData) -> HeteroData:
        v2c, cons = data[V2C], data["constraints"]
        n_cons, n_vars = cons.num_nodes, data["variables"].num_nodes
        if self.k_frac <= 0:
            return data
        k = min(max(1, round(self.k_frac * n_cons)), n_cons)
        r = min(self.r, n_cons)
        device = cons.lhs_val.device
        if k == 0:
            return data

        sources = torch.randint(0, n_cons, (k, r), device=device)
        raw = torch.empty(k, r, device=device).exponential_()
        weights = raw / raw.sum(1, keepdim=True)
        w = weights.unsqueeze(-1)

        new_has_lhs = cons.has_lhs[sources].prod(dim=1)
        new_has_rhs = cons.has_rhs[sources].prod(dim=1)
        new_lhs = (w * cons.lhs_val[sources]).sum(dim=1)
        new_rhs = (w * cons.rhs_val[sources]).sum(dim=1)
        lhs_limit = new_lhs.clamp(max=-1.0)
        rhs_limit = new_rhs.clamp(min=1.0)
        new_lhs = (
            lhs_limit + torch.rand_like(new_lhs) * (new_lhs - lhs_limit)
        ) * new_has_lhs
        new_rhs = (
            rhs_limit + torch.rand_like(new_rhs) * (new_rhs - rhs_limit)
        ) * new_has_rhs
        cons.has_lhs = torch.cat([cons.has_lhs, new_has_lhs])
        cons.has_rhs = torch.cat([cons.has_rhs, new_has_rhs])
        cons.lhs_val = torch.cat([cons.lhs_val, new_lhs])
        cons.rhs_val = torch.cat([cons.rhs_val, new_rhs])
        cons.num_nodes = n_cons + k

        ei, ea = v2c.edge_index, v2c.edge_attr
        flat_new_con = torch.arange(k, device=device).repeat_interleave(r) + n_cons
        flat_src = sources.reshape(-1)
        flat_w = weights.reshape(-1)

        # Every edge of a source row contributes (weight * coeff) to the new row.
        src_sort_idx = flat_src.argsort()
        sorted_flat_src = flat_src[src_sort_idx]
        edge_mask = torch.isin(ei[1], flat_src)
        hit_var, hit_con, hit_attr = ei[0][edge_mask], ei[1][edge_mask], ea[edge_mask]
        hit_start = torch.searchsorted(sorted_flat_src, hit_con)
        hit_end = torch.searchsorted(sorted_flat_src, hit_con, right=True)
        hit_counts = hit_end - hit_start
        cum_hit = torch.zeros(hit_counts.shape[0] + 1, dtype=torch.long, device=device)
        cum_hit[1:] = hit_counts.cumsum(0)
        hit_rep = torch.repeat_interleave(
            torch.arange(hit_counts.shape[0], device=device), hit_counts
        )
        within_hit = torch.arange(hit_rep.shape[0], device=device) - cum_hit[hit_rep]
        src_match_idx = src_sort_idx[hit_start[hit_rep] + within_hit]

        nv = hit_var[hit_rep]
        nc = flat_new_con[src_match_idx]
        na = hit_attr[hit_rep] * flat_w[src_match_idx].unsqueeze(-1)

        # Merge duplicate (row, variable) pairs.
        combined = nc * n_vars + nv
        unique, inv = torch.unique(combined, return_inverse=True)
        merged_attr = scatter(na, inv, dim=0, dim_size=len(unique), reduce="sum")
        merged_var = unique % n_vars
        merged_con = unique // n_vars
        self._extend_edge_stores(data, merged_var, merged_con, merged_attr)
        return data


class Homogenize(MILPTransform):
    """Append a dummy variable fixed to 1 (last index)."""

    _DUMMY_FEATS = {
        "is_integer": 1.0,
        "has_lb": 1.0,
        "has_ub": 1.0,
        "lb_val": 1.0,
        "ub_val": 1.0,
    }

    def forward(self, data: HeteroData) -> HeteroData:
        var_store = data["variables"]
        n_vars = var_store.num_nodes
        for key, value in var_store.items():
            if torch.is_tensor(value) and value.size(0) == n_vars:
                row = value.new_full(
                    (1, *value.shape[1:]), self._DUMMY_FEATS.get(key, 0.0)
                )
                var_store[key] = torch.cat([value, row], dim=0)
        var_store.num_nodes = n_vars + 1
        return data


class Unhomogenize(MILPTransform):
    """Remove the dummy variable, folding its column into the row bounds; transvection
    edges from it become `transvection_shift`."""

    def forward(self, data: HeteroData) -> HeteroData:
        var_store, cons = data["variables"], data["constraints"]
        v2c, c2v = data[V2C], data[C2V]
        n_vars, n_cons = var_store.num_nodes, cons.num_nodes
        d = n_vars - 1

        ei, ea = v2c.edge_index, v2c.edge_attr
        is_dummy_edge = ei[0] == d
        if is_dummy_edge.any():
            const = scatter(
                ea[is_dummy_edge].squeeze(-1),
                ei[1][is_dummy_edge],
                dim_size=n_cons,
                reduce="sum",
            ).unsqueeze(-1)
            cons.lhs_val = cons.lhs_val - const * cons.has_lhs
            cons.rhs_val = cons.rhs_val - const * cons.has_rhs
            keep = ~is_dummy_edge
            v2c.edge_index, v2c.edge_attr = ei[:, keep], ea[keep]
            c2v_keep = c2v.edge_index[1] != d
            c2v.edge_index = c2v.edge_index[:, c2v_keep]
            c2v.edge_attr = c2v.edge_attr[c2v_keep]

        data.objective_shift = (
            (data.scaling_coeff * var_store.obj_coeffs[d]).reshape(1).clone()
        )

        if TRANSVECTION in data.edge_types:
            tv = data[TRANSVECTION]
            dummy_src = tv.edge_index[0] == d
            shift = var_store.obj_coeffs.new_zeros(n_vars, tv.edge_attr.shape[1])
            shift[tv.edge_index[1][dummy_src]] = tv.edge_attr[dummy_src]
            var_store.transvection_shift = shift
            keep = ~dummy_src
            tv.edge_index, tv.edge_attr = tv.edge_index[:, keep], tv.edge_attr[keep]

        for key, value in var_store.items():
            if torch.is_tensor(value) and value.size(0) == n_vars:
                var_store[key] = value[:d]
        var_store.num_nodes = d
        return data


class TransformVariables(MILPTransform):
    """Substitute x_j <- u x_j + lambda x_l + mu for round(k_frac * candidates) targets
    (at least 1, at most k_cap). Requires Homogenize; records M - I as TRANSVECTION edges.
    """

    def __init__(
        self,
        k_frac: float,
        k_cap: int,
        lambda_bound: float = 3.0,
        u_bound: float = 2.0,
    ):
        self.k_frac = k_frac
        self.k_cap = k_cap
        self.lambda_bound = lambda_bound
        self.u_bound = u_bound

    @staticmethod
    def _empty_store(data, device):
        store = data[TRANSVECTION]
        store.edge_index = torch.zeros((2, 0), dtype=torch.long, device=device)
        store.edge_attr = torch.zeros((0, 1), device=device)

    @staticmethod
    def _random_sign(n, device):
        return torch.where(torch.rand(n, device=device) < 0.5, 1.0, -1.0)

    def _nonzero_integers(self, n, device):
        lb = int(self.lambda_bound)
        x = torch.randint(1, lb + 1, (n,), device=device).float()
        return x * self._random_sign(n, device)

    def forward(self, data: HeteroData) -> HeteroData:
        var_store, cons = data["variables"], data["constraints"]
        v2c, c2v = data[V2C], data[C2V]
        n_total = var_store.num_nodes
        dummy = n_total - 1
        n_cons = cons.num_nodes
        device = v2c.edge_index.device
        ei, ea = v2c.edge_index, v2c.edge_attr

        is_bin = var_store.is_binary.squeeze(-1).bool()
        is_int = var_store.is_integer.squeeze(-1).bool()

        # Candidate targets: real variables appearing in at least one constraint.
        has_edge = torch.zeros(n_total, dtype=torch.bool, device=device)
        if ei.numel():
            has_edge[ei[0]] = True
        cand = has_edge.clone()
        cand[dummy] = False
        cand_idx = cand.nonzero(as_tuple=True)[0]
        if cand_idx.numel() == 0 or self.k_frac <= 0:
            self._empty_store(data, device)
            return data

        k = min(
            max(1, round(self.k_frac * cand_idx.numel())), self.k_cap, cand_idx.numel()
        )
        targets = cand_idx[torch.randperm(cand_idx.numel(), device=device)[:k]]
        is_target = torch.zeros(n_total, dtype=torch.bool, device=device)
        is_target[targets] = True

        int_valued = is_int | is_bin
        int_valued[dummy] = True

        u_of = torch.ones(n_total, device=device)
        lam_of = torch.zeros(n_total, device=device)
        mu_of = torch.zeros(n_total, device=device)
        partner_of = torch.full((n_total,), dummy, dtype=torch.long, device=device)
        free_box = torch.zeros(n_total, dtype=torch.bool, device=device)

        # Sample (u, lambda, partner) per target
        bin_mask = is_bin[targets]
        if bin_mask.any():
            bt = targets[bin_mask]
            comp = torch.rand(bt.numel(), device=device) < 0.5
            u_of[bt] = torch.where(comp, -1.0, 1.0)
            lam_of[bt] = comp.float()

        non_bin_tgts = targets[~bin_mask]
        if non_bin_tgts.numel() > 0:
            is_int_nb = is_int[non_bin_tgts]
            elig = (~is_target).clone()
            elig[dummy] = False
            elig_cont_idx = elig.nonzero(as_tuple=True)[0]
            elig_int_idx = (elig & int_valued).nonzero(as_tuple=True)[0]
            n_elig_cont, n_elig_int = elig_cont_idx.numel(), elig_int_idx.numel()

            cont = non_bin_tgts[~is_int_nb]
            if cont.numel() and n_elig_cont > 0:
                partner_of[cont] = elig_cont_idx[
                    torch.randint(n_elig_cont, (cont.numel(),), device=device)
                ]
                free_box[cont] = True
            ints = non_bin_tgts[is_int_nb]
            if ints.numel() and n_elig_int > 0:
                partner_of[ints] = elig_int_idx[
                    torch.randint(n_elig_int, (ints.numel(),), device=device)
                ]
                free_box[ints] = True

            # Targets without an eligible partner are only scaled.
            has_lam_nb = ~(
                (~is_int_nb & (n_elig_cont == 0)) | (is_int_nb & (n_elig_int == 0))
            )
            if cont.numel():
                log_b = math.log(self.u_bound)
                mag = (
                    torch.empty(cont.numel(), device=device)
                    .uniform_(-log_b, log_b)
                    .exp_()
                )
                u_of[cont] = mag * self._random_sign(cont.numel(), device)
                lam_tgts = cont[has_lam_nb[~is_int_nb]]
                if lam_tgts.numel():
                    lam_of[lam_tgts] = torch.empty(
                        lam_tgts.numel(), device=device
                    ).uniform_(-self.lambda_bound, self.lambda_bound)
            if ints.numel():
                u_of[ints] = self._random_sign(ints.numel(), device)
                lam_tgts = ints[has_lam_nb[is_int_nb]]
                if lam_tgts.numel() and int(self.lambda_bound) > 0:
                    lam_of[lam_tgts] = self._nonzero_integers(lam_tgts.numel(), device)

        # Additive term for every target with a real partner.
        rp_idx = (is_target & (partner_of != dummy)).nonzero(as_tuple=True)[0]
        if rp_idx.numel():
            mu = torch.empty(rp_idx.numel(), device=device).uniform_(
                -self.lambda_bound, self.lambda_bound
            )
            rp_int = (is_int | is_bin)[rp_idx]
            if rp_int.any() and int(self.lambda_bound) > 0:
                mu[rp_int] = self._nonzero_integers(int(rp_int.sum()), device)
            mu_of[rp_idx] = mu

        # Column operations on the constraint matrix
        tgt_edge = is_target[ei[0]]
        tgt_src = ei[0][tgt_edge]
        lam_edge = lam_of[tgt_src]
        add_mask = lam_edge != 0
        add_src = partner_of[tgt_src][add_mask]  # col_l += lambda_j * col_j
        add_con = ei[1][tgt_edge][add_mask]
        add_attr = (lam_edge.unsqueeze(-1) * ea[tgt_edge])[add_mask]
        mu_edge = mu_of[tgt_src]
        mu_mask = mu_edge != 0
        if mu_mask.any():  # col_dummy += mu_j * col_j
            add_src = torch.cat(
                [
                    add_src,
                    torch.full(
                        (int(mu_mask.sum()),), dummy, dtype=add_src.dtype, device=device
                    ),
                ]
            )
            add_con = torch.cat([add_con, ei[1][tgt_edge][mu_mask]])
            add_attr = torch.cat(
                [add_attr, (mu_edge.unsqueeze(-1) * ea[tgt_edge])[mu_mask]]
            )
        scaled_attr = ea * u_of[ei[0]].unsqueeze(-1)  # col_j *= u_j

        # Finite boxes of real-partner targets become ranged rows
        has_lb = var_store.has_lb.squeeze(-1).bool()
        has_ub = var_store.has_ub.squeeze(-1).bool()
        lb_val = var_store.lb_val.squeeze(-1)
        ub_val = var_store.ub_val.squeeze(-1)

        fb_tgts = targets[free_box[targets]]
        box_tgts = fb_tgts[has_lb[fb_tgts] | has_ub[fb_tgts]]
        nb = box_tgts.numel()
        if nb > 0:
            new_con_idx = torch.arange(n_cons, n_cons + nb, device=device)
            b_var_t = torch.stack([box_tgts, partner_of[box_tgts]], dim=1).reshape(-1)
            b_con_t = new_con_idx.repeat_interleave(2)
            b_attr_t = torch.stack([u_of[box_tgts], lam_of[box_tgts]], dim=1).reshape(
                -1, 1
            )
            new_has_lhs_t = has_lb[box_tgts].float().unsqueeze(-1)
            new_has_rhs_t = has_ub[box_tgts].float().unsqueeze(-1)
            mu_b = mu_of[box_tgts]
            new_lhs_val_t = (
                (lb_val[box_tgts] - mu_b) * has_lb[box_tgts].float()
            ).unsqueeze(-1)
            new_rhs_val_t = (
                (ub_val[box_tgts] - mu_b) * has_ub[box_tgts].float()
            ).unsqueeze(-1)
        else:
            b_var_t, b_con_t = ei[0].new_empty(0), ei[1].new_empty(0)
            b_attr_t = scaled_attr.new_empty(0, 1)
            new_has_lhs_t = new_has_rhs_t = cons.has_lhs.new_empty(0, 1)
            new_lhs_val_t = new_rhs_val_t = cons.lhs_val.new_empty(0, 1)

        # Variable boxes
        # Dummy-partner targets keep an axis-aligned box: z_j = (x_j - lambda) / u.
        dummy_tgt = is_target & (partner_of == dummy)
        pos = u_of > 0
        src_lb_has = torch.where(pos, has_lb, has_ub)
        src_ub_has = torch.where(pos, has_ub, has_lb)
        new_lb_val = (torch.where(pos, lb_val, ub_val) - lam_of) / u_of
        new_ub_val = (torch.where(pos, ub_val, lb_val) - lam_of) / u_of
        out_has_lb, out_has_ub = has_lb.clone(), has_ub.clone()
        out_lb_val, out_ub_val = lb_val.clone(), ub_val.clone()
        out_has_lb[dummy_tgt] = src_lb_has[dummy_tgt]
        out_has_ub[dummy_tgt] = src_ub_has[dummy_tgt]
        out_lb_val[dummy_tgt] = new_lb_val[dummy_tgt] * src_lb_has[dummy_tgt].float()
        out_ub_val[dummy_tgt] = new_ub_val[dummy_tgt] * src_ub_has[dummy_tgt].float()
        out_has_lb[free_box] = False
        out_has_ub[free_box] = False
        out_lb_val[free_box] = 0.0
        out_ub_val[free_box] = 0.0
        var_store.has_lb = out_has_lb.float().unsqueeze(-1)
        var_store.has_ub = out_has_ub.float().unsqueeze(-1)
        var_store.lb_val = out_lb_val.unsqueeze(-1)
        var_store.ub_val = out_ub_val.unsqueeze(-1)

        # Merge added coefficients into existing edges
        if add_src.numel() > 0:
            add_key = add_src * n_cons + add_con
            cand_mask = torch.isin(ei[0], torch.unique(add_src))
            cand_idx_e = cand_mask.nonzero(as_tuple=True)[0]
            if cand_idx_e.numel() > 0:
                cand_key = ei[0][cand_idx_e] * n_cons + ei[1][cand_idx_e]
                cand_sort_idx = cand_key.argsort()
                sorted_cand_key = cand_key[cand_sort_idx]
                n_cand = cand_idx_e.numel()
                pos_a = torch.searchsorted(sorted_cand_key, add_key)
                matched = (pos_a < n_cand) & (
                    sorted_cand_key[pos_a.clamp(max=n_cand - 1)] == add_key
                )
                if matched.any():
                    orig_pos = cand_idx_e[cand_sort_idx[pos_a[matched]]]
                    scaled_attr.scatter_add_(
                        0,
                        orig_pos.unsqueeze(-1).expand(-1, scaled_attr.shape[1]),
                        add_attr[matched],
                    )
                extra_var, extra_con, extra_attr = (
                    add_src[~matched],
                    add_con[~matched],
                    add_attr[~matched],
                )
            else:
                extra_var, extra_con, extra_attr = add_src, add_con, add_attr
        else:
            extra_var, extra_con = ei[0].new_empty(0), ei[1].new_empty(0)
            extra_attr = scaled_attr.new_empty(0, 1)

        v2c.edge_index = torch.stack(
            [
                torch.cat([ei[0], extra_var, b_var_t]),
                torch.cat([ei[1], extra_con, b_con_t]),
            ]
        )
        v2c.edge_attr = torch.cat([scaled_attr, extra_attr, b_attr_t])
        c2v.edge_index = v2c.edge_index.flip(0)
        c2v.edge_attr = v2c.edge_attr.clone()

        cons.has_lhs = torch.cat([cons.has_lhs, new_has_lhs_t])
        cons.has_rhs = torch.cat([cons.has_rhs, new_has_rhs_t])
        cons.lhs_val = torch.cat([cons.lhs_val, new_lhs_val_t])
        cons.rhs_val = torch.cat([cons.rhs_val, new_rhs_val_t])
        cons.num_nodes = n_cons + nb

        # Objective: c_new = c_old . M
        obj = var_store.obj_coeffs.squeeze(-1)
        obj_new = obj * u_of
        obj_new.index_add_(0, partner_of[targets], lam_of[targets] * obj[targets])
        obj_new.index_add_(
            0, torch.full_like(targets, dummy), mu_of[targets] * obj[targets]
        )
        var_store.obj_coeffs = obj_new.unsqueeze(-1)

        # Record M - I
        tv = data[TRANSVECTION]
        ptn = lam_of[targets] != 0
        mu_t = mu_of[targets] != 0
        tv.edge_index = torch.stack(
            [
                torch.cat(
                    [
                        targets,
                        partner_of[targets][ptn],
                        torch.full_like(targets[mu_t], dummy),
                    ]
                ),
                torch.cat([targets, targets[ptn], targets[mu_t]]),
            ]
        )
        tv.edge_attr = torch.cat(
            [u_of[targets] - 1.0, lam_of[targets][ptn], mu_of[targets][mu_t]]
        ).unsqueeze(-1)
        return data
