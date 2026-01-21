# models/asgs_detr.py
# ------------------------------------------------------------------------
# Implementation of ASGS (ICCV 2025) based on SOMA code base
# ------------------------------------------------------------------------

import torch
import torch.nn.functional as F
from torch import nn
import math
import copy
import torch.distributed as dist

from util import box_ops
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized, inverse_sigmoid)

from .backbone import build_backbone
from .matcher import build_matcher
from .segmentation import (DETRsegm, PostProcessPanoptic, PostProcessSegm,
                           dice_loss, sigmoid_focal_loss)
from .deformable_transformer import build_deforamble_transformer
from .utils import GradientReversal


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class ASGS_DETR(nn.Module):
    """
    ASGS Model: Single-Domain Generalizable Open-Set Object Detection
    Based on Deformable DETR.
    """

    def __init__(self, backbone, transformer, num_classes, num_queries, num_feature_levels,
                 aux_loss=True, with_box_refine=False, two_stage=False, asgs_cfg=None):
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model

        # Background는 Sigmoid Focal Loss에서 모든 class logit이 0인 경우로 처리됩니다.
        # num_classes(4) = Known(3) + Unknown(1)
        self.num_classes = num_classes
        self.class_embed = nn.Linear(hidden_dim, num_classes)

        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.num_feature_levels = num_feature_levels

        # ASGS Configs
        self.asgs_cfg = asgs_cfg

        # Class Prototypes (Centers) and Stds initialization
        # [Known Classes]
        self.register_buffer('cls_means', torch.zeros(num_classes, hidden_dim))

        if not two_stage:
            self.query_embed = nn.Embedding(num_queries, hidden_dim * 2)

        # ... (Input Projections and Backbone setup - Same as Deformable DETR) ...
        if num_feature_levels > 1:
            num_backbone_outs = len(backbone.strides)
            input_proj_list = []
            for _ in range(num_backbone_outs):
                in_channels = backbone.num_channels[_]
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
            for _ in range(num_feature_levels - num_backbone_outs):
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
                in_channels = hidden_dim
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            self.input_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(backbone.num_channels[0], hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                )])
        self.backbone = backbone
        self.aux_loss = aux_loss
        self.with_box_refine = with_box_refine
        self.two_stage = two_stage

        # Init parameters
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(num_classes) * bias_value
        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

        num_pred = (transformer.decoder.num_layers + 1) if two_stage else transformer.decoder.num_layers
        if with_box_refine:
            self.class_embed = _get_clones(self.class_embed, num_pred)
            self.bbox_embed = _get_clones(self.bbox_embed, num_pred)
            nn.init.constant_(self.bbox_embed[0].layers[-1].bias.data[2:], -2.0)
            self.transformer.decoder.bbox_embed = self.bbox_embed
        else:
            nn.init.constant_(self.bbox_embed.layers[-1].bias.data[2:], -2.0)
            self.class_embed = nn.ModuleList([self.class_embed for _ in range(num_pred)])
            self.bbox_embed = nn.ModuleList([self.bbox_embed for _ in range(num_pred)])
            self.transformer.decoder.bbox_embed = None

        if two_stage:
            self.transformer.decoder.class_embed = self.class_embed
            for box_embed in self.bbox_embed:
                nn.init.constant_(box_embed.layers[-1].bias.data[2:], 0.0)

    def forward(self, samples: NestedTensor):
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.backbone(samples)

        srcs = []
        masks = []
        for l, feat in enumerate(features):
            src, mask = feat.decompose()
            srcs.append(self.input_proj[l](src))
            masks.append(mask)
            assert mask is not None
        if self.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    src = self.input_proj[l](features[-1].tensors)
                else:
                    src = self.input_proj[l](srcs[-1])
                m = samples.mask
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                pos.append(pos_l)

        query_embeds = None
        if not self.two_stage:
            query_embeds = self.query_embed.weight

        # Transformer forward
        hs, init_reference, inter_references, enc_outputs_class, enc_outputs_coord_unact, _, _ = \
            self.transformer(srcs, masks, pos, query_embeds)

        outputs_classes = []
        outputs_coords = []
        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            outputs_class = self.class_embed[lvl](hs[lvl])
            tmp = self.bbox_embed[lvl](hs[lvl])
            if reference.shape[-1] == 4:
                tmp += reference
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference
            outputs_coord = tmp.sigmoid()
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)
        outputs_class = torch.stack(outputs_classes)
        outputs_coord = torch.stack(outputs_coords)

        # Output preparation
        out = {
            'pred_logits': outputs_class[-1],
            'pred_boxes': outputs_coord[-1],
            'object_embedding': hs[-1],  # Last layer embeddings needed for ASGS
            'cls_means': self.cls_means,  # For ASGS Prototype access
            # [추가됨] Loss 계산을 위해 마지막 분류기 레이어를 전달해야 합니다.
            'final_classifier': self.class_embed[-1]
        }

        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)

        if self.two_stage:
            enc_outputs_coord = enc_outputs_coord_unact.sigmoid()
            out['enc_outputs'] = {'pred_logits': enc_outputs_class, 'pred_boxes': enc_outputs_coord}

        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        return [{'pred_logits': a, 'pred_boxes': b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]


class ASGSCriterion(nn.Module):
    """
    ASGS Loss Criterion including:
    1. Standard DETR Losses (Labels, Boxes)
    2. SUL Loss (Subgraph-wise Unknown-class Learning)
    3. CEC Loss (Class-wise Embedding Compaction)
    """

    def __init__(self, num_classes, matcher, weight_dict, losses, focal_alpha=0.25, asgs_cfg=None):
        super().__init__()
        self.num_classes = num_classes  # num_classes(4) = Known(3) + Unknown(1)
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.focal_alpha = focal_alpha

        # ASGS Hyperparameters
        self.asgs_cfg = asgs_cfg
        self.alpha_proto = self.asgs_cfg.get('alpha', 0.9)  # EMA factor (Paper Eq 1)
        self.K_boundary = self.asgs_cfg.get('K_boundary', 5)  # Number of boundary samples
        self.M_knn = self.asgs_cfg.get('M_knn', 5)  # Number of KNN unmatched samples
        self.delta_sim = self.asgs_cfg.get('delta', 0.6)  # Similarity threshold for ASS
        print(f"✅ ASGS Criterion Initialized with Delta (TH): {self.delta_sim}")
        self.tau_cec = self.asgs_cfg.get('tau', 0.1)  # Temperature for CEC
        # [수정 전]
        # self.unknown_idx = num_classes  <-- (4가 들어감: 배경 인덱스가 됨)

        # [수정 후]
        self.num_known_classes = num_classes - 1 # num_classes(4) -1 = 3
        self.unknown_idx = num_classes - 1 # num_classes(4) -1 = 3
        # [추가] 디버깅용 카운터
        self.debug_counter = 0

    def forward(self, outputs, targets, epoch=0):
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs' and k != 'enc_outputs'}

        # 1. Hungarian Matching
        indices = self.matcher(outputs_without_aux, targets)

        # 2. Update Prototypes (Methodology 3.2 Eq 1)
        if self.training:
            self.update_prototypes(outputs, targets, indices)

        # 3. Compute Standard Losses
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=outputs['pred_logits'].device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))

        # 4. Compute ASGS Specific Losses (SUL & CEC)
        # if self.training:
        #     # SUL Loss
        #     if 'loss_sul' in self.weight_dict:
        #         losses.update(self.get_sul_loss(outputs, targets, indices))
        #
        #     # CEC Loss
        #     if 'loss_cec' in self.weight_dict:
        #         losses.update(self.get_cec_loss(outputs, targets, indices))
        # 4. Compute ASGS Specific Losses (SUL & CEC)
        if self.training:
            # Config에서 WARM_UP 값 가져오기 (기본값 9)
            warm_up_epoch = self.asgs_cfg.get('WARM_UP', 9)

            # SUL Loss (Warm-up 적용: 설정된 Epoch 이후부터 계산)
            if 'loss_sul' in self.weight_dict:
                if epoch >= warm_up_epoch:
                    #print("warm_up epoch finish. let's begin sul loss")
                    losses.update(self.get_sul_loss(outputs, targets, indices))
                else:
                    # Warm-up 기간에는 Loss를 0으로 설정 (로그 기록용)
                    losses['loss_sul'] = torch.tensor(0.0, device=outputs['pred_logits'].device)

            # CEC Loss (보통 CEC는 Warm-up 없이 바로 학습하거나 논문에 따라 다름)
            # 여기서는 SUL과 동일하게 Warm-up을 적용할지 선택해야 합니다.
            # 논문에서 "SUL modules... after warm-up"이라고만 했다면 CEC는 놔두셔도 됩니다.
            if 'loss_cec' in self.weight_dict:
                losses.update(self.get_cec_loss(outputs, targets, indices))

        # Aux losses ... (omitted for brevity, same as standard)
        return losses

    @torch.no_grad()
    def update_prototypes(self, outputs, targets, indices):
        """ Update Class Prototypes using EMA (Eq 1) with DDP Sync """
        cls_means = outputs['cls_means']
        obj_embs = outputs['object_embedding']

        batch_idx, src_idx = self._get_src_permutation_idx(indices)
        matched_embs = obj_embs[batch_idx, src_idx]
        target_labels = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])

        for k in range(self.num_known_classes):
            # 1. 현재 GPU에서 해당 클래스(k)의 Feature 찾기
            k_embs = matched_embs[target_labels == k]

            # 2. 합계(Sum)와 개수(Count) 계산
            if k_embs.numel() > 0:
                local_sum = k_embs.sum(dim=0)
                local_count = torch.tensor(float(k_embs.size(0)), device=k_embs.device)
            else:
                local_sum = torch.zeros_like(cls_means[k])
                local_count = torch.tensor(0.0, device=k_embs.device)

            # 3. [DDP 핵심] 모든 GPU의 값을 모음 (All-Reduce)
            if is_dist_avail_and_initialized():
                dist.all_reduce(local_sum)  # 모든 GPU의 Feature 합
                dist.all_reduce(local_count)  # 모든 GPU의 샘플 개수 합

            # 4. 전역 평균(Global Mean) 계산 및 업데이트
            if local_count > 0:
                global_mean = local_sum / local_count
                global_mean = F.normalize(global_mean, p=2, dim=0)  # 정규화

                # EMA 업데이트
                updated_proto = self.alpha_proto * cls_means[k] + (1 - self.alpha_proto) * global_mean
                updated_proto = F.normalize(updated_proto, p=2, dim=0)
                cls_means[k] = updated_proto.detach()

        outputs['cls_means'] = cls_means

    def get_sul_loss(self, outputs, targets, indices):
        """ Subgraph-wise Unknown-class Learning (Eq 3) """
        obj_embs = outputs['object_embedding']  # [B, 100, D] (Raw Features)
        prototypes = outputs['cls_means']  # [K, D] (Normalized in update_prototypes)
        batch_idx, src_idx = self._get_src_permutation_idx(indices)

        subgraph_features = []

        # 디버깅: Subgraph 생성 개수 모니터링을 위한 리스트
        debug_subgraph_counts = []

        for b in range(len(targets)):
            src_idx_b = src_idx[batch_idx == b]
            tgt_labels_b = targets[b]['labels'][indices[b][1]]

            matched_q = obj_embs[b, src_idx_b]

            is_matched = torch.zeros(obj_embs.shape[1], dtype=torch.bool, device=obj_embs.device)
            is_matched[src_idx_b] = True
            unmatched_q = obj_embs[b, ~is_matched]

            if unmatched_q.size(0) == 0 or matched_q.size(0) == 0:
                continue

            unique_classes = tgt_labels_b.unique()
            for k in unique_classes:
                q_m_k = matched_q[tgt_labels_b == k]
                proto_k = prototypes[k]

                # 1. Boundary 선정: "방향(Cosine)" 기준
                q_m_k_norm = F.normalize(q_m_k, p=2, dim=1)  # 정규화

                # Cosine Distance 사용 (유사도가 낮을수록 Boundary)
                # proto_k는 이미 정규화되어 있음
                cosine_sims = torch.mm(q_m_k_norm, proto_k.unsqueeze(1)).squeeze(1)
                cosine_dists = 1 - cosine_sims

                K = min(self.K_boundary, len(cosine_dists))
                _, bound_indices = torch.topk(cosine_dists, K)

                # [중요] Subgraph 구성용 Feature는 "원본(Raw)"을 사용해야 함!
                # Classifier가 Raw Scale을 기대하기 때문
                boundary_samples = q_m_k[bound_indices]

                # 2. KNN을 위한 정규화 (선택된 Boundary만 정규화)
                # 여기서 q_m_k_norm[bound_indices]를 써도 되지만,
                # 코드 가독성과 안전성을 위해 boundary_samples를 정규화하는 것이 명확함
                boundary_norm = F.normalize(boundary_samples, p=2, dim=1)
                unmatched_norm = F.normalize(unmatched_q, p=2, dim=1)

                sim_matrix = torch.mm(boundary_norm, unmatched_norm.t())

                for i in range(len(boundary_samples)):
                    M = min(self.M_knn, unmatched_q.size(0))
                    sim_vals, top_m_idx = torch.topk(sim_matrix[i], M)

                    # [개선] 디버그 출력 효율화 (1000번에 1번만 출력)
                    self.debug_counter += 1
                    if self.debug_counter % 1000 == 0:
                        print(f"[DEBUG] Max Similarity: {sim_vals.max().item():.4f} (Threshold: {self.delta_sim})")

                    valid_mask = sim_vals > self.delta_sim
                    valid_indices = top_m_idx[valid_mask]

                    if len(valid_indices) > 0:
                        # Subgraph 구성 (Raw Features 사용)
                        nodes = torch.cat([boundary_samples[i].unsqueeze(0), unmatched_q[valid_indices]], dim=0)
                        g_bar = nodes.mean(dim=0)
                        subgraph_features.append(g_bar)

        # [개선] Subgraph 생성 실패 로깅
        if len(subgraph_features) == 0:
            # 너무 자주 출력되지 않도록 확률적 또는 카운터로 조절
            if self.debug_counter % 1000 == 0:
                print(f"[SUL WARNING] No subgraphs created! (Delta={self.delta_sim})")
            return {'loss_sul': torch.tensor(0.0, device=obj_embs.device)}

        subgraph_features = torch.stack(subgraph_features)

        # Classifier Prediction
        classifier = outputs.get('final_classifier')
        if classifier is None:
            return {'loss_sul': torch.tensor(0.0, device=obj_embs.device)}

        # pred_logits = classifier(subgraph_features)
        #
        # # Target: Unknown Class
        # target_labels = torch.full((len(subgraph_features),), self.unknown_idx,
        #                            dtype=torch.long, device=pred_logits.device)
        #
        # loss_sul = F.cross_entropy(pred_logits, target_labels)
        #
        # return {'loss_sul': loss_sul}

        pred_logits = classifier(subgraph_features)
        # NOTE:
        # Baseline classification uses sigmoid focal loss (multi-label style) :contentReference[oaicite:2]{index=2}
        # To keep objectives consistent, SUL is also implemented with sigmoid focal loss:
        # target is one-hot where only unknown index is 1.
        target_onehot = torch.zeros((pred_logits.shape[0], pred_logits.shape[1]), dtype=pred_logits.dtype, device=pred_logits.device)
        target_onehot[:, self.unknown_idx] = 1.0
        # normalize by number of subgraphs (num_boxes 역할)
        num_subg = max(int(pred_logits.shape[0]), 1)
        loss_sul = sigmoid_focal_loss(pred_logits, target_onehot, num_subg, alpha = self.focal_alpha, gamma = 2)

        return {'loss_sul': loss_sul}

    def get_cec_loss(self, outputs, targets, indices):
        """ Class-wise Embedding Compaction (Eq 5) - InfoNCE """
        obj_embs = outputs['object_embedding']
        prototypes = outputs['cls_means']
        batch_idx, src_idx = self._get_src_permutation_idx(indices)
        matched_embs = obj_embs[batch_idx, src_idx]  # [Total_Matched, D]
        matched_labels = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])

        # [수정 1] 수치 안정을 위해 모든 Embedding을 미리 정규화(Normalize) 합니다. (Cosine Similarity용)
        # prototypes는 update_prototypes에서 이미 정규화되어 있지만, 안전을 위해 여기서도 확인 가능
        # matched_embs = F.normalize(matched_embs, p=2, dim=1)
        # prototypes = F.normalize(prototypes, p=2, dim=1)
        def _safe_l2_normalize(x, dim=1, eps=1e-6):
            return x / (x.norm(p=2, dim=dim, keepdim=True).clamp_min(eps))

        # normalize embeddings for cosine sim
        matched_embs = _safe_l2_normalize(matched_embs, dim=1, eps=1e-6)

        # IMPORTANT:
        # cls_means buffer starts from zeros :contentReference[oaicite:3]{index=3} and only known-class prototypes are updated :contentReference[oaicite:4]{index=4}.
        # Normalizing a zero vector can introduce NaNs depending on torch version.
        # So we normalize safely and also restrict to known prototypes.
        prototypes = _safe_l2_normalize(prototypes, dim=1, eps=1e-6)

        loss_cec = 0.0
        count = 0

        # Iterate over each known class k
        # [수정 전]
        # for k in range(self.num_classes): <-- (Unknown인 3번까지 돎)

        # [수정 후] Known Class 개수(3개)만큼만 반복 (0, 1, 2)
        for k in range(self.num_known_classes):
            # Query: Class Prototype mu_s^k
            query = prototypes[k]  # [D]

            # Positive Keys: Matched embeddings of class k
            pos_mask = (matched_labels == k)
            if not pos_mask.any():
                continue
            positive_keys = matched_embs[pos_mask]  # [N_pos, D]

            # Negative Keys:
            # 1. Prototypes of other classes
            # 2. Matched embeddings of other classes

            # neg_proto_mask = torch.arange(self.num_classes, device=prototypes.device) != k
            # negative_protos = prototypes[neg_proto_mask]
            neg_proto_mask = torch.arange(self.num_known_classes, device=prototypes.device) != k
            negative_protos = prototypes[:self.num_known_classes][neg_proto_mask]

            neg_emb_mask = ~pos_mask
            negative_embs = matched_embs[neg_emb_mask]

            negative_keys = torch.cat([negative_protos, negative_embs], dim=0)  # [N_neg, D]

            if negative_keys.size(0) == 0:
                continue

            # Calculate InfoNCE
            # [수정 2] 이미 정규화했으므로 MatMul 결과는 코사인 유사도가 됩니다. (범위 -1 ~ 1)
            # tau(0.1)로 나누어도 최대 10, 최소 -10 이므로 exp가 발산하지 않습니다.

            # Similarity with Positives
            sim_pos = torch.matmul(positive_keys, query) / self.tau_cec  # [N_pos]

            # Similarity with Negatives
            sim_neg = torch.matmul(negative_keys, query) / self.tau_cec  # [N_neg]

            # [수정 3] LogSumExp 트릭을 사용하여 수치 안정성 확보 (선택 사항이나 권장)
            # 기존: -log( exp(pos) / (exp(pos) + sum(exp(neg))) )
            # 변형: log(exp(pos) + sum(exp(neg))) - pos

            # 여기서는 간단히 정규화만으로도 NaN은 해결되므로 기존 수식 유지하되,
            # 분모 계산 시 max trick 등을 쓰는 것이 안전하지만, 코사인 유사도 범위 내에서는 아래도 괜찮습니다.

            # exp_neg_sum = torch.sum(torch.exp(sim_neg))
            #
            # # 분모에 아주 작은 epsilon을 더해 0이 되는 것을 방지
            # denominator = torch.exp(sim_pos) + exp_neg_sum + 1e-8
            #
            # loss_k = -torch.log(torch.exp(sim_pos) / denominator)
            # Stable InfoNCE with logsumexp:
            # loss_i = -sim_pos_i + logsumexp([sim_pos_i, sim_neg_1..N])
            # Vectorized over positives.
            if sim_neg.numel() > 0:
                sim_neg_expand = sim_neg.unsqueeze(0).expand(sim_pos.size(0), -1)  # [N_pos, N_neg]
                logits_all = torch.cat([sim_pos.unsqueeze(1), sim_neg_expand], dim=1)  # [N_pos, 1+N_neg]
            else:
                logits_all = sim_pos.unsqueeze(1)  # [N_pos, 1]
            loss_k = -sim_pos + torch.logsumexp(logits_all, dim=1)

            loss_cec += loss_k.sum()
            count += len(loss_k)




            loss_cec += loss_k.sum()
            count += len(loss_k)

        if count > 0:
            loss_cec = loss_cec / count
        else:
            loss_cec = torch.tensor(0.0, device=obj_embs.device)

        return {'loss_cec': loss_cec}

    # ... (Helpers like _get_src_permutation_idx, loss_labels, etc. from standard DETR) ...
    def _get_src_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        # Implementation of standard losses (labels, boxes) needs to be here
        # Copied from SOMA/DETR standard implementation
        loss_map = {
            'labels': self.loss_labels,
            'boxes': self.loss_boxes,
        }
        if loss in loss_map:
            return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)
        return {}

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """
        Classification Loss (Sigmoid Focal Loss)
        - 배경(Background)은 [0, 0, ..., 0] 벡터가 되도록 학습합니다.
        - GT에는 Unknown 라벨이 없으므로, Unknown 클래스(마지막 인덱스)도
          이 단계에서는 모두 0(Negative)으로 학습되어 억제됩니다.
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']  # Shape: [Batch, Query, num_classes]

        idx = self._get_src_permutation_idx(indices)

        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])

        # [1] 배경 인덱스 설정
        # src_logits의 채널 수가 4개(0~3)라면, 배경 인덱스는 4가 됩니다.
        bg_class_ind = src_logits.shape[2]

        # [2] 모든 타겟을 배경(4)으로 초기화
        target_classes = torch.full(src_logits.shape[:2], bg_class_ind,
                                    dtype=torch.int64, device=src_logits.device)

        # [3] 매칭된 쿼리에만 실제 라벨(Known Class ID) 할당
        # 주의: GT에는 Unknown(3)이 없으므로, 여기에는 0, 1, 2만 들어갑니다.
        target_classes[idx] = target_classes_o

        # [4] One-hot Encoding
        # 배경(4)을 표현하기 위해 잠시 차원을 +1 하여 [Batch, Query, 5]로 만듭니다.
        target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1],
                                            dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)

        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)

        # [5] 마지막 차원(배경 인덱스) 제거
        # 결과적으로 배경이었던 쿼리는 [0, 0, 0, 0]이 됩니다.
        target_classes_onehot = target_classes_onehot[:, :, :-1]

        # [6] Focal Loss 계산
        # 기존 Baseline에 있던 unk_prob 관련 로직은 모두 삭제했습니다.
        loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_boxes, alpha=self.focal_alpha, gamma=2) * \
                  src_logits.shape[1]

        losses = {'loss_ce': loss_ce}

        if log:
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]

        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
        losses = {'loss_bbox': loss_bbox.sum() / num_boxes}
        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses


# models/asgs_detr.py 내부에 추가

class PostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api """

    @torch.no_grad()
    def forward(self, outputs, target_sizes, show_box=False):  # show_box 인자 추가
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        prob = out_logits.sigmoid()
        # Top-100 predictions selection
        topk_values, topk_indexes = torch.topk(prob.view(out_logits.shape[0], -1), 100, dim=1)
        scores = topk_values
        topk_boxes = topk_indexes // out_logits.shape[2]
        labels = topk_indexes % out_logits.shape[2]

        # Box coordinate conversion
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, 4))

        # Relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]
        return results

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def build(cfg):
    device = torch.device(cfg.DEVICE)
    backbone = build_backbone(cfg)
    transformer = build_deforamble_transformer(cfg)

    # ASGS Configuration
    asgs_cfg = {
        'K_boundary': cfg.AOOD.ASGS.K,
        'M_knn': cfg.AOOD.ASGS.M,
        'delta': cfg.AOOD.ASGS.DELTA,
        'alpha': cfg.AOOD.ASGS.ALPHA,
        'tau': cfg.AOOD.ASGS.TAU,
        'WARM_UP': cfg.AOOD.ASGS.WARM_UP  # [추가] Config에서 값(9)을 가져와 전달
    }

    model = ASGS_DETR(
        backbone,
        transformer,
        num_classes=cfg.DATASET.NUM_CLASSES,
        num_queries=cfg.MODEL.NUM_QUERIES,
        num_feature_levels=cfg.MODEL.NUM_FEATURE_LEVELS,
        aux_loss=cfg.LOSS.AUX_LOSS,
        with_box_refine=cfg.MODEL.WITH_BOX_REFINE,
        two_stage=cfg.MODEL.TWO_STAGE,
        asgs_cfg=asgs_cfg
    )

    matcher = build_matcher(cfg)

    # Loss Weights
    weight_dict = {
        'loss_ce': cfg.LOSS.CLS_LOSS_COEF,
        'loss_bbox': cfg.LOSS.BBOX_LOSS_COEF,
        'loss_giou': cfg.LOSS.GIOU_LOSS_COEF,
        'loss_sul': cfg.AOOD.ASGS.LAMBDA_SUL,  # Lambda 1 in paper
        'loss_cec': cfg.AOOD.ASGS.LAMBDA_CEC,  # Lambda 2 in paper
    }

    if cfg.LOSS.AUX_LOSS:
        aux_weight_dict = {}
        for i in range(cfg.MODEL.DEC_LAYERS - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    losses = ['labels', 'boxes']

    criterion = ASGSCriterion(
        cfg.DATASET.NUM_CLASSES,
        matcher,
        weight_dict,
        losses,
        focal_alpha=cfg.LOSS.FOCAL_ALPHA,
        asgs_cfg=asgs_cfg
    )
    criterion.to(device)

    # [수정됨] bbox 처리를 위한 PostProcess 클래스 사용
    postprocessors = {'bbox': PostProcess()}

    if cfg.MODEL.MASKS:
        postprocessors['segm'] = PostProcessSegm()
        # 필요한 경우 panoptic 추가 로직...

    return model, criterion, postprocessors