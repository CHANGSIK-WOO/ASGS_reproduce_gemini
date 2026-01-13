# models/asgs_detr.py
# ------------------------------------------------------------------------
# Implementation of ASGS (ICCV 2025) based on SOMA code base
# ------------------------------------------------------------------------

import torch
import torch.nn.functional as F
from torch import nn
import math
import copy

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

        # ASGS Note: Unknown class를 위한 추가적인 output dimension이 필요할 수 있습니다.
        # 여기서는 num_classes가 Known Class의 개수라고 가정하고, +1을 더해 Unknown을 표현합니다.
        # Background는 Sigmoid Focal Loss에서 모든 class logit이 0인 경우로 처리됩니다.
        self.num_classes = num_classes
        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)  # +1 for Unknown Class

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
        self.class_embed.bias.data = torch.ones(num_classes + 1) * bias_value
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
        self.num_classes = num_classes  # Known classes count
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.focal_alpha = focal_alpha

        # ASGS Hyperparameters
        self.asgs_cfg = asgs_cfg
        self.alpha_proto = 0.9  # EMA factor (Paper Eq 1)
        self.K_boundary = 5  # Number of boundary samples
        self.M_knn = 5  # Number of KNN unmatched samples
        self.delta_sim = 0.6  # Similarity threshold for ASS
        self.tau_cec = 0.1  # Temperature for CEC
        # [수정 전]
        # self.unknown_idx = num_classes  <-- (4가 들어감: 배경 인덱스가 됨)

        # [수정 후]
        # 1. Unknown 인덱스를 3번으로 당김 (0, 1, 2, 3)
        self.unknown_idx = num_classes - 1

        # 2. Known Class 개수 정의 (0, 1, 2 세 개만 Known)
        self.num_known_classes = num_classes - 1


    def forward(self, samples, outputs, targets, epoch=0):
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
        """ Update Class Prototypes using EMA (Eq 1) """
        cls_means = outputs['cls_means']  # Buffer [num_classes, dim]
        obj_embs = outputs['object_embedding']  # [B, N, dim]

        # Gather all matched embeddings and their labels
        batch_idx, src_idx = self._get_src_permutation_idx(indices)
        matched_embs = obj_embs[batch_idx, src_idx]
        target_labels = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])

        for k in range(self.num_classes):
            # Select embeddings belonging to class k
            k_embs = matched_embs[target_labels == k]
            if k_embs.numel() > 0:
                # Calculate Mean
                k_mean = k_embs.mean(dim=0)
                # Normalize (As per Eq 1: Normalize(alpha*mu + (1-alpha)*Mean))
                # Note: The equation order in paper usually implies updating the vector then normalizing
                updated_proto = self.alpha_proto * cls_means[k] + (1 - self.alpha_proto) * k_mean
                updated_proto = F.normalize(updated_proto, p=2, dim=0)
                cls_means[k] = updated_proto.detach()

        outputs['cls_means'] = cls_means  # Update in place

    def get_sul_loss(self, outputs, targets, indices):
        """ Subgraph-wise Unknown-class Learning (Eq 3) """
        # Data preparation
        obj_embs = outputs['object_embedding']  # [B, 100, D]
        prototypes = outputs['cls_means']  # [K, D]
        batch_idx, src_idx = self._get_src_permutation_idx(indices)

        subgraph_features = []

        # Iterate over batch to handle queries per image
        for b in range(len(targets)):
            # 1. Separate Matched (Known) and Unmatched (Potential Unknown)
            # Indices for this batch element
            src_idx_b = src_idx[batch_idx == b]
            tgt_labels_b = targets[b]['labels'][indices[b][1]]

            matched_q = obj_embs[b, src_idx_b]

            # Unmatched indices: All indices excluding src_idx_b
            all_indices = torch.arange(obj_embs.shape[1], device=obj_embs.device)
            # Create a mask for unmatched
            is_matched = torch.zeros(obj_embs.shape[1], dtype=torch.bool, device=obj_embs.device)
            is_matched[src_idx_b] = True
            unmatched_q = obj_embs[b, ~is_matched]  # Q_um

            if unmatched_q.size(0) == 0 or matched_q.size(0) == 0:
                continue

            # 2. Iterate per Known Class present in this image
            unique_classes = tgt_labels_b.unique()
            for k in unique_classes:
                # Class specific matched queries
                q_m_k = matched_q[tgt_labels_b == k]  # Q_m^k
                proto_k = prototypes[k]

                # Calculate distance to prototype
                dists = torch.norm(q_m_k - proto_k, dim=1)

                # Select Boundary Samples (Top K farthest)
                K = min(self.K_boundary, len(dists))
                _, bound_indices = torch.topk(dists, K)
                boundary_samples = q_m_k[bound_indices]  # [K, D]

                # 3. KNN & ASS for each boundary sample
                # Normalize for Cosine Sim (Eq 2 uses L2 distance on normalized features which is related to CosSim)
                boundary_norm = F.normalize(boundary_samples, p=2, dim=1)
                unmatched_norm = F.normalize(unmatched_q, p=2, dim=1)

                # Compute Cosine Similarity matrix [K, N_unmatched]
                sim_matrix = torch.mm(boundary_norm, unmatched_norm.t())

                for i in range(len(boundary_samples)):
                    # KNN: Find top M nearest unmatched samples
                    M = min(self.M_knn, unmatched_q.size(0))
                    # Note: Paper says "nearest" by L2 dist. Max Cosine Sim is equivalent for normalized vectors.
                    sim_vals, top_m_idx = torch.topk(sim_matrix[i], M)
                    # --- [디버깅용 출력 추가] ---
                    # 가장 높은 유사도가 얼마인지 확인 (너무 자주 출력되면 귀찮으니 가끔 출력)
                    if torch.rand(1).item() < 0.01:
                        print(f"[DEBUG] Max Similarity: {sim_vals.max().item():.4f} (Threshold: {self.delta_sim})")
                    # ------------------------

                    # Adaptive Subgraph Searching (ASS)
                    # Connect if similarity > delta (Paper Eq below 2)
                    valid_mask = sim_vals > self.delta_sim
                    valid_indices = top_m_idx[valid_mask]

                    if len(valid_indices) > 0:
                        # Construct Subgraph
                        # Subgraph nodes: Boundary Sample + Valid Unmatched Neighbors
                        nodes = torch.cat([boundary_samples[i].unsqueeze(0), unmatched_q[valid_indices]], dim=0)

                        # Mean Representation (G_bar)
                        g_bar = nodes.mean(dim=0)
                        subgraph_features.append(g_bar)

        if len(subgraph_features) == 0:
            return {'loss_sul': torch.tensor(0.0, device=obj_embs.device)}

        # 4. Compute Loss (Eq 3)
        # Classify subgraph mean as "Unknown" class
        subgraph_features = torch.stack(subgraph_features)  # [N_subgraphs, D]

        # Classifier Prediction
        # Note: self.class_embed has weights [num_classes+1, D]
        # We assume the last class index (self.num_classes) represents "Unknown"
        # Since we are using an external criterion, we might need access to the classifier weights.
        # Ideally, pass the classifier in init or outputs.
        # Here we assume outputs has 'pred_logits' produced by the same linear layer,
        # but we need to pass G_bar through it.
        # Let's assume we can access the linear layer from the model or pass it to criterion.
        # HACK: Using functional linear with weights from 'final_classifier' if stored in outputs by model

        # Assuming ASGS_DETR stores the classifier weight in outputs just like SOMA did?
        # SOMA did: out['final_classifier'] = self.class_embed[-1]
        # We will assume ASGS_DETR does the same or we pass it.

        # Let's assume we modified ASGS_DETR to output the classifier layer or weights
        # Or simpler: access via stored weights if accessible.
        # For this snippet, I'll calculate logits using a placeholder variable `classifier_weights`
        # In integration, ensure `outputs['final_classifier']` is available.
        classifier = outputs.get('final_classifier')
        if classifier is None:
            # Fail-safe or raise error
            print("classifier is None")
            return {'loss_sul': torch.tensor(0.0, device=obj_embs.device)}

        pred_logits = classifier(subgraph_features)  # [N_subgraphs, num_classes + 1]

        # Target is Unknown Class (index = self.unknown_idx)
        # We create a target tensor
        target_labels = torch.full((len(subgraph_features),), self.unknown_idx,
                                   dtype=torch.long, device=pred_logits.device)

        # Standard Cross Entropy or Focal Loss
        # Paper implies log likelihood, effectively CE
        loss_sul = F.cross_entropy(pred_logits, target_labels)

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
        matched_embs = F.normalize(matched_embs, p=2, dim=1)
        prototypes = F.normalize(prototypes, p=2, dim=1)

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
            neg_proto_mask = torch.arange(self.num_classes, device=prototypes.device) != k
            negative_protos = prototypes[neg_proto_mask]

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

            exp_neg_sum = torch.sum(torch.exp(sim_neg))

            # 분모에 아주 작은 epsilon을 더해 0이 되는 것을 방지
            denominator = torch.exp(sim_pos) + exp_neg_sum + 1e-8

            loss_k = -torch.log(torch.exp(sim_pos) / denominator)

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
        Classification Loss (Focal Loss) 계산
        ASGS 수정: 배경(Background) 쿼리에 대해서는 Unknown Class의 Loss를 무시(Ignore)하여
        loss_sul과의 충돌을 방지함.
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
            src_logits = outputs['pred_logits']
            idx = self._get_src_permutation_idx(indices)
            target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])

            # Initialize with Background Class (typically self.num_classes + 1 in logic,
            # but Focal Loss expects one-hot with last dim removed for background)
            # Here we follow SOMA:
            # valid classes: 0 to num_classes (including Unknown)
            # target_classes filled with num_classes + 1 (Background)

            # Note: In ASGS_DETR, we defined num_classes output as known + 1.
            # So known indices: 0..N-1. Unknown index: N.
            # We assume dataset targets only contain known labels (0..N-1).

            target_classes = torch.full(src_logits.shape[:2], self.num_classes + 1,
                                        dtype=torch.int64, device=src_logits.device)
            target_classes[idx] = target_classes_o

            target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1],
                                                dtype=src_logits.dtype, layout=src_logits.layout,
                                                device=src_logits.device)
            target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)
            target_classes_onehot = target_classes_onehot[:, :, :-1]  # Remove extra background slot

            loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_boxes, alpha=self.focal_alpha,
                                         gamma=2) * \
                      src_logits.shape[1]
            losses = {'loss_ce': loss_ce}
            # --- [추가해야 할 부분] ---
            if log:
                # 매칭된 쿼리에 대해서 분류 정확도를 계산하여 class_error로 기록
                losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
            # ------------------------
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
        'K_boundary': 5,
        'M_knn': 5,
        'delta': 0.6,
        'alpha': 0.9,
        'tau': 0.1,
        'WARM_UP': cfg.AOOD.OPEN_SET.WARM_UP  # [추가] Config에서 값(9)을 가져와 전달
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
        'loss_sul': 1.0,  # Lambda 1 in paper
        'loss_cec': 0.1  # Lambda 2 in paper
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