import torch
from torch import Tensor
from mmrotate.registry import MODELS
from mmdet.models.utils import levels_to_images
from .oriented_ddq_fcn import OrientedDDQFCN, align_tensor


@MODELS.register_module()
class OrientedAdaMixerDDQ(OrientedDDQFCN):
    """Oriented AdaMixer RPN head for ."""

    def construct_query(self, feat_dict: dict, scores: list,
                        bboxes: list, query_ids: list, num: int=300):
        '''
        Args:
            feat_dict (dict): 'cls_scores_list' (list[Tensor]): (bs, num_class, h, w),
                'bbox_preds_list' (list[Tensor]): (bs, 5, h, w),
                'cls_feats' (list[Tensor]): (bs, c, h, w),
                'reg_feats' (list[Tensor]): (bs, c, h, w).
            scores (list[Tensor]): (n, num_class).
            bboxes (list[Tensor]): (n, 5), where 5 represent (c_x, c_y, w, h, radian).
            query_ids (list[Tensor]): (n, )
            num (int): number of queries.
        Returns:
            query_xyzrt  (Tensor): (bs, num_query, 5), where 5 represent (c_x, c_y, z, r, radian)
            query_content (Tensor): (bs, num_query, 256).
        '''
        cls_feats = feat_dict['cls_feats']
        reg_feats = feat_dict['reg_feats']
        cls_feats = levels_to_images(cls_feats)
        reg_feats = levels_to_images(reg_feats)
        num_img = len(cls_feats)
        all_img_proposals = []
        all_img_object_feats = []
        for img_id in range(num_img):
            # Handle empty detection results
            if len(scores[img_id]) == 0 or scores[img_id].numel() == 0:
                # Create empty tensors with proper shape for padding
                singl_cls_feats = cls_feats[img_id]
                singl_reg_feats = reg_feats[img_id]
                object_feats = torch.cat([singl_cls_feats, singl_reg_feats], dim=-1)
                object_feats = self.compress(object_feats)
                
                # Create empty proposals and features with correct shape
                # They will be padded to num_proposals by align_tensor
                device = object_feats.device
                dtype_feat = object_feats.dtype
                # Ensure proper shape: (0, feature_dim) for features, (0, 5) for bboxes
                if object_feats.dim() >= 2:
                    feat_dim = object_feats.shape[-1]
                else:
                    feat_dim = object_feats.numel() if object_feats.numel() > 0 else 256
                
                all_img_proposals.append(torch.zeros((0, 5), device=device, dtype=torch.float32))
                all_img_object_feats.append(torch.zeros((0, feat_dim), device=device, dtype=dtype_feat))
                continue
            
            singl_scores = scores[img_id].max(-1).values
            singl_bboxes = bboxes[img_id]
            single_ids = query_ids[img_id]
            singl_cls_feats = cls_feats[img_id]
            singl_reg_feats = reg_feats[img_id]

            object_feats = torch.cat([singl_cls_feats, singl_reg_feats],
                                     dim=-1)

            object_feats = object_feats.detach()
            singl_bboxes = singl_bboxes.detach()

            object_feats = self.compress(object_feats)

            select_ids = torch.sort(singl_scores,
                                    descending=True).indices[:num]
            single_ids = single_ids[select_ids]
            singl_bboxes = singl_bboxes[select_ids]

            object_feats = object_feats[single_ids]
            all_img_object_feats.append(object_feats)
            all_img_proposals.append(singl_bboxes)

        all_img_object_feats = align_tensor(all_img_object_feats, self.num_proposals)   # (bs, num_query, 256)
        all_img_proposals = align_tensor(all_img_proposals, self.num_proposals)         # (bs, num_query, 5)

        xy, wh, radian = all_img_proposals[..., 0:2], all_img_proposals[..., 2:4], all_img_proposals[..., 4:]
        z = (wh).prod(-1, keepdim=True).sqrt().log2()           # bs, num_query, 1
        r = (wh[..., 1:2]/wh[..., 0:1]).log2()                  # bs, num_query, 1
        # NOTE: xyzr **not** learnable
        xyzrt = torch.cat([xy, z, r, radian], dim=-1).detach()  # bs, num_query, 5

        return dict(query_xyzrt=xyzrt,
                    query_content=all_img_object_feats)