from pathlib import Path
from math import isclose

import torch

from ultralytics.nn.modules import DCRN, GCAM
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.loss import BboxLossDSNIM


ROOT = Path(__file__).resolve().parents[1]


def test_dsnim_alpha_schedule_changes():
    """验证 DS-NIM 的动态权重会随 epoch 变化。"""
    loss = BboxLossDSNIM(reg_max=16)

    loss.set_epoch(0, 300)
    assert isclose(loss._get_alpha(), 0.2)

    loss.set_epoch(149, 300)
    assert isclose(loss._get_alpha(), 0.4989966555)

    loss.set_epoch(299, 300)
    assert isclose(loss._get_alpha(), 0.8)


def test_dsnim_hybrid_loss_forward_backward():
    """验证 DS-NIM 混合定位损失可以正常前向与反传。"""
    loss = BboxLossDSNIM(reg_max=16)
    loss.set_epoch(299, 300)

    pred_dist = torch.randn(1, 2, 64, requires_grad=True)
    pred_bboxes = torch.tensor([[[4.0, 4.0, 8.0, 8.0], [8.0, 8.0, 12.0, 12.0]]], requires_grad=True)
    anchor_points = torch.tensor([[6.0, 6.0], [10.0, 10.0]])
    target_bboxes = torch.tensor([[[4.5, 4.5, 8.5, 8.5], [8.0, 8.0, 12.0, 12.0]]])
    target_scores = torch.tensor([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]])
    fg_mask = torch.tensor([[True, True]])
    target_scores_sum = target_scores.sum()
    imgsz = torch.tensor([640.0, 640.0])
    stride = torch.ones(1, 2, 1)

    loss_iou, loss_dfl = loss(
        pred_dist,
        pred_bboxes,
        anchor_points,
        target_bboxes,
        target_scores,
        target_scores_sum,
        fg_mask,
        imgsz,
        stride,
    )
    total_loss = loss_iou + loss_dfl
    total_loss.backward()

    assert torch.isfinite(total_loss)
    assert pred_dist.grad is not None
    assert pred_bboxes.grad is not None


def test_full_model_dcrn_links_to_gcam():
    """验证完整模型中 DCRN 能拿到前序 GCAM 引用。"""
    model = DetectionModel(str(ROOT / "ultralytics/cfg/models/11/yolo11-full.yaml"), nc=3, verbose=False)
    dcrn = next(layer for layer in model.model if isinstance(layer, DCRN))

    assert isinstance(dcrn.gc_module, GCAM)
    assert model.model[-1].f == [17, 21, 24]
