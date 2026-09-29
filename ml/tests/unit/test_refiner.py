import numpy as np
import pytest
import torch

from nowcast_ml.config import RefinerConfig
from nowcast_ml.evaluation.metrics import crps_ensemble
from nowcast_ml.models.refiner import NullRefiner, build_refiner
from nowcast_ml.models.refiner.diffusion import DiffusionRefiner, cosine_alpha_bar
from nowcast_ml.models.refiner.unet import ConditionalUNet

CFG = RefinerConfig(base_channels=16, channel_mults=[1, 2], timesteps=200, sample_steps=10)


def test_schedule():
    ab = cosine_alpha_bar(1000)
    assert ab.shape == (1000,) and torch.all(ab[1:] < ab[:-1])
    assert 0.99 < ab[0] <= 1.0 and ab[-1] < 1e-3


def test_unet_shape_and_zero_init():
    net = ConditionalUNet(c_out=12, c_cond=20, base=16, channel_mults=(1, 2, 2))
    y = net(torch.randn(2, 12, 16, 24), torch.tensor([3, 150]), torch.randn(2, 20, 16, 24))
    assert y.shape == (2, 12, 16, 24) and torch.all(y == 0)  # zero-initialised output layer


def test_sample_shapes_and_seeding():
    torch.manual_seed(0)
    r = DiffusionRefiner(t_out=4, cond_channels=6, cfg=CFG).eval()
    for p in r.net.out.parameters():  # non-trivial denoiser
        torch.nn.init.normal_(p, std=0.01)
    cond = torch.randn(1, 7, 6, 16, 16)
    det = torch.randn(1, 4, 16, 16)
    a = r.sample(cond, det, n_members=3, seed=1)
    b = r.sample(cond, det, n_members=3, seed=1)
    c = r.sample(cond, det, n_members=3, seed=2)
    assert a.shape == (1, 3, 4, 16, 16) and torch.isfinite(a).all()
    assert torch.equal(a, b) and not torch.equal(a, c)
    assert not torch.allclose(a[0, 0], a[0, 1])  # members differ


def test_refiner_learns_constant_residual():
    """Truth = deterministic + 1 everywhere -> sampled members should average ~ det + 1."""
    torch.manual_seed(0)
    r = DiffusionRefiner(t_out=2, cond_channels=4, cfg=CFG)
    opt = torch.optim.Adam(r.parameters(), lr=2e-3)
    cond = torch.randn(8, 7, 4, 16, 16)
    det = torch.randn(8, 2, 16, 16)
    target = det + 1.0
    valid = torch.ones_like(det, dtype=torch.bool)
    for _ in range(300):
        loss = r.loss(cond, det, target, valid)
        opt.zero_grad()
        loss.backward()
        opt.step()
    r.eval()
    resid = r.sample(cond[:2], det[:2], n_members=4, seed=0) - det[:2, None]
    assert abs(resid.mean().item() - 1.0) < 0.1 and resid.std().item() < 0.2


def test_refiner_represents_uncertainty():
    """Truth = det +/- 1 with an unpredictable sign -> members must spread (not collapse to the mean)."""
    torch.manual_seed(0)
    r = DiffusionRefiner(t_out=1, cond_channels=2, cfg=CFG)
    opt = torch.optim.Adam(r.parameters(), lr=2e-3)
    cond = torch.zeros(16, 7, 2, 8, 8)
    det = torch.zeros(16, 1, 8, 8)
    valid = torch.ones_like(det, dtype=torch.bool)
    for _ in range(500):
        sign = torch.randint(0, 2, (16, 1, 1, 1)).float() * 2 - 1
        loss = r.loss(cond, det, det + sign, valid)
        opt.zero_grad()
        loss.backward()
        opt.step()
    r.eval()
    m = r.sample(cond[:1], det[:1], n_members=16, seed=0)[0, :, 0].mean(
        dim=(-1, -2)
    )  # field mean per member
    assert m.std().item() > 0.5, m  # a collapsed (deterministic) model would give ~0
    assert (m > 0.5).any() and (m < -0.5).any()  # both modes are sampled


def test_build_refiner():
    assert isinstance(build_refiner(None), NullRefiner)
    assert isinstance(build_refiner("diffusion", 12, 28, CFG), DiffusionRefiner)
    with pytest.raises(ValueError):
        build_refiner("gan")


def test_crps_hand_values():
    # members {0, 2}, obs 1: mean|X-y| = 1, sum_ij|Xi-Xj| = 4 -> 4 / (2*4) = 0.5 -> CRPS 0.5
    assert crps_ensemble(np.array([[0.0], [2.0]]), np.array([1.0]))[0] == pytest.approx(0.5)
    # single member = absolute error
    assert crps_ensemble(np.array([[3.0]]), np.array([1.0]))[0] == pytest.approx(2.0)
    # perfect deterministic forecast
    assert crps_ensemble(np.array([[1.0], [1.0]]), np.array([1.0]))[0] == 0.0
