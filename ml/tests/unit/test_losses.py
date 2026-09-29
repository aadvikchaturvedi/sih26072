import torch

from nowcast_ml.models.losses import BinaryFocalLoss, IntensityWeightedMSE


def test_intensity_weights():
    loss = IntensityWeightedMSE((20.0, 35.0), (2.0, 5.0))
    dbz = torch.tensor([[0.0, 19.9, 20.0, 34.9, 35.0, 60.0, float("nan")]])
    torch.testing.assert_close(loss.weight_map(dbz), torch.tensor([[1.0, 1, 2, 2, 5, 5, 1]]))


def test_weighted_mse_value_and_mask():
    loss = IntensityWeightedMSE((35.0,), (5.0,))
    pred = torch.tensor([1.0, 1.0, 1.0])
    target = torch.tensor([0.0, 0.0, 0.0])
    dbz = torch.tensor([10.0, 40.0, 40.0])
    valid = torch.tensor([True, True, False])
    # (1*1 + 5*1) / 2 valid pixels
    assert loss(pred, target, dbz, valid).item() == 3.0
    # same error at a strong-echo pixel costs 5x more
    a = loss(torch.tensor([1.0]), torch.tensor([0.0]), torch.tensor([10.0]), torch.tensor([True]))
    b = loss(torch.tensor([1.0]), torch.tensor([0.0]), torch.tensor([50.0]), torch.tensor([True]))
    assert b.item() == 5 * a.item()


def test_focal_gamma0_alpha_neg_equals_bce():
    logits = torch.randn(100)
    y = (torch.rand(100) > 0.7).float()
    fl = BinaryFocalLoss(alpha=-1, gamma=0.0)(logits, y)
    bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, y)
    torch.testing.assert_close(fl, bce)


def test_focal_downweights_easy_examples():
    fl = BinaryFocalLoss(alpha=-1, gamma=2.0)
    bce = lambda z, y: torch.nn.functional.binary_cross_entropy_with_logits(z, y)  # noqa: E731
    easy, hard = torch.tensor([4.0]), torch.tensor([-1.0])
    y = torch.tensor([1.0])
    ratio_easy = fl(easy, y) / bce(easy, y)
    ratio_hard = fl(hard, y) / bce(hard, y)
    assert ratio_easy < 0.001 and ratio_hard > 0.5


def test_focal_alpha_and_mask():
    fl = BinaryFocalLoss(alpha=0.25, gamma=0.0)
    z = torch.zeros(2)
    y = torch.tensor([1.0, 0.0])
    ln2 = torch.log(torch.tensor(2.0))
    torch.testing.assert_close(fl(z, y), (0.25 * ln2 + 0.75 * ln2) / 2)
    torch.testing.assert_close(fl(z, y, torch.tensor([True, False])), 0.25 * ln2)
