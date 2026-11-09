import numpy as np

from gameagent.nn import MLP


def test_forward_shapes_and_save_load_roundtrip(tmp_path):
    net = MLP([12, 64, 64, 3], seed=1)
    x = np.random.default_rng(0).standard_normal((5, 12))
    out = net.forward(x)
    assert out.shape == (5, 3)
    assert np.isfinite(out).all()
    assert net.n_params == 12 * 64 + 64 + 64 * 64 + 64 + 64 * 3 + 3
    path = str(tmp_path / "net.npz")
    net.save(path)
    net2 = MLP.load(path)
    assert np.allclose(net2.forward(x), out)


def test_training_steps_reduce_mse_on_a_fixed_batch():
    rng = np.random.default_rng(0)
    net = MLP([12, 32, 3], seed=2)
    x = rng.standard_normal((16, 12))
    target = rng.standard_normal((16, 3))
    losses = []
    for _ in range(30):
        pred = net.forward(x)
        err = pred - target
        losses.append(float(np.mean(err ** 2)))
        net.backward(2 * err / err.size)
        net.adam_step(lr=0.01)
    assert losses[-1] < 0.5 * losses[0], f"{losses[0]:.3f} -> {losses[-1]:.3f}"
