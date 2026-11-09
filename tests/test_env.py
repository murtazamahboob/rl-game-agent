import numpy as np

from gameagent.env import HEIGHT, MAX_STEPS, STATE_SIZE, WIDTH, NeonCatchEnv


def test_reset_is_deterministic_with_same_seed():
    a, b = NeonCatchEnv(seed=3), NeonCatchEnv(seed=3)
    assert np.array_equal(a.reset(), b.reset())
    for _ in range(20):
        sa, _, _, _ = a.step(1)
        sb, _, _, _ = b.step(1)
        assert np.array_equal(sa, sb)


def test_state_size_and_agent_bounds():
    env = NeonCatchEnv(seed=0)
    s = env.reset()
    assert s.shape == (STATE_SIZE,)
    assert np.isfinite(s).all()
    for _ in range(MAX_STEPS):
        env.step(0)  # always left
    assert env.agent_x == 0


def test_catching_an_orb_scores_plus_one():
    env = NeonCatchEnv(seed=0)
    env.reset()
    env.objects = [{"x": env.agent_x, "y": HEIGHT - 2, "kind": 1}]
    _, reward, _, info = env.step(1)  # stay, orb lands on the agent
    assert info["orbs"] == 1
    assert reward > 0.9  # +1 orb, minus the tiny step penalty


def test_bomb_on_the_agent_is_a_penalty():
    env = NeonCatchEnv(seed=0)
    env.reset()
    env.objects = [{"x": env.agent_x, "y": HEIGHT - 2, "kind": -1}]
    _, reward, _, info = env.step(1)
    assert info["bombs"] == 1
    assert reward < -0.9


def test_episode_ends_after_max_steps():
    env = NeonCatchEnv(seed=0)
    env.reset()
    done = False
    for _ in range(MAX_STEPS):
        _, _, done, _ = env.step(1)
    assert done


def test_difficulty_raises_bomb_rate():
    easy = NeonCatchEnv(seed=0, difficulty=0.0)
    hard = NeonCatchEnv(seed=0, difficulty=1.0)
    assert hard.bomb_rate > easy.bomb_rate
    assert hard.orb_rate < easy.orb_rate


def test_render_ascii_marks_agent_and_objects():
    env = NeonCatchEnv(seed=0)
    env.reset()
    env.objects = [{"x": 2, "y": 1, "kind": 1}, {"x": 3, "y": 1, "kind": -1}]
    text = env.render_ascii()
    assert "A" in text and "O" in text and "X" in text
    assert len(text.splitlines()) == HEIGHT
    assert all(len(line.split()) == WIDTH for line in text.splitlines())
