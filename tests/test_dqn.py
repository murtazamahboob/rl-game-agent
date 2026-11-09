import numpy as np

from gameagent.dqn import DQNAgent, ReplayBuffer
from gameagent.train import evaluate, evaluate_random, train_agent


def test_replay_buffer_samples_correct_shapes():
    buf = ReplayBuffer(capacity=10, seed=0)
    for i in range(12):  # over capacity: oldest evicted
        buf.add(np.zeros(12), 1, 0.5, np.ones(12), False)
    assert len(buf) == 10
    s, a, r, s2, d = buf.sample(4)
    assert s.shape == (4, 12) and a.shape == (4,) and s2.shape == (4, 12)


def test_agent_act_is_greedy_when_epsilon_zero():
    agent = DQNAgent(seed=0)
    s = np.zeros(12)
    greedy = int(np.argmax(agent.online.forward(s)))
    assert agent.act(s, epsilon=0.0) == greedy


def test_learn_returns_finite_loss_after_warmup():
    agent = DQNAgent(seed=0)
    rng = np.random.default_rng(0)
    for _ in range(100):
        agent.buffer.add(rng.standard_normal(12), 1, 0.5,
                         rng.standard_normal(12), False)
    loss = agent.learn()
    assert loss is not None and np.isfinite(loss) and loss >= 0
    agent.sync_target()
    assert np.allclose(agent.target.weights[0], agent.online.weights[0])


def test_epsilon_decays_over_training():
    agent = DQNAgent(seed=0)
    agent.set_epsilon(0, 800)
    assert agent.epsilon == 1.0
    agent.set_epsilon(800, 800)
    assert abs(agent.epsilon - 0.05) < 1e-9


def test_training_improves_reward():
    """Short run (seeded): the last 40 episodes must beat the first 40."""
    _, history = train_agent(episodes=200, seed=0, log_every=0)
    assert history["final_100_mean_reward"] > history["first_100_mean_reward"], (
        f"no learning: first-100 {history['first_100_mean_reward']:.2f} vs "
        f"final-100 {history['final_100_mean_reward']:.2f}")
    assert history["train_steps"] > 0


def test_trained_agent_beats_random_on_same_seeds():
    agent, _ = train_agent(episodes=200, seed=0, log_every=0)
    trained = evaluate(agent, episodes=10, seed=999)
    random = evaluate_random(episodes=10, seed=999)
    assert trained["mean_reward"] > random["mean_reward"], (
        f"trained {trained['mean_reward']:.2f} <= random {random['mean_reward']:.2f}")
