from play_doom.dqn_agent import DqnAgent
from play_doom.epsilon_greedy_action import EpsilonGreedyAction
from play_doom.game_env import GameEnv
from play_doom.model import Model

if __name__ == "__main__":
    env = GameEnv(window_visible=True)

    deep_learning_model = Model(env.action_size)
    action_model = EpsilonGreedyAction()

    agent = DqnAgent(env, deep_learning_model, action_model)
    agent.run_training(episode_size=500, env=env)
