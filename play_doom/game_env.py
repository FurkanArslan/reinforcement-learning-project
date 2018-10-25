import os
import vizdoom as viz
import numpy as np

configs_path = "settings"

LEFT = [1, 0, 0]
RIGHT = [0, 1, 0]
SHOOT = [0, 0, 1]


class GameEnv:

    def __init__(self, screen_format=viz.ScreenFormat.GRAY8, window_visible=False):
        self.game = viz.DoomGame()
        self.load_config(screen_format, window_visible)
        self.game.init()

        self.possible_actions = [LEFT, RIGHT, SHOOT]

    def load_config(self, screen_format, window_visible, config_file='basic.cfg', scenario_file='basic.wad'):
        scenario_path = os.path.join(configs_path, scenario_file)
        config_path = os.path.join(configs_path, config_file)

        self.game.load_config(config_path)
        self.game.set_doom_scenario_path(scenario_path)

        self.game.set_screen_format(screen_format)
        self.game.set_window_visible(window_visible)

    def action_size(self):
        return self.game.get_available_buttons_size()

    def get_state(self):
        return self.game.get_state().screen_buffer

    def create_new_episode(self):
        self.game.new_episode()

        return self.get_state()

    def make_action(self, action):
        reward = self.game.make_action(action)
        done = self.game.is_episode_finished()

        if done:
            next_state = np.zeros((84, 84), dtype=np.int)
        else:
            next_state = self.get_state()

        return reward, done, next_state

