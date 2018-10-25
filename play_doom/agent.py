from play_doom.experince_replay import Experience


class Agent(object):
    def __init__(self, env):
        self.experiences = Experience()
        self.env = env

    def episode(self, env, batch_size=10, epoch=0):
        pass

    def train_agent(self, batch_size):
        pass

    def get_action(self, state):
        pass

