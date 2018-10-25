import random
import numpy as np


class EpsilonGreedyAction:
    def __init__(self, explore_start=1.0, explore_stop=0.01, decay_rate=0.0001):
        self.explore_start = explore_start
        self.explore_stop = explore_stop
        self.decay_rate = decay_rate
        self.decay_step = 0

    def get_action(self, state, possible_actions, model):
        explore_probability = self.explore_stop + (self.explore_start - self.explore_stop) * np.exp(
            -self.decay_rate * self.decay_step)

        # Random değer eğer epsilon değerimizden küçük ise rastgele hareket seç = keşif(exploration)
        if explore_probability > np.random.rand():
            action = random.choice(possible_actions)
        else:  # Random değer eğer epsilon değerimizden büyük ise greedy eylemi seç = sömürü(exploitation)

            # Eylemlerin değerlerini hesap et
            q_values = model.sess.run(model.output_, feed_dict={
                model.inputs_: state.reshape((1, *state.shape))
            })

            # Maximum değere sahip eylemi seç (greedy eylem)
            choice = np.argmax(q_values)
            action = possible_actions[int(choice)]
            self.decay_step += 1

        return action, explore_probability
