import random
import numpy as np

from collections import deque

MEMORY_SIZE = 1000000


class Experience:
    def __init__(self, max_size=MEMORY_SIZE):
        self.buffer = deque(maxlen=max_size)

    def fill_experience_array_with_random(self, env, batch_size, model):
        state = env.create_new_episode()
        state, stacked_frames = model.stack_frames(state, True)

        for i in range(batch_size):
            action = random.choice(env.possible_actions)
            reward, done, next_state = env.make_action(action)

            if done:
                self.add((state, action, reward, next_state, done))

                state = env.create_new_episode()
                state, stacked_frames = model.stack_frames(state, True)

            else:
                next_state, stacked_frames = model.stack_frames(stacked_frames, next_state, False)

                self.add((state, action, reward, next_state, done))

                state = next_state

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        buffer_size = len(self.buffer)
        index = np.random.choice(np.arange(buffer_size), size=batch_size, replace=False)

        return [self.buffer[i] for i in index]

    def batch_sample(self):
        batch = self.sample()

        states = np.array([each[0] for each in batch], ndmin=3)
        actions = np.array([each[1] for each in batch])
        rewards = np.array([each[2] for each in batch])
        next_states = np.array([each[3] for each in batch], ndmin=3)
        dones = np.array([each[4] for each in batch])

        return states, actions, rewards, next_states, dones
