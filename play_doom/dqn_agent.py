import numpy as np

from play_doom.agent import Agent


class DqnAgent(Agent):
    def __init__(self, env, model, action_model, discount_rate=0.95):
        super().__init__(env)
        self.model = model
        self.action_model = action_model
        self.discount_rate = discount_rate

    def run_training(self, episode_size, env):
        for episode in range(episode_size):
            total_reward = self.episode(env)

            # Save model every 5 episodes
            if episode % 5 == 0:
                self.model.save_model()
                print("\n Episode: %d, Model Saved" % episode)

            print('*' * 70)
            print('Episode Finished! Total Reward is %d\n' % total_reward)
            print('*' * 70)

    def episode(self, env, batch_size=64, epoch=0):
        total_reward = 0
        step = 0
        loss = 0
        episode_rewards = []

        state = env.create_new_episode()
        state = self.model.stack_frames(state, True)
        done = False

        while not done:
            step += 1

            action, explore_probability = self.get_action(state)

            reward, done, next_state = env.make_action(action)

            episode_rewards.append(reward)

            if done:
                next_state = self.model.stack_frames(next_state, False)

                total_reward = np.sum(episode_rewards)

                print('Episode: {}'.format(epoch),
                      'Total reward: {}'.format(total_reward),
                      'Training loss: {:.4f}'.format(loss),
                      'Explore P: {:.4f}'.format(explore_probability))

                self.experiences.add((state, action, reward, next_state, done))
            else:
                if step % 10 == 0:
                    print('Episode: {}'.format(epoch),
                          'Step: {}'.format(step),
                          'Action: {}'.format(action),
                          'Training loss: {:.4f}'.format(loss),
                          'Explore P: {:.4f}'.format(explore_probability))

                next_state = self.model.stack_frames(next_state, False)

                self.experiences.add((state, action, reward, next_state, done))

                state = next_state

            loss = self.train(batch_size)

        return total_reward

    def train_agent(self, batch_size):
        target_qs_batch = []
        states, actions, rewards, next_states, dones = self.experiences.batch_sample()

        qs_next_state = self.model.sess.run(self.model.output_, feed_dict={self.model.inputs_: next_states})

        # Set Q_target = r if the episode ends at s+1, otherwise set Q_target = r + DISCOUNT_RATE*maxQ(s', a')
        for i in range(0, batch_size):
            terminal = dones[i]

            if terminal:  # Eğer terminal durumunda ise Q_target = r
                target_qs_batch.append(rewards[i])

            else:  # Q_target = r + DISCOUNT_RATE*maxQ(s', a')
                target = rewards[i] + self.discount_rate * np.max(qs_next_state[i])
                target_qs_batch.append(target)

        targets = np.array([each for each in target_qs_batch])

        loss, _ = self.model.sess.run([self.model.loss_, self.model.optimizer_],
                                      feed_dict={self.model.inputs_: states,
                                                 self.model.target_Q: targets,
                                                 self.model.actions_: actions})

        return loss

    def get_action(self, state):
        return self.action_model.get_action(state, self.env.possible_actions, self.model)
