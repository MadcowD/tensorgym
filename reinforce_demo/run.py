import gym
from gym import wrappers
import complete
import numpy
import matplotlib.pyplot as plt
import tensorflow as tf
import time

NUM_EPISODES = 800
MAX_STEPS = 200
FAIL_PENALTY = -100
# LEARNING_RATE = 0.0001 # hidden layer 10/20
LEARNING_RATE = 0.002 # hidden layer 5
# LEARNING_RATE = 0.1 # hidden layer 3
DISCOUNT_FACTOR = 0.9
TRAIN_EVERY_NUM_EPISODES = 1
EPOCH_SIZE = 1
MEM_SIZE = 100

RECORD = False


def train(agent, env, sess, num_episodes=NUM_EPISODES):
  history = []
  for i in range(NUM_EPISODES):
    if i % 20 == 0:
      print("Episode {}".format(i + 1))
      print("Doing evaluation.")
      tot_reward = 0
      cur_state = env.reset()
      for t in range(MAX_STEPS):
        time.sleep(0.05)
        env.render()
        action = agent.get_action(cur_state, sess)
        cur_state, reward, done, info = env.step(action)
        if done:
          print("\tAgent lasted for {}".format(t))
          break
    cur_state = env.reset()
    episode = []
    for t in range(MAX_STEPS):
      action = agent.get_action(cur_state, sess)
      next_state, reward, done, info = env.step(action)
      if done:
        reward = FAIL_PENALTY
        episode.append([cur_state, action, next_state, reward, done])
        # if i % 10 == 0:
        # print(("Episode finished after {} timesteps".format(t + 1)))
        # print(agent.get_policy(cur_state, sess))
        history.append(t + 1)
        break
      episode.append([cur_state, action, next_state, 1, done])
      cur_state = next_state
      if t == MAX_STEPS - 1:
        history.append(t + 1)
        print(("Episode finished after {} timesteps".format(t + 1)))
    # agent.add_episode(episode)
    if i % TRAIN_EVERY_NUM_EPISODES == 0:
      # print('train at episode {}'.format(i))
      agent.learn(episode, sess)
  return agent, history


agent = complete.PolicyGradientNNAgent(lr=LEARNING_RATE,
                                          gamma=DISCOUNT_FACTOR,
                                          state_size=4,
                                          action_size=2,
                                          n_hidden_1=5,
                                          n_hidden_2=5)


env = gym.make('CartPole-v0')
env._max_episode_steps = 200
if RECORD:
  env = wrappers.Monitor(env, '/tmp/cartpole-experiment-2', force=True)


with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  agent, history = train(agent, env, sess)


if RECORD:
  env.monitor.close()

window = 10
avg_reward = [numpy.mean(history[i*window:(i+1)*window]) for i in range(int(len(history)/window))]
f_reward = plt.figure(1)
plt.plot(numpy.linspace(0, len(history), len(avg_reward)), avg_reward)
plt.ylabel('Rewards')
f_reward.show()
print('press enter to continue')
input()
