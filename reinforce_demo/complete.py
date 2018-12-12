import gym
import numpy as np
import random
import tensorflow as tf

class PolicyGradientNNAgent():

  def __init__(self,
    lr=0.5, 
    gamma=0.99, 
    lam=0.002,
    state_size=4,
    action_size=2,
    n_hidden_1=20,
    n_hidden_2=20,
    scope="pg"
    ):
    """
    args
      epsilon           exploration rate
      epsilon_anneal    linear decay rate per call of learn() function (iteration)
      end_epsilon       lowest exploration rate
      lr                learning rate
      gamma             discount factor
      state_size        network input size
      action_size       network output size
    """
    self.lr = lr
    self.lam = lam
    self.gamma = gamma
    self.state_size = state_size
    self.action_size = action_size
    self.total_steps = 0
    self.n_hidden_1 = n_hidden_1
    self.n_hidden_2 = n_hidden_2
    self.scope = scope


    self.state_input = tf.placeholder(tf.float32, [None, self.state_size])
    self.action = tf.placeholder(tf.int32, [None])
    self.target = tf.placeholder(tf.float32, [None])

    self.create_policy(self.state_input)
    self.create_training(self.action, self.target)


  def create_policy(self, state_input):
    """
    Create the policy network.
    """
    layer_1 = tf.layers.dense(state_input, self.n_hidden_1, activation=tf.nn.relu)
    layer_2 = tf.layers.dense(layer_1, self.n_hidden_2, activation=tf.nn.relu)

    self.action_values = tf.layers.dense(layer_2, self.action_size)
    self.action_prob = tf.nn.softmax(self.action_values)

  def create_training(self, action_taken, target_value_function):
    """
    Create the training function.
    """
    action_mask = tf.one_hot(action_taken, self.action_size, 1.0, 0.0)
    self.action_value_pred = tf.reduce_sum(self.action_prob * action_mask, 1)

    self.l2_loss = tf.add_n([ tf.nn.l2_loss(v) for v in tf.trainable_variables()  ]) 

    # POLICY GRADIENT LOSS!
    self.pg_loss = tf.reduce_mean(-tf.log(self.action_value_pred) * target_value_function)

    self.loss = self.pg_loss + self.lam * self.l2_loss
    self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
    self.train_op = self.optimizer.minimize(self.loss, global_step=tf.contrib.framework.get_global_step())


  def get_action(self, state, sess):
    """
    Randomly sample the policy.
    """
    pi = self.get_policy(state, sess)
    random_discrte_action = np.random.choice(list(range(self.action_size)), p=pi)
    return random_discrte_action


  def get_policy(self, state, sess):
    """returns policy as probability distribution of actions"""
    pi = sess.run(self.action_prob, feed_dict={self.state_input: [state]})
    return pi[0]


  def learn(self, episode, sess,):
    """
    The training loop for a single episode.
    Args:
      episode: A list of (State, action, next_state, reward, done) pairs
      sess: The Tensorflow session.
    """
    for t in range(len(episode)):
      self.total_steps = self.total_steps + 1
      target = sum([self.gamma**i * r for i, (s, a, s1, r, d) in enumerate(episode[t:])])
      state, action, next_state, reward, done = episode[t]
      feed_dict = { self.state_input: [state], self.target: [target], self.action: [action] }
      sess.run([self.train_op, self.loss], feed_dict)