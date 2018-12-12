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
    # Fill this out
    pass

  def create_training(self, action_taken, target_value_function):
    """
    Create the training function.
    """
    # Fill fill this out
    pass


  def get_action(self, state, sess):
    """
    Randomly sample the policy.
    """

    # Fill this out
    pass

  def get_policy(self, state, sess):
    """returns policy as probability distribution of actions"""

    # Fill this out 
    pass

  def learn(self, episode, sess,):
    """
    The training loop for a single episode.
    Args:
      episode: A list of (State, action, next_state, reward, done) pairs
      sess: The Tensorflow session.
    """

    # Fill this out 
    pass