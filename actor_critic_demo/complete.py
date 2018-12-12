import gym
import numpy as np
import random
import tensorflow as tf

class PolicyGradientNNAgent():

  def __init__(self,
    actor_lr=0.2, 
    critic_lr=0.5, 
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
    self.actor_lr = actor_lr
    self.critic_lr = critic_lr
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

    self.create_actor(self.state_input)
    self.actor_vars =   tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='actor')
    self.create_critic(self.state_input)
    self.critic_vars =   tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='critic')

    self.create_actor_training(self.action)
    self.create_critic_training(self.action, self.target)


  def create_actor(self, state_input):
    """
    Create the policy network.
    """
    with tf.variable_scope("actor"):
      layer_1 = tf.layers.dense(state_input, self.n_hidden_1, activation=tf.nn.relu)
      layer_2 = tf.layers.dense(layer_1, self.n_hidden_2, activation=tf.nn.relu)

      last_layer = tf.layers.dense(layer_2, self.action_size)
      self.action_prob = tf.nn.softmax(last_layer)

  def create_critic(self, state_input):
    """
    Creates a vanilla actor critic critic. This estimates the action-value.
    """
    with tf.variable_scope("critic"):
      layer_1 = tf.layers.dense(state_input, self.n_hidden_1, activation=tf.nn.relu)
      layer_2 = tf.layers.dense(layer_1, self.n_hidden_2, activation=tf.nn.relu)
      self.q_val = tf.layers.dense(layer_2, self.action_size)

  def create_actor_training(self, action_taken):
    """
    Create the training function.
    """
    action_mask = tf.one_hot(action_taken, self.action_size, 1.0, 0.0)
    q_for_action = tf.reduce_sum(self.q_val * action_mask, 1)
    prob_for_action = tf.reduce_sum(self.action_prob  * action_mask, 1)

    # POLICY GRADIENT LOSS!
    self.pg_loss = tf.reduce_mean(-tf.log(prob_for_action) * q_for_action)

    l2_loss = tf.add_n([ tf.nn.l2_loss(v) for v in self.actor_vars  ]) 
    self.actor_loss = self.pg_loss + self.lam * l2_loss
    optimizer = tf.train.AdamOptimizer(learning_rate=self.actor_lr)
    self.actor_train_op = optimizer.minimize(self.actor_loss, var_list=self.actor_vars, global_step=tf.contrib.framework.get_global_step())

  def create_critic_training(self, action_taken, target_value):
    """
    Creates the training function for the critic.
    """
    action_mask = tf.one_hot(action_taken, self.action_size, 1.0, 0.0)
    q_for_action = tf.reduce_sum(self.q_val * action_mask, 1)

    # TD UPDATE
    self.critic_loss = tf.reduce_sum(tf.square(q_for_action - target_value))

    l2_loss = tf.add_n([ tf.nn.l2_loss(v) for v in self.critic_vars  ]) 
    self.critic_loss = self.critic_loss + self.lam * l2_loss
    optimizer = tf.train.AdamOptimizer(learning_rate=self.critic_lr)
    self.critic_train_op = optimizer.minimize(self.critic_loss, var_list=self.critic_vars, global_step=tf.contrib.framework.get_global_step())

  def predict_target(self, state, action, sess):
    feed_dict = {self.state_input:  [state]}
    pred_q_val = sess.run(self.q_val, feed_dict=feed_dict)
    q_val_for_action = pred_q_val[0][action]
    return q_val_for_action

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
    tot_ls = 0
    for t in range(len(episode)):
      self.total_steps = self.total_steps + 1
      
      state, action, next_state, reward, done = episode[t]

      # First we do the policy gradient update.
      feed_dict = { self.state_input: [state], self.action: [action] }
      sess.run([self.actor_train_op, self.actor_loss], feed_dict)

      # Then we do the TD update by constructing the target for the critic.
      # 1. Sample policy pi on next state.
      next_action = self.get_action(next_state, sess)
      # 2. If not done, predict the expected future reward
      if done:
        target = reward
      else:
        # We need to use the critic to predict the expected future reward.
        expected_future = self.predict_target(next_state, next_action, sess)
        target = reward + self.gamma*expected_future
      # Now run gradient descent using this target on the critic.
      feed_dict = { 
        self.state_input: [state], 
        self.action: [action],
        self.target: [target]
      }
      _, ls = sess.run([self.critic_train_op, self.critic_loss], feed_dict)
      tot_ls += ls
    return tot_ls