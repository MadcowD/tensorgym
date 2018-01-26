import tensorflow as tf 
import numpy as np
import argparse
import os
import gym

BINDINGS = {
    'w': 1,
    'a': 3,
    's': 4,
    'd': 2}
SHARD_SIZE = 2000


def get_options():
    parser = argparse.ArgumentParser(description='Clone some expert data..')
    parser.add_argument('bc_data', type=str,
        help="The main datastore for this particular expert.")

    args = parser.parse_args()
    return args


def process_data(bc_data_dir):
    """
    Runs training for the agent.
    """
    # Load the file store. 
    # In the future (TODO) move this to a seperate thread.
    states, actions = [], []
    shards = [x for x in os.listdir(bc_data_dir) if x.endswith('.npy')]
    print("Processing shards: {}".format(shards))
    for shard in shards:
        shard_path = os.path.join(bc_data_dir, shard)
        with open(shard_path, 'rb') as f:
            data = np.load(f)
            shard_states, unprocessed_actions = zip(*data)
            shard_states = [x.flatten() for x in shard_states]
            
            # Add the shard to the dataset
            states.extend(shard_states)
            actions.extend(unprocessed_actions)

    states = np.asarray(states, dtype=np.float32)
    actions = np.asarray(actions, dtype=np.float32)/2
    print("Processed with {} pairs".format(len(states)))
    return states, actions

def create_model():
    """
    Creates the model.
    """
    state_ph = tf.placeholder(tf.float32, shape=[None, 2])
    # Process the data

    # # Hidden neurons
    with tf.variable_scope("layer1"):
        hidden = tf.layers.dense(state_ph, 128, activation=tf.nn.relu)

    with tf.variable_scope("layer2"):
        hidden = tf.layers.dense(hidden, 128, activation=tf.nn.relu)
    # Make output layers
    with tf.variable_scope("layer3"):
        logits = tf.layers.dense(hidden, 2) 
    # Take the action with the highest activation
    with tf.variable_scope("output"):
        action = tf.argmax(input=logits, axis=1)

    return state_ph, action, logits

def create_training(logits):
    """
    Creates the model.
    """
    label_ph = tf.placeholder(tf.int32, shape=[None])

    # Convert it to a onehot. 1-> [1,0,0,0]
    with tf.variable_scope("loss"):
        onehot_labels = tf.one_hot(indices=tf.cast(label_ph, tf.int32), depth=2) # 4 actions

        loss = tf.losses.softmax_cross_entropy(
            onehot_labels=onehot_labels, logits=logits)
        loss = tf.reduce_mean(loss)

        tf.summary.scalar('loss', loss)

    with tf.variable_scope("training"):
        optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)
        train_op = optimizer.minimize(loss=loss)

    return train_op, loss, label_ph

def run_main(opts):
    # Create the environment with specified arguments
    state_data, action_data = process_data(opts.bc_data)

    env = gym.make('MountainCar-v0')
    env._max_episode_steps = 1200

    x, model, logits = create_model()
    train, loss, labels = create_training(logits)

    sess = tf.Session()

    # Create summaries
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter('./logs',sess.graph)

    sess.run(tf.global_variables_initializer())


    tick = 0
    while True:
        done = False
        obs = env.reset()
        while not done:

            env.render()
             # Get a random batch from the data
            batch_index = np.random.choice(len(state_data), 64) #Batch size
            state_batch, action_batch = state_data[batch_index], action_data[batch_index]

            # Train the model.
            _, cur_loss, cur_summaries = sess.run([train, loss, merged], feed_dict={
                x: state_data,
                labels: action_data
                })
            print("Loss: {}".format(cur_loss))
            train_writer.add_summary(cur_summaries, tick)


            # Handle the toggling of different application states
            action = sess.run(model, feed_dict={
                x: [obs.flatten()]
            })[0]*2 


            obs, reward, done, info = env.step(action)


            tick += 1


if __name__ == "__main__":
    # Parse arguments
    opts = get_options()
    # Start the main thread.
    run_main(opts)