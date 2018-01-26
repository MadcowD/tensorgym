import tensorflow as tf 
import numpy as np
import argparse
import os
import gym

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


    return state_ph, action, logits

def create_training(logits):
    """
    Creates the training method.
    """
    

    return train_op, loss, label_ph

def run_main(opts):
    # Create the environment with specified arguments
    state_data, action_data = process_data(opts.bc_data)

    env = gym.make('MountainCar-v0')
    env._max_episode_steps = 1200





    while True:
        done = False
        obs = env.reset()
        while not done:
            env.render()
            # Train




            # Get the action


            obs, reward, done, info = env.step(action)


if __name__ == "__main__":
    # Parse arguments
    opts = get_options()
    # Start the main thread.
    run_main(opts)