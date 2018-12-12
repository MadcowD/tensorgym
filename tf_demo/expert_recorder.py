"""
The expert recorder.
"""
import argparse
import getch
import random
import gym
import numpy as np
import time
import os

BINDINGS = {
    'a': 0,
    'd': 2}
SHARD_SIZE = 2000

def get_options():
    parser = argparse.ArgumentParser(description='Records an expert..')
    parser.add_argument('data_directory', type=str,
        help="The main datastore for this particular expert.")

    args = parser.parse_args()

    return args


def run_recorder(opts):
    """
    Runs the main recorder by binding certain discrete actions to keys.
    """
    ddir = opts.data_directory

    record_history = [] # The state action history buffer.

    env = gym.make('MountainCar-v0')
    env._max_episode_steps = 1200

    ##############
    # BIND KEYS  #
    ##############

    action = None
    esc = False


    shard_suffix = ''.join(random.choice('0123456789ABCDEF') for i in range(16))
    sarsa_pairs = []

    print("Welcome to the expert recorder")
    print("To record press either a or d to move the agent left or right.")
    print("Once you're finished press + to save the data.")
    print("NOTE: Make sure you've selected the console window in order for the application to receive your input.")

    while not esc:

        done = False
        _last_obs = env.reset()
        while not done:
            env.render()
            # Handle the toggling of different application states



            # Take the current action if a key is pressed.
            action = None
            while action is None:
                keys_pressed  = getch.getch()
                if keys_pressed is '+':
                    esc = True
                    break

                pressed = [x for x in BINDINGS if x in keys_pressed]
                action = BINDINGS[pressed[0]] if len(pressed) > 0 else None

            if esc:
                print("ENDING")
                done = True
                break

            obs, reward, done, info = env.step(action)
            
            no_action = False
            sarsa = (_last_obs, action)
            _last_obs = obs
            sarsa_pairs.append(sarsa)

        if esc:
            break



    print("SAVING")
    # Save out recording data.
    num_shards = int(np.ceil(len(sarsa_pairs)/SHARD_SIZE))
    for shard_iter in range(num_shards):
        shard = sarsa_pairs[
            shard_iter*SHARD_SIZE: min(
                (shard_iter+1)*SHARD_SIZE, len(sarsa_pairs))]

        shard_name = "{}_{}.npy".format(str(shard_iter), shard_suffix)
        if not os.path.exists(ddir):
            os.makedirs(ddir)
        with open(os.path.join(ddir, shard_name), 'wb') as f:
            np.save(f, sarsa_pairs)

if __name__ == "__main__":
    opts = get_options()
    run_recorder(opts)