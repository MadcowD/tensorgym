"""
The expert recorder.
"""
import argparse
import keyboard
import random
import gym
import gym_minecraft
import numpy as np
import time
import os

from config import (
    GYM_RESOLUTION,
    MALMO_IP,
    BINDINGS,
    SHARD_SIZE,
    RECORD_INTERVAL)




def get_options():
    parser = argparse.ArgumentParser(description='Process some integers.')
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

    env = gym.make('MinecraftDefaultWorld1-v0')
    env.init(
        start_minecraft=None,
        client_pool=[('127.0.0.1', 10000)],
        continuous_discrete = True,
        videoResolution=GYM_RESOLUTION,
        add_noop_command=True)

    ##############
    # BIND KEYS  #1
    ##############

    keyboard.unhook_all()
    keys_pressed = {}
    action = ""
    record = False
    esc = False

    def keyboard_hook(event):
        """
        The key manager for interaction with minecraft.
        Allow sfor simultaneous execution of movement 
        """
        nonlocal action, keys_pressed, record, esc
        if event.event_type is keyboard.KEY_DOWN:
            keys_pressed[event.name] = True
        else:   
            if 'r' in keys_pressed: record = not record
            if '+' in keys_pressed: esc = True
            if event.name in keys_pressed:
                del keys_pressed[event.name]

        
        actions_to_process = []
        for kmap, default in BINDINGS:
            pressed = [x for x in kmap if x in keys_pressed]
            if len(pressed) > 1 or len(pressed) == 0:
                actions_to_process.append(default)
            else:
                actions_to_process.append(kmap[pressed[0]])


        action = "\n".join(actions_to_process)

    keyboard.hook(keyboard_hook)


    shard_suffix = ''.join(random.choice('0123456789ABCDEF') for i in range(16))
    sarsa_pairs = []
    _old_record = record
    done = False
    last_action_time = time.time()
    _last_action = ''
    _last_obs = env.reset()


    no_action = False

    while not done:
        env.render()
        
        # Handle the toggling of different application states
        if _old_record is not record:
            print("Recording: ", record)
            _old_record = record
        if esc:
            print("ENDING")
            done = True
            break

        #  make actions if and only if 
        # the awllotted recording interval has past
        # or instantantelously make actions if we're not recording.
        cur_time = time.time()
        if cur_time - last_action_time > RECORD_INTERVAL or not record:
            if keys_pressed :
                obs, reward, done, info = env.step(action)
                no_action = False
            else:
                obs, reward, done, info  = env.step(action)
                no_action = True

            # Record the data
            if record:
                # When the agent stops acting, record a no action
                # Otherwise wait untill it acts again.
                if ((no_action and _last_action is not action)
                        or (no_action and not (_last_obs == obs).all())
                        or not no_action):
                    sarsa = (obs, action[:])
                    sarsa_pairs.append(sarsa)

                    print("recording", len(sarsa_pairs))

            # Update the action time.
            last_action_time = cur_time
            _last_action = action
            _last_obs = obs
        else:
            env.step(_last_action)


    keyboard.unhook(keyboard_hook)

    print("SAVING")
    # Save out recording data.
    num_shards = int(np.ceil(len(sarsa_pairs)/SHARD_SIZE))
    for shard_iter in range(num_shards):
        shard = sarsa_pairs[
            shard_iter*SHARD_SIZE: min(
                (shard_iter+1)*SHARD_SIZE, len(sarsa_pairs))]

        shard_name = "{}_{}.npy".format(str(shard_iter), shard_suffix)
        with open(os.path.join(ddir, shard_name), 'wb') as f:
            np.save(f, sarsa_pairs)

if __name__ == "__main__":
    opts = get_options()
    run_recorder(opts)