# Tensorflow + Gym + RL Tutorial.

This repo contains notes for a tutorial on reinforcement learning. The tutorial is centered around Tensorflow and OpenAI Gym, two libraries for conducitng deep learning and the agent-environment loop, respectively, in Python. A general outline is as follows:

1. Gym: `gym_demo`

	a. Explain at a high level what reinforcement learning is and how the agent-environment loop works. Particularly, explain that there has been no standard library for describing this procedure for a long time.
	
	b. Introduce Gym as this standard wrapper, presenting various environments.
	
	c. Show some simple code (`gym_demo`) of taking random actions in the mountain car environment. 
	
2. Tensorflow Explaination:

	a. Explain that to make an agent which acts well in an environment, we need to develop techniques for producing an optimal agent with respect to expected future reward. At this point feel free to introduce various soa approaches to reinforcement learning, and specifically, how important deep learning is in this context.
	
	b. Show motivating example of trying to do deep learning from scratch in Numpy (its hard, you have to manually do error backpropagation.) Then compare to the TF code, hopefully you can show the power of TF at this point.
	
	c. Explain different pieces of Tensorflow, variables, ops, sessions, etc.

3. Behavioral Cloning + Tensorflow Demo `tf_demo` 

	a. You'll need to add some slides about how BC works
	
	
	c. Now to show behavioral cloning in action run `python3 expert_recorder.py ./data/`
		This will let you create a small dataset of when the agent should turn left or right at certain
		points in mountain car. (See the YouTube video to see how I do this https://youtu.be/0rsrDOXsSeM?t=1370)
		When you run `expert_recorder.py`, select the terminal window in which you ran the command, then use the `a` and `d` keys to move the agent left and right respectively. The environment will only tick if you press either of these keys. Once you've finished recording press `+` to save the data to the folder specified.
		Once you've recorded the data you're going to want to then talk about building a model to learn from it.
		
		
	d. Now we actually build the model 
	
	- Fill in create_model, create_training, and run_main.
	- Run the model `python3 complete.py ./data/`, it should work instantly (discrease learning rate for a cooler effect.)
	
	- At this point you can show Tensorboard to examine your neural network structure. To do this, execute following command in the same directory as the code. 
	```
	tensorboard --logdir ./logs
	```

4. Reinforce Demo.

	a. Explain how we can use reinforcement learning to actually improve performance of existing BC agents or train agents from scratch.
	
	b. In `reinforce_demo` fill out the cofde in `demo.py` using that from `run.py` and then run `python3 run.py`. The script will then evaluate the policy gradient algorithm on CartPole for a number of episodes and after 300 episodes it will plot the progress of REINFORCE. Since REINFORCE is an unstable algorithm, expect that occasionally it will learn and then unlearn the policy.
	

5. Actor Critic Demo.

	a. Explain actor critic algorithms. @Russ You'll need to make some slides for these.
	
	
	c. Then explain vanilla action-value actor critic methods (these are basically policy gradient but using TD learning to estimate the action value and then updating the actor.)
	
	
	b. In `actor_critic_demo` you will fill out the code in `demo.py`. In particular you should fill out `create_critic_training`, `create_actor_training`, and the second half of the `learn` method. Then when you're done you can run `python3 run.py` and this should train the action-value actor-critic methods. Generally these train to about 2000 episodes, maybe 10 minutes or so. I've included some reward plots for you to use in the `actor_critic_demo` folder.


## Slides

I presume you will want to use your own slides for all of these tutorials. I have included my slides which explain tensorflow and gym below, but they do not have anything on REINFORCE or actor-critic methods. Additionally, I've attached the video from last year's recitation so you can see an example of how I present everything. 


Video: https://www.youtube.com/watch?v=0rsrDOXsSeM&t=1107s 
Slides: https://drive.google.com/open?id=1Sw-0wk6nhzSN5X-nk4GsvPyEvWF---lQ77FV1Nhedk8

## How to present the demos.
For the coding demos, I make a print out of `complete.py` (for each of the demo folders found in this repo.) 
Then on my laptop I open up `demo.py` and fill in the relevant empty methods. You should be able to figure out which
need to be done for each individual demo.  Finally, when I want to run it, instead of risking the mistakes I made, 
I just run `complete.py`. 

Note: In the case of reinforce and actor-critic I do the same except I run `run.py`.

## Demo Setup.
To the run the demos you'll need to install `python3` and `pip3` (you can try `python` 2.7 with `pip`, but I'm not sure if the demos will run.)  To do this, please checkout  https://realpython.com/installing-python/

Once you've installed python, you'll need to install various different packages to get the whole demo to run. In particular, you'll need to run
```bash
pip3 install gym
pip3 install tensorboard
pip3 install tensorflow
pip3 install getch
pip3 install matplotlib
```
