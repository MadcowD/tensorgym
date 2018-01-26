import gym


def main():
	# Make the environment
	env = gym.make('MountainCar-v0')

	# Record the environment
	env = gym.wrappers.Monitor(env, '.', force=True)

	for episode in range(100):
		done = False
		obs = env.reset()

		while not done: # Start with while True
			env.render()

			action = env.action_space.sample()
			obs, reward, done, info = env.step(action)



if __name__ == '__main__':
	main()