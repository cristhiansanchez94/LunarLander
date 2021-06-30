import gym 
import os 
from stable_baselines import results_plotter
from stable_baselines.bench import Monitor
from callback_class import  SaveOnBestTrainingRewardCallback
from stable_baselines import DQN
from stable_baselines.common.evaluation import evaluate_policy 
import matplotlib.pyplot as plt
from array2gif import write_gif
import numpy as np

def train_model(env_name:str, time_steps:str,model_name:str,callback_freq: int):
    log_dir = os.getcwd()

    # Create environment
    env = gym.make(env_name)
    env = Monitor(env, log_dir)
    callback = SaveOnBestTrainingRewardCallback(check_freq=callback_freq, log_dir=log_dir,filename =model_name)

    # Instantiate the agent
    model = DQN('MlpPolicy', env, learning_rate=1e-3, prioritized_replay=True, verbose=1)
    # Train the agent
    model.learn(total_timesteps=int(time_steps),callback = callback)

    results_plotter.plot_results([log_dir], time_steps, results_plotter.X_TIMESTEPS, model_name)
    plt.show()

    # Load the best trained agent
    model = DQN.load(callback.save_path+'/'+model_name,env=gym.make(env_name))

    # Evaluate the agent
   # mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)

    return model

def generate_gif(gif_name: str, model):
    images = []
    obs = model.get_env().reset()
    img = model.env.render(mode='rgb_array')
    for _ in range(350):
        images.append(img)
        action, _ = model.predict(obs)
        obs, _, _ ,_ = model.env.step(action)
        img = model.env.render(mode='rgb_array')
    images_pos = [np.concatenate([img[:,:,0].reshape([1,400,600]),img[:,:,1].reshape([1,400,600]),img[:,:,2].reshape([1,400,600])],axis=0) for i,img in enumerate(images) if i%2==0]
    write_gif(images_pos,'{}.gif'.format(gif_name),fps=29)