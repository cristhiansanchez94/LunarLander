# LunarLander
Repository to host files related to the LunarLander experiment on Open AI Gym.


The following files were implemented: 
- LunaTrainer: This file contains the functions used to train the model in a specific environment. Also, it contains a function that generates the gif file. 
- LunaLauncher: The folder with the files of the environment where the rocket is launched instead of landed. The specific folder LunarLauncher/envs/ contains the file with the modifications to the LunarLander environment so the rocket flies to a specific position. 
The main difference between this environment and the previous one is that the rocket starts at the landing area, and by changing the reward function, the structure of the environment and the target point, the goal of launching the rocket is achieved. 

Both environments were implemented in a jupyter notebook that is not included in the code repository. 
