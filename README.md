# Reinforcement Learning Project 

- Alexandre Gonçalves
- Gaspar Pereira
- João Henriques
- Rita Wang
- Victoria Goon
  
### Results

| Environment    | Algorithm   | Average Return | Std. Dev. | Success Rate |
| -------------- | ----------- | -------------- | --------- | ------------ |
| **LunarLander-v3** |           |                |           |              |
|                | DQN         | 195.57         | 71.07     | 67%          |
|                | Rainbow-DQN | **262.36**     | 49.81     | **93%**      |
|                | PPO         | 172.96         | 102.27    | 66.7%        |
| **CarRacing-v3** |           |                |           |              |
|                | PPO         | **874.59**     | 167.95    | **40%**      |
|                | SAC         | 848.38 (sotch.)  | 145.29    | 30% (stoch.) |

Trained for 500,000 timesteps for each. <br>
Evaluated for 30 episodes <br>

Success Rate definition - proportion of episodes with reward over:  
- Lunar Lander: > 200
- Car Racing: > 900
  
### LunarLander-v3 - Rainbow-DQN
https://github.com/user-attachments/assets/2ee71215-b8a5-434f-863c-0aa7584517cb

### LunarLander-v3 - PPO
https://github.com/user-attachments/assets/eb7c8b9b-1531-4d06-a6be-0c2faa76a8c7


### CarRacing-V3 - SAC 
Evaluated with a stochastic policy

https://github.com/user-attachments/assets/b09baaa2-e97f-419b-b060-9d86c31fa2dc

### CarRacing-V3 - PPO
Evaluated with a deterministic policy

https://github.com/user-attachments/assets/2bde3989-2aa6-4c98-ac3a-646e8294e8be



