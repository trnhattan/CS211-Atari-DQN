# 📚 CS211-Atari-DQN
This is an assignment of **CS221-Artificial Intelligent**

## 📂 Tasks:
* **Implement and evaluation 03 algorithms:**
  * Deep Q-Network (DQN)
  * Double DQN
  * Dueling DQN

* **Benchmark on 2 games:**
  * Breakout
  * Space Invaders

## 📋 Used models:
* **DQN and Double DQN**
![CNN Archietecture of DQN and Double DQN](https://user-images.githubusercontent.com/63542739/168975491-3bad06f1-8d69-4395-a567-645cbfc7e670.png)

* **Dueling DQN**
![CNN Archietecture of Dueling DQN](https://user-images.githubusercontent.com/63542739/168975895-74efb49b-210e-4f47-9ccb-576200b58470.png)

## 📈 Results:
To make it fair, I have trained 2 games with 2 different time and steps:
* **Breakout:** Avg 7 hours and 1m steps
* **SpaceInvader:** Avg 10 hours and 2m steps

And achieved the results below:
|      **Model/Game**      | **Breakout** | **SpaceInvaders** |
|:------------------------:|:------------:|:-----------------:|
| **Deep Q-Network (DQN)** |      404     |      **2865**     |
|           **Double DQN** |      414     |        2360       |
|          **Dueling DQN** |    **425**   |        2165       |

## 📌 Usage:
### Training:
```bash
usage: dqn.py [-h] [--model MODEL] 
                   [--env_id ENV_ID] 
                   [--resume RESUME] 
                   [--file_weight_path FILE_WEIGHT_PATH] 
                   [--file_saveName FILE_SAVENAME] 
                   [--run_time RUN_TIME]
                   [--wandb_project WANDB_PROJECT]
                   [--wandb_entity WANDB_ENTITY] 
                   [--wandb_session WANDB_SESSION]
                   [--wandb_id WANDB_ID]
                   [--wandb_resume WANDB_RESUME]
                   [--wandb_relogin WANDB_RELOGIN]
                   [--device DEVICE]

optional arguments:
  -h, --help            show this help message and exit
  --model MODEL         select model: base, double, dueling
  --env_id ENV_ID       enviroment id
  --resume RESUME       continue traning
  --file_weight_path FILE_WEIGHT_PATH
                        pretrained weight path
  --file_saveName FILE_SAVENAME
                        weight file name
  --wandb_project WANDB_PROJECT
                        wandb project name
  --wandb_entity WANDB_ENTITY
                        wandb username
  --wandb_session WANDB_SESSION
                        wandb running title
  --wandb_id WANDB_ID   wandb running id
  --wandb_resume WANDB_RESUME
                        wandb continue existed seasion_name executed
  --wandb_relogin WANDB_RELOGIN
                        wandb force relogin
  --device DEVICE       select GPU or CPU for session
```

### Model observe:
```bash
usage: observe.py [-h] model env_id model_weight_path

positional arguments:
  model              select model: base, double, dueling
  env_id             enviroment id
  model_weight_path  model weight file path

optional arguments:
  -h, --help         show this help message and exit
```

## 🔖 Reference:
The baseline for this repository was based on [these](https://www.youtube.com/watch?v=NP8pXZdU-5U&list=PLZeihsNsdQdRdhni8U5KIdxsRIicW498s) videos. You can check this for more details.
