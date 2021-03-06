import os
import logging
import wandb

logging.basicConfig(format="%(message)s", level=logging.INFO)
LOGGER = logging.getLogger('Deep-Q-Learning-Network')

def init_wandb(project="Deep-Q-Learning-Network", entity='devzxje', name="DQN-v1", id="DQN-v1", session_resume=None, relogin=False):
    if relogin:
        os.system('wandb login --relogin')
    else:
        os.system('wandb login')

    if session_resume is not None:
        LOGGER.info(colorstr('black', 'bold', f'Continue last run'))
        wandb.init(project=project, entity=entity, id=id, resume=session_resume)
    else:
        wandb.init(project=project, entity=entity, name=name, id=id)

    return wandb

def colorstr(*input):
    *args, string = input if len(input) > 1 else ('blue', 'bold', input[0]) 
    colors = {'black': '\033[30m',  # basic colors
              'red': '\033[31m',
              'green': '\033[32m',
              'yellow': '\033[33m',
              'blue': '\033[34m',
              'magenta': '\033[35m',
              'cyan': '\033[36m',
              'white': '\033[37m',
              'bright_black': '\033[90m',  # bright colors
              'bright_red': '\033[91m',
              'bright_green': '\033[92m',
              'bright_yellow': '\033[93m',
              'bright_blue': '\033[94m',
              'bright_magenta': '\033[95m',
              'bright_cyan': '\033[96m',
              'bright_white': '\033[97m',
              'end': '\033[0m',  # misc
              'bold': '\033[1m',
              'underline': '\033[4m'}
    return ''.join(colors[x] for x in args) + f'{string}' + colors['end']
