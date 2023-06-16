import argparse
import numpy
import json
import utils
from utils import device
import hashlib
import torch
import numpy as np
from copy import deepcopy


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NumpyEncoder, self).default(obj)



def hash_state(state):
    state_string = json.dumps(state, cls=NumpyEncoder, sort_keys=True).encode('utf-8')
    return hashlib.sha256(state_string).hexdigest()

def get_next_states(env, state, agent):
    """
    Returns all possible next states given current state
    """
    # find all possible actions
    action_space = env.action_space.n

    # initialize next_states list
    next_states = []

    # get next state for each action
    for action in range(action_space):
        # Create a new environment instance
        env_new = deepcopy(env)


        obs_new, _, _, _, _ = env_new.step(action)
        next_states.append(hash_state(obs_new))

    return set(next_states)

def dfs(env, state, agent, depth):
    if depth == 0:
        return [hash_state(state)]
    
    next_states = get_next_states(env, state, agent)

    # for each of the next states, get their next states and append them to all_states

    all_states = []

    for next_state in next_states:
        all_states.extend(dfs(env, next_state, agent, depth=depth-1))
    
    return all_states

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--env", required=True,
                    help="name of the environment to be run (REQUIRED)")
parser.add_argument("--model", required=True,
                    help="name of the trained model (REQUIRED)")
parser.add_argument("--seed", type=int, default=0,
                    help="random seed (default: 0)")
parser.add_argument("--shift", type=int, default=0,
                    help="number of times the environment is reset at the beginning (default: 0)")
parser.add_argument("--argmax", action="store_true", default=False,
                    help="select the action with highest probability (default: False)")
parser.add_argument("--pause", type=float, default=0.1,
                    help="pause duration between two consequent actions of the agent (default: 0.1)")
parser.add_argument("--gif", type=str, default=None,
                    help="store output as gif with the given filename")
parser.add_argument("--episodes", type=int, default=1000000,
                    help="number of episodes to visualize")
parser.add_argument("--memory", action="store_true", default=False,
                    help="add a LSTM to the model")
parser.add_argument("--text", action="store_true", default=False,
                    help="add a GRU to the model")

args = parser.parse_args()

# Set seed for all randomness sources
utils.seed(args.seed)

# Set device
print(f"Device: {device}\n")

# Load environment
env = utils.make_env(args.env, args.seed)
for _ in range(args.shift):
    env.reset()
print("Environment loaded\n")

# Load agent
model_dir = utils.get_model_dir(args.model)
agent = utils.Agent(env.observation_space, env.action_space, model_dir,
                    argmax=args.argmax, use_memory=args.memory, use_text=args.text)
print("Agent loaded\n")

# Run the agent
if args.gif:
    from array2gif import write_gif
    frames = []


for episode in range(args.episodes):
    obs, _ = env.reset()

    while True:


        action = agent.get_action(obs)
        obs, reward, terminated, truncated, _ = env.step(action)

        # next_states now contains a dictionary of possible states at t+1, t+2, etc.        
        next_states =  dfs(env, obs, agent, depth=3)
        print(len(next_states))

        done = terminated | truncated
        agent.analyze_feedback(reward, done)

        if done:
            break

if args.gif:
    print("Saving gif... ", end="")
    write_gif(numpy.array(frames), args.gif+".gif", fps=1/args.pause)
    print("Done.")

# num_states_per_step now contains the number of unique states at each step