import os
import sys

sys.path.insert(1, os.path.join(sys.path[0], '..'))
import argparse
from examples.ising_model.multiagent.environment import IsingMultiAgentEnv
import examples.ising_model as ising_model
import numpy as np
import time

np.random.seed(13)

parser = argparse.ArgumentParser(description=None)
parser.add_argument('-n', '--num_agents', default=100, type=int)
parser.add_argument('-t', '--temperature', default=1, type=float)
parser.add_argument('-epi', '--episode', default=1, type=int)
parser.add_argument('-ts', '--time_steps', default=10000, type=int)
parser.add_argument('-lr', '--learning_rate', default=0.1, type=float)
parser.add_argument('-dr', '--decay_rate', default=0.99, type=float)
parser.add_argument('-dg', '--decay_gap', default=2000, type=int)
parser.add_argument('-ac', '--act_rate', default=1.0, type=float)
parser.add_argument('-ns', '--neighbor_size', default=4, type=int)
parser.add_argument('-s', '--scenario', default='Ising.py',
                    help='Path of the scenario Python script.')
parser.add_argument('-p', '--plot', default=0, type=int)
args = parser.parse_args()

# load scenario from script
ising_model = ising_model.load(args.scenario).Scenario()
# create multiagent environment
env = IsingMultiAgentEnv(world=ising_model.make_world(num_agents=args.num_agents,
                                                      agent_view=1),
                         reset_callback=ising_model.reset_world,
                         reward_callback=ising_model.reward,
                         observation_callback=ising_model.observation,
                         done_callback=ising_model.done)

n_agents = env.n
n_states = env.observation_space[0].n
n_actions = env.action_space[0].n
dim_Q_state = args.neighbor_size + 1
act_rate = args.act_rate
n_episode = args.episode
max_steps = args.time_steps
temperature = args.temperature
if_plot = args.plot
lr = args.learning_rate
decay_rate = args.decay_rate
decay_gap = args.decay_gap

if if_plot:
  import matplotlib.pyplot as plt


def boltzman_explore(Q, temper, state, agent_index):
  action_probs_numes = []
  denom = 0
  for i in range(n_actions):
    try:
      val = np.exp(Q[agent_index, state, i] / temper)
    except OverflowError:
      return i
    action_probs_numes.append(val)
    denom += val
  action_probs = [x / denom for x in action_probs_numes]

  return np.random.choice(n_actions, 1, p=action_probs)


folder = "./ising_figs/" + time.strftime("%Y%m%d-%H%M%S") \
         + "-" + str(n_agents) + "-" + str(temperature) \
         + "-" + str(lr) + "-" + str(act_rate) + "/"
if not os.path.exists(folder):
  os.makedirs(folder)

epi_display = []
reward_target = np.array([[2, -2],
                          [1, -1],
                          [0, 0],
                          [-1, 1],
                          [-2, 2]])

for i_episode in range(n_episode):

  obs = env.reset()
  obs = np.stack(obs)

  order_param = 0.0
  max_order, max_order_step = 0.0, 0
  o_up, o_down = 0, 0

  Q = np.zeros((n_agents, dim_Q_state, n_actions))

  if if_plot:
    plt.figure(2)
    plt.ion()
    ising_plot = np.zeros((int(np.sqrt(n_agents)), int(np.sqrt(n_agents))), dtype=np.int32)
    im = plt.imshow(ising_plot, cmap='gray', vmin=0, vmax=1, interpolation='none')
    im.set_data(ising_plot)

  timestep_display = []
  done_ = 0
  current_t = 0.3

  for t in range(max_steps):
    action = np.zeros(n_agents, dtype=np.int32)

    if t % decay_gap == 0:
      current_t *= decay_rate

    if current_t < temperature:
      current_t = temperature

    for i in range(n_agents):
      obs_flat = np.count_nonzero(obs[i] == 1)
      action[i] = boltzman_explore(Q, current_t, obs_flat, i)

    display = action.reshape((int(np.sqrt(n_agents)), -1))

    action_expand = np.expand_dims(action, axis=1)

    obs_, reward, done, order_param, ups, downs = env.step(action_expand)
    obs_ = np.stack(obs_)

    mse = 0
    act_group = np.random.choice(n_agents, int(act_rate * n_agents), replace=False)
    for i in act_group:
      obs_flat = np.count_nonzero(obs[i] == 1)
      Q[i, obs_flat, action[i]] = Q[i, obs_flat, action[i]] \
                                  + lr * (reward[i] - Q[i, obs_flat, action[i]])
      mse += np.power((Q[i, obs_flat, action[i]] - reward_target[obs_flat, action[i]]), 2)

    mse /= n_agents
    obs = obs_

    timestep_display.append(display)

    if order_param > max_order:
      max_order, max_order_step = order_param, t
      o_up, o_down = ups, downs
      if if_plot:
        plt.figure(2)
        ising_plot = display
        im.set_data(ising_plot)
        plt.savefig(folder + '%d-%d-%d-%.3f-%s.png'
                    % (t, ups, downs, order_param, time.strftime("%Y%m%d-%H%M%S")))
      print("+++++++++++++++++++++++++++++")

    if abs(max_order - order_param) < 0.001:
      done_ += 1
    else:
      done_ = 0

    if done_ == 500 or t > max_steps:  # or order_param == 1.0:
      # if the order param stop for 500 steps, then quit
      break

    print('E: %d/%d, reward = %f, mse = %f, Order = %f, Up = %d, Down = %d' %
          (i_episode, t, sum(reward), mse, order_param, ups, downs))

  if if_plot:
    plt.figure(2)
    ising_plot = display
    im.set_data(ising_plot)
    plt.savefig(folder + '%d-%d-%d-%.3f-%s.png'
                % (t, ups, downs, order_param, time.strftime("%Y%m%d-%H%M%S")))

  print('Episode: %d, MaxO = %f at %d (%d/%d)' %
        (i_episode, max_order, max_order_step, o_up, o_down))
  epi_display.append(timestep_display)

np.save(folder + 'display', np.asarray(epi_display))
