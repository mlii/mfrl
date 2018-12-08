"""Battle
"""

import argparse
import os
import tensorflow as tf
import numpy as np
import magent

from examples.battle_model.algo import spawn_ai
from examples.battle_model.algo import tools
from examples.battle_model.senario_battle import battle


BASE_DIR = os.path.dirname(os.path.abspath(__file__))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--algo', type=str, choices={'ac', 'mfac', 'mfq', 'il'}, help='choose an algorithm from the preset', required=True)
    parser.add_argument('--oppo', type=str, choices={'ac', 'mfac', 'mfq', 'il'}, help='indicate the opponent model')
    parser.add_argument('--n_round', type=int, default=50, help='set the trainning round')
    parser.add_argument('--render', action='store_true', help='render or not (if true, will render every save)')
    parser.add_argument('--map_size', type=int, default=40, help='set the size of map')  # then the amount of agents is 64
    parser.add_argument('--max_steps', type=int, default=400, help='set the max steps')
    parser.add_argument('--idx', nargs='*', required=True)

    args = parser.parse_args()

    # Initialize the environment
    env = magent.GridWorld('battle', map_size=args.map_size)
    env.set_render_dir(os.path.join(BASE_DIR, 'examples/battle_model', 'build/render'))
    handles = env.get_handles()

    tf_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    tf_config.gpu_options.allow_growth = True

    main_model_dir = os.path.join(BASE_DIR, 'data/models/{}-0'.format(args.algo))
    oppo_model_dir = os.path.join(BASE_DIR, 'data/models/{}-1'.format(args.oppo))

    sess = tf.Session(config=tf_config)
    models = [spawn_ai(args.algo, sess, env, handles[0], args.algo + '-me', args.max_steps), spawn_ai(args.oppo, sess, env, handles[1], args.oppo + '-opponent', args.max_steps)]
    sess.run(tf.global_variables_initializer())

    models[0].load(main_model_dir, step=args.idx[0])
    models[1].load(oppo_model_dir, step=args.idx[1])

    runner = tools.Runner(sess, env, handles, args.map_size, args.max_steps, models, battle, render_every=0)
    win_cnt = {'main': 0, 'opponent': 0}

    for k in range(0, args.n_round):
        runner.run(0.0, k, win_cnt=win_cnt)

    print('\n[*] >>> WIN_RATE: [{0}] {1} / [{2}] {3}'.format(args.algo, win_cnt['main'] / args.n_round, args.oppo, win_cnt['opponent'] / args.n_round))
