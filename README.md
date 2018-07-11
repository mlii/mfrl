# Mean Field Multi-Agent Reinforcement Learning 

A PyTorch implementation of MF-Q in the paper [Mean Field Multi-Agent Reinforcement Learning ](https://arxiv.org/pdf/1802.05438.pdf).

## Example

![image](https://github.com/mlii/mfrl/blob/master/resources/line.gif)
 
 An 20*20 Ising model example under the low temperature.

## Code structure

- `main_MFQ_Ising.py`: contains code for running tabular based MFQ for Ising model.

- `./examples/`: contains scenarios for Ising Model and Battle Game (also models).

- `battle.py`: contains code for running Battle Game with trained model

- `train_battle.py`: contains code for training Battle Game models

## Compile MAgent platform and run

Before running Battle Game environment, you need to compile it. You can get more helps from: [MAgent](https://github.com/geek-ai/MAgent)

**Steps for compiling**

```shell
cd examples/battle_model
./build.sh
```

**Steps for training models under Battle Game settings**

1. Add python path in your `~/.bashrc` or `~/.zshrc`:

    ```shell
    vim ~/.zshrc
    export PYTHONPATH=./examples/battle_model/python:${PYTHONPATH}
    source ~/.zshrc
    ```

2. Run training script for training (e.g. mfac):

    ```shell
    python3 train_battle.py --algo mfac
    ```

    or get help:

    ```shell
    python3 train_battle.py --help
    ```


## Paper citation

If you found it helpful, consider citing the following paper:

<pre>
@article{2018arXiv180205438Y,
   author = {{Yang}, Y. and {Luo}, R. and {Li}, M. and {Zhou}, M. and {Zhang}, W. and 
	{Wang}, J.},
   title = "{Mean Field Multi-Agent Reinforcement Learning}",
   journal = {ArXiv e-prints},
   eprint = {1802.05438},
   year = 2018,
   month = feb
}
</pre>
