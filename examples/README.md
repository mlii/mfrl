# Example Scenarios

Environments and scenarios for Ising Model.

## Code structure

- `./ising_model/Ising.py`: contains code for Ising Model scenario.

- `./ising_model/multiagent/environment.py`: contains code for environment simulation (_step() function, etc.).

- `./ising_model/multiagent/core.py`: contains classes for objects (Entities, Agents, etc.) that are used throughout the code.

- `./battle_model/`: contains engine files of MAgent and implementation of the algorithms

## Compile Ising environment and run

**Requirements**
- `python==3.6.1`
- `gym==0.9.2` (might work with later versions)
- `matplotlib` if you would like to produce Ising model figures