# ModelFree

### Some comments
- A2C not fully there yet, TRPO not there yet
- in general, if we record a transition up to "Done" or if we update as soon as we reach "Done", the info collected is very little. Done is reached after 1/2 transitions, how to prevent this?

### Write up & reference
- [Model updating after interventions paradoxically introduces bias](http://proceedings.mlr.press/v130/liley21a/liley21a.pdf) with [suppl](http://proceedings.mlr.press/v130/liley21a/liley21a-supp.pdf)
- [Model update - write up](https://www.overleaf.com/read/yhntntbxtrtb)
- [recent model-free & model-based blend](https://www.overleaf.com/read/skwrxkyysvvc)

### Environments
The environments created follow the OpenAI Gym architecture. To use it:
- git clone https://github.com/claudia-viaro/gym-update.git
- cd gym-update
- !pip install gym-update
- import gym
- import gym_update
- env =gym.make('update-v0')

