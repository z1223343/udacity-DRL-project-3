
## Udacity DRL Project 3: Collaboration & Competition

### Description of Implementation

The environment is solved using a multi-agent deep deterministic policy gradient (DDPG) algorithm. Training proceeds as follows:

1. The 2 agents each receive a different state vector (with 24 elements) from the environment
1. Each agent feeds its state vector through the local actor network to get an action vector (with 2 elements) as output. Noise based on an Ornstein-Uhlenbeck process is added to the predicted actions to encourage exploration
1. Each agent then receives a next state vector and a reward from the environment (as well as a termination signal that indicates if the episode is complete)
- The experience tuple `(state, action, reward, next state)` of each agent is added to a common replay buffer
- A random sample of experience tuples is drawn from the replay buffer (once it contains enough) 
- The sample is used to update the weights of the local critic network:
    1. The next state is fed into the target actor to obtain the next action
    1. The (next state, next action) pair is fed into the target critic to obtain an action value, Q_next
    1. The action value for the current state is then computed as Q_cur = reward + gamma*Q_next
    1. The (current state, current action) pair are fed into the local critic to obtain a predicted action value, Q_pred
    1. The MSE loss is computed between Q_cur and Q_pred, and the weights of the local critic are updated accordingly
- The sample is used to update the weights of the local actor network:
    1. The current state is fed into the local actor to obtain predicted a action
    1. Each (current state, predicted action) pair for the sample is fed into the local critic to obtain action values
    1. The negative mean of the predicted Q values is used as a loss function to update the weights of the local actor
- The target actor and critic networks are then soft-updated by adjusting their weights slightly toward those of their local counterparts
- The states that were obtained in step (3) then become the current states for each agent and the process repeats from step (2)

#### Learning Algorithms

This project considers a multi-agent implementation of the DDPG algorithm.

#### Agent Hyperparameters

- `GAMMA = 0.99`
- `TAU = 0.001` 
- `LR_ACTOR = 0.001` 
- `LR_CRITIC = 0.001`
- `BUFFER_SIZE = 100000` 
- `BATCH_SIZE = 256` 
- `theta = 0.15` `sigma = 0.05` 


#### Network Architectures and Hyperparameters

The actor network takes a state vector (24 elements) as input and returns an action vector (2 elements). It was modelled with a feedforward deep neural network comprising a 24 dimensional input layer, two hidden layers with 128 neurons and ReLU activations and a 2 dimensional output layer with a tanh activation to ensure that the predicted actions are in the range -1 to +1. Batch normalisation was applied to the input and two hidden layers. 

The critic network takes the state and action vectors as input, and returns a scalar Q value as output. It was modelled with a feedforward deep neural network with a 24 dimensional input layer (for the state vector) that was fully connected to 128 neurons in the first hidden layer with ReLU activations. The outputs of the first layer were batch normalised and concatenated with the 2 dimensional action vector as input to the second hidden layer, which also comprised 128 neurons with ReLU activations. Finally, the second hidden layer mapped to an output layer with single neuron and linear activation (outputs a single real number). 


### Results

![results.png](/assets/results.png)


### Future Plans for Improvement

- **Hyperparameter tuning** - I focused on tuning hidden size and add random exploration. Other parameters might be also good candidates and speed up the training.
- **MAPPO** -  I started this project with my own adaptation of PPO for multi agent. It learned something, but wasn't nearly as good as MADDPG. I would like to make it work
- **Try it on other environment** - Soccer environment sounds like a good idea to tackle.


