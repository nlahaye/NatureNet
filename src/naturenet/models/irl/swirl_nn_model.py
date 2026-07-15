import jax
import jax.numpy as jnp

from flax import linen as nn
from flax.training import train_state

import optax

class MLP(nn.Module):
    hidden_size: int
    n_hidden: int
    n_actions: int

    def setup(self):
        self.dense1 = nn.Dense(self.hidden_size)
        #self.dense2 = nn.Dense(int(self.hidden_size/8))
        #self.dense3 = nn.Dense(int(self.hidden_size/16))
        self.dense4 = nn.Dense(self.n_hidden*self.n_actions)

    def __call__(self, x):
        x = x.astype(jnp.bfloat16)
        x = self.dense1(x)
        x = nn.leaky_relu(x)
        #x = self.dense2(x)
        #x = nn.leaky_relu(x)
        #x = self.dense3(x)
        #x = nn.leaky_relu(x)
        x = self.dense4(x) 
        X = nn.tanh(x)
        x = x.reshape((self.n_hidden, self.n_actions))
        return x

# Create the model
def create_model(rng, n_states, n_hidden, hidden_size, n_actions):
    model = MLP(hidden_size=hidden_size, n_hidden=n_hidden, n_actions=n_actions)
    input_size = n_states*n_states
    params = model.init(rng, jnp.ones((1, input_size)))['params']
    return model, params



# Training state to hold the model parameters and optimizer state
def create_train_state(rng, learning_rate, n_states, n_hidden, hidden_size, n_actions):
    model, params = create_model(rng, n_states, n_hidden, hidden_size, n_actions)
    tx = optax.adam(learning_rate)  # Adam optimizer
    state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)
    return state



