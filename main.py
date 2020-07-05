from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import RIGHT_ONLY
env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0')
env = JoypadSpace(env, RIGHT_ONLY)

from random import randint
import numpy as np
import os
import tensorflow as tf
from model import generate_model

model_file_path = './tensormario'
if os.path.exists(model_file_path):
    model = tf.keras.models.load_model(model_file_path)
else:
    model = generate_model()

done = True
last_state = None
identity = np.identity(env.action_space.n) 

while 1:
    for step in range(5000):
        if done:
            state = env.reset()

        if randint(0, 6) == 0 or not isinstance(last_state, (np.ndarray, np.generic)):
            action = env.action_space.sample()
        else:
            action = np.argmax(model.predict(np.expand_dims(last_state, axis=0)))

        state, reward, done, info = env.step(action)
        last_state = state
        if reward > 0:
            model.train_on_batch(x=np.expand_dims(last_state, axis=0), y=identity[action: action+1])

        env.render()
    model.save(model_file_path)

env.close()