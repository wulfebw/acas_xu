
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
import keras
import numpy as np
import theano
import theano.tensor as T

import replay_memory
import state_generator
from state_generator import K_SIZE

TARGET_UPDATE_FREQ = 100
FINAL_EXPLORATION = 0.1
FINAL_EXPLORATION_FRAME = 75000000000.0  

SAVE_FREQ = 2000
SAVE_FILENAME = '../models/snapshots/{}.h5'

class QNetwork(object):

    def __init__(self, input_shape, batch_size, num_actions, discount, update_rule, rng=0):
        self.input_shape = input_shape
        self.batch_size = batch_size
        self.num_actions = num_actions
        self.discount = discount
        self.update_rule = update_rule

        self.rng = rng if rng else np.random.RandomState()

        self.actions = state_generator.ACTIONS
        self.saveFreq = SAVE_FREQ
        self.replayStartSize = replay_memory.INIT_SIZE
        self.finalExploration = FINAL_EXPLORATION
        self.finalExplorationSize = FINAL_EXPLORATION_FRAME
        self.targetNetworkUpdateFrequency = TARGET_UPDATE_FREQ

        self.initialize_network()
        self.update_counter = 0
        self.counter = -1.0
        self.K_SIZE = K_SIZE
        self.COC = state_generator.COC

    def getAction(self, state):
        self.counter += 1
        if self.counter < self.replayStartSize:
            return self.actions[self.rng.randint(self.num_actions)]
        else:
            num = self.rng.rand()
            actInd = 0
            if num >= np.min([(self.counter - self.replayStartSize), self.finalExplorationSize]) / self.finalExplorationSize * (1 - self.finalExploration):
                actInd = self.rng.randint(self.num_actions)
            else:
                actInd = np.argmax(self.model.predict(state.reshape(
                    1, state.shape[0]), batch_size=1, verbose=0))
            return self.actions[actInd]

    def initialize_network(self):

        def both(y_true, y_pred):
            d = y_true - y_pred
            #c = 0*d+0.01
            #d = T.switch(abs(d)>0.01,c,d)
            a = d**2
            b = 0
            l = T.switch(y_true < 0, a, b)
            cost1 = l  # T.sum(l, axis=-1)

            return cost1

        target = Sequential()

        target.add(Dense(128, input_dim=self.input_shape,
                         init='uniform', activation='relu'))
        # target.add(Dense(512, init='uniform', activation='relu'))
        # target.add(Dense(512, init='uniform', activation='relu'))
        # target.add(Dense(128, init='uniform', activation='relu'))
        # target.add(Dense(128, init='uniform', activation='relu'))
        target.add(Dense(128, init='uniform', activation='relu'))
        target.add(Dense(self.num_actions, init='uniform'))

        target.compile(loss=both, optimizer=self.update_rule)

        model = Sequential()
        model.add(Dense(128, input_dim=self.input_shape,
                        init='uniform', activation='relu'))
        # model.add(Dense(512, init='uniform', activation='relu'))
        # model.add(Dense(512, init='uniform', activation='relu'))
        # model.add(Dense(128, init='uniform', activation='relu'))
        # model.add(Dense(128, init='uniform', activation='relu'))
        model.add(Dense(128, init='uniform', activation='relu'))
        model.add(Dense(self.num_actions, init='uniform'))

        model.compile(loss=both, optimizer=self.update_rule)
        self.target = target
        self.model = model

    def train(self, (states, actions, rewards, next_states)):

        if np.size(states) == 1:
            return
        if self.update_counter % self.saveFreq == 0:
            self.saveModel()
        if self.update_counter % self.targetNetworkUpdateFrequency == 0:
            self.reset_target_network()
        self.update_counter += 1

        modelValues = np.zeros((np.size(actions), self.num_actions)) + 1.0
        q_target = self.target.predict(next_states, batch_size=512)
        for i in range(len(actions)):
            q_target_temp = np.mean(q_target[i * K_SIZE:(i + 1) * K_SIZE], axis=0)
            indTarget = np.argmax(q_target_temp)
            indModel = int(actions[i] * 18.0 / np.pi) + 2
            if actions[i] == self.COC:
                indModel = 5

            reward = np.mean(rewards[i * K_SIZE:(i + 1) * K_SIZE])
            modelValues[i, indModel] = reward + \
                self.discount * q_target_temp[indTarget]

        self.model.train_on_batch(states, modelValues)

    def reset_target_network(self):
        self.target.set_weights(self.model.get_weights())

    def getModel(self):
        return self.model

    def getTarget(self):
        return self.target

    def saveModel(self):
        self.model.save_weights(SAVE_FILENAME.format(self.update_counter), overwrite=True)

    def test(self, (states, actions, rewards, next_states)):
        if np.size(states) == 1:
            return -1
        q_model = self.model.predict(states, batch_size=512)
        loss = 0.0
        q_target = self.target.predict(next_states, batch_size=512)
        for i in range(len(actions)):
            indModel = int(actions[i] * 18.0 / np.pi) + 2
            if actions[i] == self.COC:
                indModel = 5

            reward = np.mean(rewards[i * K_SIZE:(i + 1) * K_SIZE])
            q_target_temp = np.mean(q_target[i * K_SIZE:(i + 1) * K_SIZE], axis=0)
            indTarget = np.argmax(q_target_temp)
            loss += (q_model[i, indModel] - reward -
                     self.discount * q_target_temp[indTarget])**2
        return loss / len(q_model)
