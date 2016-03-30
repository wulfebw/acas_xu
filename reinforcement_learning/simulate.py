
import math
import sys
import time

import qnetwork
import plot_acas_xu
import replay_memory
import state_generator
from state_generator import K_SIZE

TRAIN_FREQ = 30 
GAMMA = 0.95
SOLVER = 'adamax'

# Logging Constants
PRINT_FREQ = 3000
ICON_FILE = "../media/AirplaneIcon.png"

def simulate():

    draw = False
    print 'building network...'
    if draw:
        pltAcas = plot_acas_xu.Plot_ACAS_XU(state_generator.RMAX, ICON_FILE, 1)
    sg = state_generator.StateGenerator(state_generator.RMAX, state_generator.RMIN, state_generator.VMIN, state_generator.VMAX, K_SIZE)
    q = qnetwork.QNetwork(state_generator.NUM_INPUTS, replay_memory.BATCH_SIZE, state_generator.NUM_ACTIONS, GAMMA, SOLVER)
    repMem = replay_memory.ReplayMemory()
    count = 0

    dt = state_generator.DT
    dti = state_generator.DTI
    state = sg.randomStateGenerator()
    i = 0
    print 'starting training...'
    while True:

        for j in range(TRAIN_FREQ):
            i += 1
            action = q.getAction(state)
            nextStates, rewards = sg.getNextState(state, action, dt, dti)
            stateNorm, nextStateNorm = sg.normState(state, nextStates)
            repMem.store((stateNorm, action, rewards, nextStateNorm))
            state = nextStates[0]
            count += 1
            if draw:
                pltAcas.updateState(state, action)
                pltAcas.draw()
                time.sleep(0.3)

            if sg.checkRange(state) or i > 100:
                i = 0
                state = sg.randomStateGenerator()

        if count % PRINT_FREQ == 0 and count >= replay_memory.INIT_SIZE:
            print "Samples: %d, Trainings: %d" % (count, (count - replay_memory.INIT_SIZE) / TRAIN_FREQ), "Loss: %.3e" % q.test(repMem.sample_batch())
            sys.stdout.flush()

        elif (count % 10000 == 0):
            print"Samples: %d" % count
            sys.stdout.flush()
            
        q.train(repMem.sample_batch())

        # pltAcas.updateState(state,action)
        # pltAcas.draw()

if __name__ == '__main__':
    simulate()
