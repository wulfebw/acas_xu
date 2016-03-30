
import math
import numpy as np

# MDP Constants
SIGMA_ACT_CONFLICT = 0.00000000001  # 4.0/180.0*np.pi
SIGMA_ACT_NO_CONFLICT = 0.00000000001  # 10.0/180.0*np.pi
SIGMA_V = 0.0000000001  # 2.0

# DT is the time difference between states
DT = 5
# DTI is the stepsize in incrementing time
# this exists to make the simulation more realistic
DTI = 0.5

# reward constants
LAMDA1 = 1
LAMDA2 = 0.0
LAMDA3 = 0.0002
LAMDA4 = 0.01

ACTIONS = np.array([-20.0 / 180.0 * np.pi, -10.0 / 180.0 * np.pi, 0.0,
                    10.0 / 180.0 * np.pi, 20.0 / 180.0 * np.pi, 1.0 / 180.0 * np.pi])

# Clear of conflict
COC = 1.0 / 180.0 * np.pi

# max and min distances between aircraft
RMAX = 3000.0
RMIN = 500.0

# min and max velocities
VMIN = 10.0
VMAX = 20.0
NUM_INPUTS = 5
NUM_ACTIONS = 6

# means and ranges for normalizing inputs
# not sure if these make sense since the mean probably is not 1500 right?
MEANS = np.array([1500.0, 0.0, 0.0, 15.0, 15.0])
RANGES = np.array([3000.0, 2 * np.pi, 2 * np.pi, 10.0, 10.0])
SIG_PARAM = 1.0 / 100.0

K_SIZE = 1

class StateGenerator(object):

    def __init__(self, rMax, rMin, speedMin, speedMax, K_SIZE):
        self.rMax = rMax
        self.rMin = rMin
        self.speedMin = speedMin
        self.speedMax = speedMax
        self.sigmaActConflict = SIGMA_ACT_CONFLICT
        self.sigmaActNoConflict = SIGMA_ACT_NO_CONFLICT
        self.sigmaV = SIGMA_V
        self.size = K_SIZE

        self.lamda1 = LAMDA1
        self.lamda2 = LAMDA2
        self.lamda3 = LAMDA3
        self.lamda4 = LAMDA4
        self.COC = COC
        self.num_states = NUM_INPUTS
        self.means = MEANS
        self.ranges = RANGES
        self.SIG_PARAM = SIG_PARAM

    def getNextState(self, state, actionCom, dt, dti):
        r0 = state[0]
        th0 = state[1]
        psi0 = state[2]
        vown0 = state[3]
        vint0 = state[4]
        nextStates = np.zeros((self.size, self.num_states))
        rewards = np.zeros(self.size)
        for j in range(self.size):

            i = 0
            rMinimum = r0
            r = r0
            th = th0
            psi = psi0
            vown = vown0
            vint = vint0
            action, vown, vint = self.getTrueValues(actionCom, vown, vint)
            if (vown > self.speedMax):
                vown = self.speedMax
            if vown < self.speedMin:
                vown = self.speedMin
            if (vint > self.speedMax):
                vint = self.speedMax
            if vint < self.speedMin:
                vint = self.speedMin
            while i < dt:
                i += dti

                x2 = r * math.cos(th) + vint * math.cos(psi) * dti
                y2 = r * math.sin(th) + vint * math.sin(psi) * dti
                x1 = vown * dti
                y1 = 0
                psi1 = 9.80 * math.tan(action) / vown * dti
                psi = psi - psi1
                xabs = x2 - x1
                yabs = y2 - y1

                r = math.hypot(yabs, xabs)
                rMinimum = np.min([rMinimum, r])
                th = math.atan2(yabs, xabs) - psi1

                if (psi > math.pi):
                    psi -= 2 * math.pi
                elif (psi < -math.pi):
                    psi += 2 * math.pi
            nextStates[j][0] = r
            nextStates[j][1] = th
            nextStates[j][2] = psi
            nextStates[j][3] = vown
            nextStates[j][4] = vint

            Isep = 0
            Icoc = 0
            if rMinimum < self.rMin:
                Isep = 1
            if actionCom != self.COC:
                Icoc = 1

            # Triangle Reward Shape
            rewards[j] = -self.lamda1 * Isep * (self.rMin - rMinimum) / self.rMin

            # Penalty for banking
            rewards[j] += -self.lamda3 * (actionCom * 180 / np.pi)**2

            # Penalty for alerting
            rewards[j] += -self.lamda4 * Icoc

        return (nextStates, rewards)

    def randomStateGenerator(self):
        state = np.zeros(self.num_states)
        state[0] = abs(np.random.randn()) / 2 * self.rMax
        while (state[0] > self.rMax):
            state[0] = abs(np.random.randn()) / 2 * self.rMax
        state[1] = (np.random.rand() - 0.5) * 2 * np.pi
        state[2] = (np.random.rand() - 0.5) * 2 * np.pi
        state[3] = np.random.rand() * (self.speedMax - self.speedMin) + self.speedMin
        state[4] = np.random.rand() * (self.speedMax - self.speedMin) + self.speedMin
        return state

    def checkRange(self, state):
        if state[0] > self.rMax:
            return True
        return False

    def getTrueValues(self, action, vown, vint):

        if action == self.COC:
            action = np.random.normal(action, self.sigmaActNoConflict, 1)
        else:
            action = np.random.normal(action, self.sigmaActConflict, 1)
        vown = np.random.normal(vown, self.sigmaV, 1)
        vint = np.random.normal(vint, self.sigmaV, 1)
        return (action, vown, vint)

    def normState(self, state, nextState):
        newState = np.zeros(state.shape)
        newNextState = np.zeros(nextState.shape)

        for i in range(5):
            newState[i] = (state[i] - self.means[i]) / self.ranges[i]
            for j in range(K_SIZE):
                newNextState[j][i] = (
                    nextState[j][i] - self.means[i]) / self.ranges[i]
        return (newState, newNextState)

    def unNormState(self, state):
        newState = np.zeros(state.shape)
        for i in range(5):
            newState[i] = state[i] * self.ranges[i] + self.means[i]
        return newState
