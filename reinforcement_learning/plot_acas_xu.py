

import matplotlib.pyplot as plt
import numpy as np
from IPython import display
import scipy

import state_generator

class Plot_ACAS_XU(object):

    def __init__(self, maxSize, iconfile, iconSize):
        self.maxSize = maxSize
        self.icon = scipy.ndimage.imread(iconfile)
        self.iconSize = iconSize
        self.r = 0
        self.theta = 0
        self.psi = 0
        self.vOwn = 0
        self.vInt = 0
        self.action = 0

        self.fig = plt.figure()
        self.axIntNum = 2
        ax = self.fig.add_axes([0, 0, 0.8, 1])
        axOwn = self.fig.add_axes([0.4 - 0.03 * self.iconSize,
                                   0.5 - 0.03 * self.iconSize, 0.06 * self.iconSize, 0.06 * self.iconSize])
        axInt = self.fig.add_axes([0.1, 0.1, 0.1, 0.1])
        axText = self.fig.add_axes([0.8, 0.4, 0.3, 0.6])

        axOwn.imshow(self.icon)
        axOwn.set_xticks([])
        axOwn.set_yticks([])
        axOwn.axis('off')

        axInt.imshow(self.icon)
        axInt.set_xticks([])
        axInt.set_yticks([])
        axInt.axis('off')

        axText.set_xticks([])
        axText.set_yticks([])
        axText.axis('off')

        ax.set_xlim((-self.maxSize, self.maxSize))
        ax.set_ylim((-self.maxSize, self.maxSize))
        an = np.linspace(0, 2 * np.pi, 100)
        ax.plot(self.maxSize * np.cos(an), self.maxSize * np.sin(an), 'g')
        ax.set_aspect('equal')

    def updateState(self, state, action):
        self.r = state[0]
        self.theta = state[1]
        self.psi = state[2]
        self.vOwn = state[3]
        self.vInt = state[4]
        self.action = action

    def draw(self):
        # self.fig.delaxes(self.fig.axes[2])
        x = self.r * math.cos(self.theta)
        y = self.r * math.sin(self.theta)

        size = self.iconSize * (abs(np.sin(self.psi)) + abs(np.cos(self.psi)))
        num = self.axIntNum
        self.fig.axes[num].set_position([x / self.maxSize / 3.0 + 0.4 - 0.03 * size,
                                         y / self.maxSize / 2 + 0.5 - 0.03 * size, 0.06 * size, 0.06 * size])
        self.fig.axes[num].cla()
        self.fig.axes[num + 1].cla()

        textOwn = 'Speed Own: %.2f' % self.vOwn
        textInt = 'Speed Intruder: %.2f' % self.vInt
        textAct = 'Action: %d' % (self.action * 180 / np.pi)
        textHeading = 'Intruder Heading: %.1f' % (self.psi * 180 / np.pi)

        self.fig.axes[num + 1].text(0.0, 0.8, textOwn,
                                    verticalalignment='bottom', horizontalalignment='left',
                                    color='g', fontsize=14)
        self.fig.axes[num + 1].text(0.0, 0.6, textInt,
                                    verticalalignment='bottom', horizontalalignment='left',
                                    color='g', fontsize=14)
        self.fig.axes[num + 1].text(0.0, 0.4, textAct,
                                    verticalalignment='bottom', horizontalalignment='left',
                                    color='g', fontsize=14)
        self.fig.axes[num + 1].text(0.0, 0.2, textHeading,
                                    verticalalignment='bottom', horizontalalignment='left',
                                    color='g', fontsize=14)

        dst = scipy.ndimage.rotate(self.icon, self.psi * 180 / np.pi, cval=255)
        self.fig.axes[num].imshow(dst)

        self.fig.axes[num].set_xticks([])
        self.fig.axes[num].set_yticks([])
        self.fig.axes[num].axis('off')

        self.fig.axes[num + 1].set_xticks([])
        self.fig.axes[num + 1].set_yticks([])
        self.fig.axes[num + 1].axis('off')

        display.clear_output(wait=True)
        display.display(plt.gcf())
