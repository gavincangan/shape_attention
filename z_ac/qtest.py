from env.macros import *
from env.gworld import *
from env.visualize import *
from collections import deque

from time import time
import os

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import RMSprop, Adam
from keras.callbacks import TensorBoard

from demo_vis import ShapeAgent


class ImWorldModel:
    def __init__(self):
        self.num_iter = 10000
        self.gamma = 0.975
        self.epsilon = 0.65
        self.batchsize = 40
        self.episode_maxlen = 1000
        self.replay = deque(maxlen=400)
        # self.init_env()
        # self.init_model()

    def init_model(self):
        reward_model = Sequential()
        reward_model.add(Dense(256, kernel_initializer="lecun_uniform", input_shape=(2 * WORLD_W * WORLD_H + Actions.NUM_ACTIONS + Observe.TotalOptions,)))
        reward_model.add(Activation('relu'))

        reward_model.add(Dense(128, kernel_initializer="lecun_uniform"))
        reward_model.add(Activation('relu'))
        reward_model.add(Dropout(0.2))

        reward_model.add(Dense(64, kernel_initializer="lecun_uniform"))
        reward_model.add(Activation('relu'))
        reward_model.add(Dropout(0.2))

        reward_model.add(Dense(27, kernel_initializer="lecun_uniform", activation='relu'))
        reward_model.add(Dropout(0.2))

        reward_model.add(Dense(1, kernel_initializer="lecun_uniform", activation='linear'))

        imworld_model = Sequential()
        imworld_model.add(Dense(256, kernel_initializer="lecun_uniform", input_shape=(2 * WORLD_W * WORLD_H + Actions.NUM_ACTIONS + Observe.TotalOptions,)))
        imworld_model.add(Activation('relu'))

        imworld_model.add(Dense(128, kernel_initializer="lecun_uniform"))
        imworld_model.add(Activation('relu'))
        imworld_model.add(Dropout(0.2))

        imworld_model.add(Dense(64, kernel_initializer="lecun_uniform"))
        imworld_model.add(Activation('relu'))
        imworld_model.add(Dropout(0.2))

        imworld_model.add(Dense(64, kernel_initializer="lecun_uniform"))
        imworld_model.add(Activation('relu'))
        imworld_model.add(Dropout(0.2))

        imworld_model.add(Dense(128, kernel_initializer="lecun_uniform"))
        imworld_model.add(Activation('relu'))
        imworld_model.add(Dropout(0.2))

        imworld_model.add(Dense(256, kernel_initializer="lecun_uniform"))
        imworld_model.add(Activation('relu'))
        imworld_model.add(Dropout(0.2))

        imworld_model.add(Dense(2 * WORLD_W * WORLD_H, kernel_initializer="lecun_uniform", activation='tanh'))

        adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=0.1, decay=0.0)

        reward_model.compile(adam, 'mse')
        imworld_model.compile(adam, 'mse')

        self.reward_model = reward_model
        self.imworld_model = imworld_model
        # model.load_weights(WTS_REWARD_MODEL)

    def save_model(self):
        self.imworld_model.save_weights(WTS_IMWORLD_MODEL)
        self.reward_model.save_weights(WTS_REWARD_MODEL)

    def load_model(self):
        if(os.path.isfile(WTS_IMWORLD_MODEL)):
            self.imworld_model.load_weights(WTS_IMWORLD_MODEL)
        if(os.path.isfile(WTS_REWARD_MODEL)):
            self.reward_model.load_weights(WTS_REWARD_MODEL)

class StepMemory:
    def __init__(self):
        self.state = None
        self.action = None
        self.reward = 0
        self.newstate = None
        self.obs_memory = []

if __name__ == "__main__":

    sa = ShapeAgent()
    sa.init_model()
    sa.load_model()

    im = ImWorldModel()
    im.init_model()
    im.load_model()

    
    im.epsilon = 0.1
    sa.epsilon = 0.1
    
    for i in range(sa.num_iter):
        done_flag = False
        step_count = 0
        sa.init_env()
        agents = sa.env.get_agents()
        print('Iter: %s' %(i) )
        while(step_count < im.episode_maxlen and not done_flag):
            # print '\n>> Count: ', i, ' -- ', step_count
            random.shuffle(agents)
            step_count += 1
            shape_reward = False
            for agent in agents:
                if not done_flag:
                    obs_quads = range(Observe.TotalOptions)
                    obs_quad = Observe.Quadrant1
                    while obs_quads and obs_quad < Observe.NUM_QUADRANTS:
                        state = sa.env.get_agent_state(agent).reshape(1, 2 * WORLD_H * WORLD_W)
                        qval_obs = sa.obs_model.predict(state, batch_size=1)

                        if(random.random() < im.epsilon):
                            # obs_quad = random.randint(Observe.Quadrant1, Observe.TotalOptions)
                            random.shuffle(obs_quads)
                            obs_quad = obs_quads.pop()
                        else:
                            left_quads = qval_obs[ :, np.array(obs_quads) ]
                            obs_quad_indx = np.argmax( left_quads )
                            obs_quad = obs_quads[obs_quad_indx]
                            if(obs_quad in obs_quads):
                                obs_quads.remove(obs_quad)
                            else:
                                print obs_quad, ' ---- ', obs_quads
                                raise NotImplementedError

                        if(obs_quad < Observe.NUM_QUADRANTS):
                            sa.env.observe_quadrant(agent, obs_quad)
                            new_state = sa.env.get_agent_state(agent).reshape(1, 2 * WORLD_H * WORLD_W)

                    state = sa.env.get_agent_state(agent).reshape(1, 2 * WORLD_H * WORLD_W)
                    qval_act = sa.act_model.predict(state, batch_size=1)

                    if(random.random() < sa.epsilon):
                        action = np.random.randint(Actions.RIGHT, Actions.WAIT)
                    else:
                        action = (np.argmax(qval_act))

                    act_reward = sa.env.agent_action(agent, action)
                    sa.env.share_beliefs(agent)

                    closeness_reward = sa.env.check_formation(agent) * RWD_CLOSENESS

                    shape_reward = shape_reward or sa.env.check_shape()

                    # print ('Agent #%s \tact:%s actQ:%s \n\t\tobs:%s obsQ:%s \n\t\tactR:%s, shapeR:%s' % (agent, action, qval_act, obs_quad, qval_obs, act_reward, shape_reward))
                    # print ('Agent #%s actR:%s, shapeR:%s' % (agent, closeness_reward))

            if(shape_reward == True):
                print('Count: %s -- Shape formed in %s steps!' %( i, step_count ) )
                done_flag = True
