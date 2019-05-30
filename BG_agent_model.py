import nengo
import numpy as np

import gym

model=nengo.Network()

env = gym.make('CartPole-v0').env

class EnvironmentInterface(object):
    def __init__(self,env,stepSize =7):
        self.env = env
        self.n_actions = env.action_space.n
        self.state_dim = env.observation_space.shape[0]
        self.t=0
        self.stepsize = stepSize
        self.output = np.zeros(self.n_actions)
        self.state = env.reset()
        self.reward= 0
        self.current_action = 0

    def take_action(self,action):
        self.state,self.reward,self.done,_=env.step(action)
        if self.done:
            self.reward = -2
            self.state = env.reset()

    def get_reward(self,t):
        return self.reward
    
    def sensor(self,t):
        return self.state

    
    def step(self,t,x):
        if int(t*1000)%self.stepsize == 0:
            self.current_action = np.argmax(x) #np.argmax(self.output)#
            self.take_action(self.current_action)
    
    def calculate_Q(self,t,x):

        if int(t*1000) % self.stepsize == 1:
            qmax = x[np.argmax(x)]
            self.output = x
            self.output[self.current_action] = 0.9*qmax + self.reward
            
        return self.output
    
        
            
        
tau = 0.01

fast_tau = 0
slow_tau = 0.01
n_action =2
envI=EnvironmentInterface(env)

state_dimensions=envI.state_dim
n_actions = envI.n_actions


from gym import wrappers
from datetime import datetime
#Video Capturing Mechanism 
filename="test"
is_monitor=False
# env.close()
if is_monitor:
    #filename = os.path.basename(__file__).split('.')[0]
    monitor_dir = './' + filename + '_' + str(datetime.now())
    env = wrappers.Monitor(env, monitor_dir)
    env.reset()


D=2

with model:
    sensor = nengo.Node(envI.sensor)
    reward = nengo.Node(envI.get_reward)
    
    sensor_net = nengo.Ensemble(n_neurons=1000,dimensions=envI.state_dim,radius=2)
    
    nengo.Connection(sensor,sensor_net)
    
    # action_net = nengo.Ensemble(n_neurons=1000,dimensions=envI.n_actions,radius=10)
    
    # learning_conn=nengo.Connection(sensor_net,action_net,function=lambda x:[0,0],learning_rule_type=nengo.PES(1e-3, pre_tau=slow_tau),synapse=tau)
   
    q_node = nengo.Node(envI.calculate_Q,size_in=2,size_out=2)
    
    step_node = nengo.Node(envI.step,size_in=2)
    
    bg = nengo.networks.actionselection.BasalGanglia(D)
    
    learning_conn=nengo.Connection(sensor_net,bg.input,function=lambda x:[0,0],learning_rule_type=nengo.PES(1e-3, pre_tau=slow_tau),synapse=tau)
    
    # nengo.Connection(action_net,bg.input,synapse=fast_tau)

    thal = nengo.networks.actionselection.Thalamus(D)

    nengo.Connection(bg.output, thal.input,synapse=fast_tau)


    nengo.Connection(thal.output,step_node,synapse=fast_tau)
    
    nengo.Connection(thal.output,q_node,synapse=fast_tau)
    
    nengo.Connection(q_node,learning_conn.learning_rule,transform =-1,synapse=fast_tau) ##0.9*Q(s',a')+r
    
    nengo.Connection(thal.output,learning_conn.learning_rule,transform =1,synapse=slow_tau)#Q(s,a)



# env

# with model:
#     sensor  = nengo.Node(envI.sensor)

#     reward = nengo.Node(envI.get_reward)

#     update_node = nengo.Node(envI.step2,size_in=n_action,size_out=6)

#     q_Node = nengo.Node(size_in=n_action)


#     state = nengo.Ensemble(n_neurons=1000,dimensions=4,
#             intercepts=nengo.dists.Choice([0.15]), radius=4)

#     learn_signal = nengo.Ensemble(n_neurons=1000, dimensions=n_action)
    
#     action_value = nengo.Ensemble(n_neurons = 1000,dimensions=n_action)

#     nengo.Connection(sensor,state,synapse=None)

#     q_conn =nengo.Connection(state,update_node,function=lambda x:[0,0],
#     # transform=weight_init(shape=(n_action, 1000)),
#     learning_rule_type=nengo.PES(1e-3, pre_tau=slow_tau),
#                               synapse=fast_tau)
            
#     nengo.Connection(update_node[0:n_action],learn_signal,transform =-1,
#                     synapse=slow_tau)
        
#     nengo.Connection(update_node[n_action:2*n_action],learn_signal,transform=1,
#                     synapse = fast_tau)
                
#     nengo.Connection(learn_signal,q_conn.learning_rule,transform=-1,
#                     synapse=fast_tau)

    
#     nengo.Connection(update_node[2*n_action:], q_Node, synapse=fast_tau)





# with nengo.Simulator(model) as sim:
#     sim.run(10.0)



# sim=nengo.Simulator(model)

# for i in range(200000):
#     sim.step()


