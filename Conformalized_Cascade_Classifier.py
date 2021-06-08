#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import random
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Input, Dense, Dropout, Lambda, Reshape, Activation
from keras.optimizers import Adam
from keras import layers
from keras import Input
from keras.models import Model
from keras.utils.np_utils import to_categorical,normalize

from sklearn.utils import shuffle
from keras import backend as K
from functools import partial
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
import math

EPISODES = 2500
Ep_pretrain=500

class DQNAgent:
    def __init__(self, n_features, n_classes):
        self.n_features = n_features
        self.n_classes = n_classes
        self.n_action = n_features+n_classes
        self.memory = deque(maxlen=2000)
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.hidden_units1 = 6
        self.hidden_units2 = 6
        self.model1 = self._build_rejectselect()
        self.model2 = self._build_classifier()

    def _build_rejectselect(self):       
        input_tensor = Input(shape=(2*self.n_features,))
        hidden = layers.Dense(self.hidden_units1, activation='relu')(input_tensor)
        hidden = layers.Dense(self.hidden_units1, activation='relu')(hidden)
        output_features = layers.Dense(self.n_features, activation='softplus',name='features')(hidden)
        model1 = Model(input_tensor,output_features)
        model1.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model1
    
    def _build_classifier(self):
        input_mask  = Input(shape=(n_features,))
        hidden      = layers.Dense(self.hidden_units2, activation='relu')(input_mask)
        coeffvector = layers.Dense(n_features*n_classes, activation='linear')(hidden)
        coeffvector = Reshape((n_features,n_classes))(coeffvector)
    
        input_values = Input(shape=(n_features,))
        mul_output = Lambda(lambda xy: K.batch_dot(xy[0],xy[1],axes=[1, 1]))([coeffvector,input_values])
        output = Activation('softmax')(mul_output)
    
        model2 = Model(inputs=[input_mask,input_values], outputs=output)
        model2.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
        return model2

    def remember(self, state, mask, action, cost, next_state, next_mask, done):
        self.memory.append((state, mask, action, cost, next_state, next_mask, done))

    def act(self, state, mask):
        action_available = np.concatenate((np.ones(n_features)-mask,np.ones(n_classes)))
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.n_action,p=action_available/np.sum(action_available))
        maskI = np.reshape(mask, [1, self.n_features])
        stateI = np.concatenate((np.reshape(state[0],(self.n_features,1)), np.reshape(mask,(self.n_features,1))))
        stateI = np.reshape(stateI, [1, 2*self.n_features])
        act1 = self.model1.predict(stateI)[0]*action_available[:self.n_features]
        act2 = self.model2.predict([maskI,state])[0]
        act_values = np.concatenate((np.reshape(act1,(self.n_features,1)), np.reshape(act2,(self.n_classes,1))))
        act_values[act_values==0]=100
        return np.argmin(act_values)  # returns action

    def pretrain(self,Xb,Mb,Yb):
        self.model2.fit([Xb,Mb],Yb,epochs=40)

    def replay(self, batch_size,e):
        minibatch = random.sample(self.memory, batch_size)
        '''minibatch'''
        BstateI = np.zeros((batch_size,2*self.n_features))
        Btarget_f = np.zeros((batch_size,self.n_features))
        Btarget_c = np.zeros((batch_size,self.n_classes))
        BmaskI = np.zeros((batch_size,self.n_features))
        Bstate = np.zeros((batch_size,self.n_features))
        compt=0
        for state, mask, action, cost, next_state, next_mask, done in minibatch:
            target = cost
            if not done:
                next_action_available = np.ones(n_features)-next_mask
                next_maskI = np.reshape(next_mask, [1, self.n_features])
                next_stateI = np.concatenate((np.reshape(next_state[0],(self.n_features,1)), np.reshape(next_mask,(self.n_features,1))))
                next_stateI = np.reshape(next_stateI, [1, 2*self.n_features])
                act1 = self.model1.predict(next_stateI)[0]*next_action_available
                act2 = self.model2.predict([next_maskI,next_state])[0]
                act_values = np.concatenate((np.reshape(act1,(self.n_features,1)),np.reshape(act2,(self.n_classes,1))))
                act_values[act_values==0]=100
                target = (cost + np.amin(act_values))
                
            maskI = np.reshape(mask, [1, self.n_features])
            stateI = np.concatenate((np.reshape(state[0],(self.n_features,1)), np.reshape(mask,(self.n_features,1))))
            stateI = np.reshape(stateI, [1, 2*self.n_features])
            target_f = self.model1.predict(stateI)
            target_c= self.model2.predict([maskI,state])
            if action<self.n_features:
                target_f[0][action] = target
            else:
                target_c[0][action-self.n_features] = target
            BstateI[compt] = stateI[0]
            Btarget_f[compt] = target_f[0]
            Btarget_c[compt] = target_c[0]
            BmaskI[compt] = maskI[0]
            Bstate[compt] = state[0]
            compt+=1
        self.model1.fit(BstateI, Btarget_f, epochs=1, verbose=0)
        self.model2.fit([BmaskI,Bstate],  Btarget_c, epochs=1, verbose=0)        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def actpretrain(self, state, mask):
        action_available = np.concatenate((np.ones(n_features)-mask,np.ones(n_classes)))
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.n_action,p=action_available/np.sum(action_available))
        maskI = np.reshape(mask, [1, self.n_features])
        stateI = np.concatenate((np.reshape(np.zeros(self.n_features),(self.n_features,1)), np.reshape(mask,(self.n_features,1))))
        stateI = np.reshape(stateI, [1, 2*self.n_features])
        act1 = self.model1.predict(stateI)[0]*action_available[:self.n_features]
        act2 = self.model2.predict([maskI,state])[0]
        act_values = np.concatenate((np.reshape(act1,(self.n_features,1)), np.reshape(act2,(self.n_classes,1))))
        act_values[act_values==0]=100
        return np.argmin(act_values)  # returns action

    def replaypretrain(self, batch_size,e):
        minibatch = random.sample(self.memory, batch_size)
        '''minibatch'''
        BstateI = np.zeros((batch_size,2*self.n_features))
        Btarget_f = np.zeros((batch_size,self.n_features))
        Btarget_c = np.zeros((batch_size,self.n_classes))
        BmaskI = np.zeros((batch_size,self.n_features))
        Bstate = np.zeros((batch_size,self.n_features))
        compt=0
        for state, mask, action, cost, next_state, next_mask, done in minibatch:
            target = cost
            if not done:
                next_action_available = np.ones(n_features)-next_mask
                next_maskI = np.reshape(next_mask, [1, self.n_features])
                next_stateI = np.concatenate((np.reshape(np.zeros(self.n_features),(self.n_features,1)), np.reshape(next_mask,(self.n_features,1))))
                next_stateI = np.reshape(next_stateI, [1, 2*self.n_features])
                act1 = self.model1.predict(next_stateI)[0]*next_action_available
                act2 = self.model2.predict([next_maskI,next_state])[0]
                act_values = np.concatenate((np.reshape(act1,(self.n_features,1)),np.reshape(act2,(self.n_classes,1))))
                act_values[act_values==0]=100
                target = (cost + np.amin(act_values))    
            maskI = np.reshape(mask, [1, self.n_features])
            stateI = np.concatenate((np.reshape(np.zeros(self.n_features),(self.n_features,1)), np.reshape(mask,(self.n_features,1))))
            stateI = np.reshape(stateI, [1, 2*self.n_features])
            target_f = self.model1.predict(stateI)
            target_c= self.model2.predict([maskI,state])
            if action<self.n_features:
                target_f[0][action] = target
            else:
                target_c[0][action-self.n_features] = target
            BstateI[compt] = stateI[0]
            Btarget_f[compt] = target_f[0]
            Btarget_c[compt] = target_c[0]
            BmaskI[compt] = maskI[0]
            Bstate[compt] = state[0]
            compt+=1
        self.model1.fit(BstateI, Btarget_f, epochs=1, verbose=0)
        self.model2.fit([BmaskI,Bstate],  Btarget_c, epochs=1, verbose=0)        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


class ENVdata:
    def __init__(self, Vcost, n_features, n_classes):
        self.Vcost = Vcost
        self.n_features = n_features
        self.n_classes = n_classes
        self.n_action=n_features+n_classes
        
    def step(self, action, state, mask, x, y):
        next_mask = np.copy(mask)
        next_state = x*next_mask
        if action<n_features:
            next_mask[action] = 1
            next_state = x*next_mask
            cost = self.Vcost[action]
            done = 0
        else:
            done = 1
            cost =1
            if action-(n_features)==y:
                cost = 0
        return next_state, next_mask, cost, done

    def reset(self, Xmat, ymat):
        done = 0
        state = np.zeros(n_features)
        mask = np.zeros(n_features)
        ide = np.random.choice(len(Xmat[:,0]))
        x = Xmat[ide,:]
        y = ymat[ide]
        return state, mask, x, y, done
    
    def reset2(self, ide, Xmat, ymat):
        done = 0
        state = np.zeros(n_features)
        mask = np.zeros(n_features)
        x = Xmat[ide,:]
        y = ymat[ide]
        return state, mask, x, y, done


#if __name__ == "__main__":
def Train_dqn(X_train, X_test, y_train, y_test, X_valid, y_valid, conform, alpha, n_classes):        
    n_features=len(X_train[0,:])
    agent = DQNAgent( n_features, n_classes)
    '''Pretraining the classifier'''
    print("Pretraining the classifier...")
    NN = 40
    Y = to_categorical(y_train)
    Yb = Y
    X = X_train
    Xb = X
    M = np.ones((len(X_train[:,0]),n_features))
    Mb =np.ones((len(X_train[:,0])*NN,n_features))
    for k in np.arange(NN-1):
        Xb  = np.vstack((Xb,X))
        Yb  = np.vstack((Yb,Y))
    for k in np.arange(len(X_train[:,0])*NN):
        for i in  np.arange(n_features):
            Mb[k,i] = np.random.choice(np.arange(2))
    Xb = np.multiply(Xb,Mb)
    Yb = np.ones((len(X_train[:,0])*NN,n_classes))-Yb
    agent.pretrain(Xb,Mb,Yb)
    
    '''Pretraining the rejecter'''
    print("Pretraining the rejector...")
    env = ENVdata( Vcost, n_features, n_classes)
    # agent.load("./save/cartpole-dqn.h5")
    batch_size = 32
    idi = 0
    for e in range(Ep_pretrain):
        if idi<len(X_train[:,0]) :
            state, mask, x, y, done = env.reset2( idi, X_train, y_train)
            idi+=1
        else :
            state, mask, x, y, done = env.reset( X_train, y_train)
        state = np.reshape(state, [1, n_features])
        while not done:
            # env.render()
            action = agent.actpretrain(state, mask)
            next_state, next_mask, cost, done = env.step(action, state, mask, x, y)
            cost = cost
            next_state = np.reshape(next_state, [1,n_features])
            agent.remember(state, mask, action, cost, next_state, next_mask, done)
            state = np.copy(next_state)
            mask = np.copy(next_mask)
            if done:
                print("episode: {}/{}, e: {:.2}"
                      .format(e, Ep_pretrain, agent.epsilon))
                break    
        if len(agent.memory) > batch_size:
            agent.replaypretrain(batch_size,e)
    
    '''Fitting the model'''
    
    agent.epsilon = 0.2
    env = ENVdata( Vcost, n_features, n_classes)
    batch_size = 32
    idi = 0
    for e in range(EPISODES):
        if idi<len(X_train[:,0]) :
            state, mask, x, y, done = env.reset2( idi, X_train, y_train)
            idi+=1
        else :
            state, mask, x, y, done = env.reset( X_train, y_train)
        state = np.reshape(state, [1, n_features])
        while not done:
            # env.render()
            action = agent.act(state, mask)
            next_state, next_mask, cost, done = env.step(action, state, mask, x, y)
            cost = cost
            next_state = np.reshape(next_state, [1,n_features])
            agent.remember(state, mask, action, cost, next_state, next_mask, done)
            state = np.copy(next_state)
            mask = np.copy(next_mask)
            if done:
                print("episode: {}/{}, e: {:.2}"
                      .format(e, EPISODES, agent.epsilon))
                break    
        if len(agent.memory) > batch_size:
            agent.replay(batch_size,e)
    
    return agent
    
    
    
def Test_dqn(agent, X_train, X_test, y_train, y_test, X_valid, y_valid, conform, alpha, n_classes):          
    env = ENVdata( Vcost, n_features, n_classes)

    if conform == "bloc_conform" :
        ''' Computing the E scores on the validation '''
    
        th = math.ceil((1 - alpha)*(1 + len(X_valid[:,0])))
        if th >= len(X_valid[:,0]):
                     th = len(X_valid[:,0]) - 1
    
        E_scores = np.zeros(len(X_valid[:,0]))
        for k in range(len(X_valid[:,0])):
            state, mask, x, y, done = env.reset2(k,X_valid, y_valid)
            cost_cum=0
            e = 1 
            tmp_sum = 0
            
            while not done:
                
                if (e == 1):
                    state = np.reshape(state, [1, n_features])
                    action_available = np.concatenate((np.ones(n_features)-mask,np.ones(n_classes)))
                    maskI = np.reshape(mask, [1, n_features])
                    stateI = np.concatenate((np.reshape(state[0],(n_features,1)), np.reshape(mask,(n_features,1))))
                    stateI = np.reshape(stateI, [1, 2*n_features])
                    act1 = agent.model1.predict(stateI)[0]*action_available[:n_features]
                    act2 = agent.model2.predict([maskI,state])[0]
                    act_values = np.reshape(act1,(n_features,1))
                    act_values[act_values==0]=100
                    action = np.argmin(act_values)
                else:
                    state = np.reshape(state, [1, n_features])
                    action_available = np.concatenate((np.ones(n_features)-mask,np.ones(n_classes)))
                    maskI = np.reshape(mask, [1, n_features])
                    stateI = np.concatenate((np.reshape(state[0],(n_features,1)), np.reshape(mask,(n_features,1))))
                    stateI = np.reshape(stateI, [1, 2*n_features])
                    act1 = agent.model1.predict(stateI)[0]*action_available[:n_features]
                    act2 = agent.model2.predict([maskI,state])[0]
                    act_values = np.concatenate((np.reshape(act1,(n_features,1)), np.reshape(act2,(n_classes,1))))
                    act_values[act_values==0]=100
                    action = np.argmin(act_values) 
                
            
                next_state, next_mask, cost, done = env.step(action, state, mask, x, y)
                cost_cum += cost
                state = np.copy(next_state)
                mask = np.copy(next_mask)
                e += 1
                
            E_scores[k] = 1.0 - np.max([act_values[action], 1-act_values[action]])
        
        descending_E_scores = np.sort(E_scores)[::-1]
        tau = descending_E_scores[th]
        
    
    '''Testing the model'''
    NFolds = 10
    kf = KFold(n_splits=NFolds)
    Result = np.zeros((NFolds,2))
    iFold = 0
    for train_index, test_index in kf.split(X_test):

        X_train_tmp, X_test_tmp = X_test[train_index], X_test[test_index]
        y_train_tmp, y_test_tmp = y_test[train_index], y_test[test_index]

        Result_tmp = -np.ones((len(X_test_tmp[:,0]),n_classes+n_features+1))
        
        for k in range(len(X_test_tmp[:,0])):
            state, mask, x, y, done = env.reset2(k,X_test_tmp, y_test_tmp)
            cost_cum=0
            e = 1 
            tmp_sum = 0
        
            while not done:
             
                if (e == 1):
                    state = np.reshape(state, [1, n_features])
                    action_available = np.concatenate((np.ones(n_features)-mask,np.ones(n_classes)))
                    maskI = np.reshape(mask, [1, n_features])
                    stateI = np.concatenate((np.reshape(state[0],(n_features,1)), np.reshape(mask,(n_features,1))))
                    stateI = np.reshape(stateI, [1, 2*n_features])
                    act1 = agent.model1.predict(stateI)[0]*action_available[:n_features]
                    act2 = agent.model2.predict([maskI,state])[0]
                    act_values = np.reshape(act1,(n_features,1))
                    act_values[act_values==0]=100
                    action = np.argmin(act_values)
                else:
                    state = np.reshape(state, [1, n_features])
                    action_available = np.concatenate((np.ones(n_features)-mask,np.ones(n_classes)))
                    maskI = np.reshape(mask, [1, n_features])
                    stateI = np.concatenate((np.reshape(state[0],(n_features,1)), np.reshape(mask,(n_features,1))))
                    stateI = np.reshape(stateI, [1, 2*n_features])
                    act1 = agent.model1.predict(stateI)[0]*action_available[:n_features]
                    act2 = agent.model2.predict([maskI,state])[0]
                    act_values = np.concatenate((np.reshape(act1,(n_features,1)), np.reshape(act2,(n_classes,1))))
                    act_values[act_values==0]=100
                    action = np.argmin(act_values) 
                  
                if conform == 'bloc_conform' and action > n_features  and (tau <= 1 - tmp_sum) and e < n_features+1: 
                    act_values = np.reshape(act1,(n_features,1))
                    act_values[act_values==0]=100
                    action = np.argmin(act_values) 

                    
                next_state, next_mask, cost, done = env.step(action, state, mask, x, y)
                
                Result_tmp[k,e] =action 
                tmp_sum += act_values[action]
 
                cost_cum += cost
                state = np.copy(next_state)
                mask = np.copy(next_mask)
                e += 1
                if conform == 'bloc_conform' :
                    Result_tmp[k,0] = tau
                else :
                    Result_tmp[k,0] = cost_cum
                Result_tmp[k, n_classes+n_features] = action-n_features
                
        mean_nb_features = 0
        for i in range(Result_tmp.shape[0]) : 
            tmp = np.copy(Result_tmp[i, 1:-1])
            for j in tmp :
                if j != -1 and j != n_features and j!= n_features + 1:
                    mean_nb_features += 1
        mean_nb_features /= len(y_test_tmp)

        accuracy = sum(Result_tmp[:, n_classes+n_features] == y_test_tmp)/len(y_test_tmp)
        Result[iFold,0] = accuracy
        Result[iFold,1] = mean_nb_features
        iFold += 1
       
    return Result
 

#######################################################
#######################################################

from sklearn.preprocessing import StandardScaler
from rulefit import RuleFit
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.model_selection import train_test_split

alpha = 0.1

real_data = np.loadtxt('Glaucoma.txt')
X = real_data[:,0:-1]
y = real_data[:,-1]
y[y == -1] = 0
n_classes = 2

NMC = 10
results = np.zeros((NMC, 5)) 
mean_nb_features = np.zeros((NMC, 5))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.1) 
n_features = X_train.shape[1]
cost_compt=16
Vcost=1e-3*np.ones(n_features)/(n_features*4)

# Define the scaler 
scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
X_valid = scaler.transform(X_valid)

# Train cascade classifier with a 
trained_agent = Train_dqn(X_train, X_test, y_train, y_test, X_valid,y_valid, 'standard',alpha,n_classes)
Result = Test_dqn(trained_agent,X_train, X_test, y_train, y_test, X_valid,y_valid, 'standard',alpha,n_classes)

results[:,0] = Result[:,0]
mean_nb_features[:,0] = Result[:,1]
    
# Conformalised classifier with abstention
Result_conform = Test_dqn(trained_agent, X_train, X_test, y_train, y_test, X_valid, y_valid, 'bloc_conform', alpha,n_classes)
results[:,1] = Result_conform[:,0]
mean_nb_features[:,1] = Result_conform[:,1] 

# RuleFit
gb = GradientBoostingClassifier(n_estimators=100,max_depth=4)

rf = RuleFit(tree_size=4,sample_fract='default',max_rules=2000,
                 memory_par=0.01,model_type='r',
                 tree_generator=gb,
                rfmode='classify',lin_trim_quantile=0.025,
                lin_standardise=True, exp_rand_tree_size=True,random_state=1)
rf.fit(X_train, y_train)
    
rules = rf.get_rules()
rules = rules[rules.coef != 0].sort_values("support", ascending=False)


if (math.ceil(np.mean(mean_nb_features[:,0])) // 2 > len(rules) ) :
    rules = rules[rules.importance >= np.sort(rules['importance'])[::-1][math.ceil(np.mean(mean_nb_features[:,0]))//2]]
elif (math.ceil(np.mean(mean_nb_features[:,0])) > len(rules) ) :
    rules = rules[rules.importance >= np.sort(rules['importance'])[::-1][math.ceil(np.mean(mean_nb_features[:,0]))]]
else :
    rules = rules[rules.importance >= np.sort(rules['importance'])[::-1][len(rules)//2]]

# Reduce the number of rules 

NFolds = 10
kf = KFold(n_splits=NFolds)
iFold = 0
for train_index, test_index in kf.split(X_test):
    X_train_tmp, X_test_tmp = X_test[train_index], X_test[test_index]
    y_train_tmp, y_test_tmp = y_test[train_index], y_test[test_index]

    y_test_predicted = rf.predict(X_test_tmp)
    results[iFold,2] = sum(y_test_predicted == y_test_tmp)/len(y_test_tmp)
    mean_nb_features[iFold, 2] = len(rules)
    iFold += 1


X_transformed_train = rf.transform(X_train)
X_transformed_test = rf.transform(X_test)
X_transformed_valid = rf.transform(X_valid)
X_transformed_train_reduced = X_transformed_train[:,rules.index]
X_transformed_test_reduced = X_transformed_test[:,rules.index]
X_transformed_valid_reduced = X_transformed_valid[:,rules.index]

n_features = X_transformed_train_reduced.shape[1]
Vcost=1e-3*np.ones(n_features)/(n_features*4)   

agent_rules = Train_dqn(X_transformed_train_reduced, X_transformed_test_reduced, y_train, y_test, X_transformed_valid_reduced, y_valid, 'standard', alpha,n_classes)  
Result_C_rules = Test_dqn(agent_rules, X_transformed_train_reduced, X_transformed_test_reduced, y_train, y_test, X_transformed_valid_reduced, y_valid, 'standard', alpha,n_classes)  
results[:, 3] = Result_C_rules[:, 0] 
mean_nb_features[:, 3] = Result_C_rules[:, 1]
    
Result_rules = Test_dqn(agent_rules,X_transformed_train_reduced, X_transformed_test_reduced, y_train, y_test, X_transformed_valid_reduced, y_valid, 'bloc_conform', alpha, n_classes)  
results[:, 4] = Result_rules[:, 0]
mean_nb_features[:, 4] = Result_rules[:, 1]


np.savetxt('accuracy.txt', results, fmt='%f')
np.savetxt('mean_nb_features.txt', mean_nb_features, fmt='%f')

#######################
import matplotlib.patches as mpatches

# Boxplot 

results = np.loadtxt('results.txt', dtype=float)
labs = np.round(np.mean(np.loadtxt('mean_nb_features.txt', dtype=float),axis=0),2)
mean_nb_features = np.loadtxt('mean_nb_features.txt', dtype=float)

box=plt.boxplot(results, positions=np.arange(results.shape[1])+1,patch_artist=True)
colors = ['blue', 'violet', 'green', 'tan', 'purple']
 
for patch, color in zip(box['boxes'], colors):
    patch.set_facecolor(color)
 
    
plt.xticks(np.arange(results.shape[1])+1, labs)
blue_patch = mpatches.Patch(color='blue', label='Cascade with abstention')
violet_patch = mpatches.Patch(color='violet', label='Conformal Cascade')
green_patch = mpatches.Patch(color='green', label='RuleFit')
tan_patch = mpatches.Patch(color='tan', label='RuleFit Cascade')
purple_patch = mpatches.Patch(color='purple', label='Conformal RuleFit Cascade')

plt.legend(handles=[blue_patch, violet_patch, green_patch, tan_patch, purple_patch])
plt.xlabel("Mean nb of features")
plt.ylabel("Test accuracy")
plt.savefig('performance.png') 

