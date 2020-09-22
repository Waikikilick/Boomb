# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 09:30:43 2019
布洛赫球上的归零操作
@author: Waikikilick
"""
#说明：本程序中，量子态的行向量形式用 psi 来代表，如：psi = np.matrix([[1+2j],[3+4j]])
#                      数组形式用 state 来代表，如 state = np.array([1,3,2,4])


import numpy as np
from scipy.linalg import expm

class Env( object):
    def __init__(self, 
        action_space=[0,1], #允许的动作，默认两个分立值，只是默认值，真正值由调用时输入
        dt=0.1, theta_num=5, varpsi_num=10,
        test_theta_mul=1, test_varpsi_mul=1, noise_a=0):
        super(Env, self).__init__() #应该就是为了秀一把
        self.action_space = action_space
        self.n_actions = len(self.action_space)
        self.n_features = 4 #描述状态所用的长度
        # self.target =  np.mat([[1], [1]], dtype=complex)/np.sqrt(2) #最终的目标态
        self.target =  np.mat([[1], [0]], dtype=complex) #最终的目标态为 |0>
        # self.state = np.array([1,0,0,0])
        self.nstep= 0 
        self.dt=dt
        
        self.noise_a = noise_a
       
        
        self.noise_normal = np.array([ 1.62434536, -0.61175641, -0.52817175, -1.07296862,  0.86540763,
       -2.3015387 ,  1.74481176, -0.7612069 ,  0.3190391 , -0.24937038,
        1.46210794, -2.06014071, -0.3224172 , -0.38405435,  1.13376944,
       -1.09989127, -0.17242821, -0.87785842,  0.04221375,  0.58281521,
       -1.10061918,  1.14472371,  0.90159072,  0.50249434,  0.90085595,
       -0.68372786, -0.12289023, -0.93576943, -0.26788808,  0.53035547,
       -0.69166075, -0.39675353, -0.6871727 , -0.84520564, -0.67124613,
       -0.0126646 , -1.11731035,  0.2344157 ,  1.65980218,  0.74204416,
       -0.19183555])
        #noise_normal 为均值为 0 ，标准差为 1 的正态分布的随机数组成的数组
        #该随机数由 np.random.seed(1) 生成: np.random.seed(1) \ noise_uniform = np.random.normal(loc=0.0, scale=1.0, size=41)
        
        self.noise = self.noise_a * self.noise_normal #uniform
        
        #在布洛赫球上选择训练点，这些训练点由下面的参数决定
        self.theta = np.linspace(np.pi,0,theta_num,endpoint=True) 
        self.varpsi = np.linspace(0,np.pi*2,varpsi_num,endpoint=False)
        
        #在布洛赫球上选择测试点，这些测试点由下面的参数决定
        test_theta = np.linspace(np.pi,0,theta_num+test_theta_mul*(theta_num-1),endpoint=True)
        test_theta = list(set(test_theta).difference(set(self.theta)))
        self.test_theta = np.array(test_theta)
        self.test_theta.sort()   #让选择的 test_theta 从小到大排列（有序）
        
        test_varpsi = np.linspace(0,np.pi*2,varpsi_num*(1+test_varpsi_mul),endpoint=False)
        test_varpsi = list(set(test_varpsi).difference(set(self.varpsi)))
        self.test_varpsi = np.array(test_varpsi)
        self.test_varpsi.sort()  #让选择的 test_varpsi 从小到大排列（有序）
        
        
    def trrset(self,k,kk):
        #测试过程中，归位到位于布洛赫球上的某一测试点上
        #k,kk 用来索引这次 trrset 使用第几个点.k 是 test_theta; kk 是 test_varpsi
        
        self.init_psi = np.mat([[np.cos(self.test_theta[k]/2)],[np.sin(self.test_theta[k]/2)*(np.cos(self.test_varpsi[kk])+np.sin(self.test_varpsi[kk])*(0+1j))]])
        #根据 theta 和 varpsi 将 init_state 写成列向量的形式
        self.init_state = np.array([self.init_psi[0,0].real, self.init_psi[1,0].real, self.init_psi[0,0].imag, self.init_psi[1,0].imag])
        # np.array([1实，2实，1虚，2虚])
        
        return self.init_state
    
        
        
    def rrset(self,k,kk):
        #归位到布洛赫球上的要训练的点 
        #k,kk 用来索引这次 rrset 使用第几个随机点.k 是 theta; kk 是 varpsi
        
        self.init_psi = np.mat([[np.cos(self.theta[k]/2)],[np.sin(self.theta[k]/2)*(np.cos(self.varpsi[kk])+np.sin(self.varpsi[kk])*(0+1j))]])
        #根据 theta 和 varpsi 将 init_state 写成列向量的形式
        self.init_state = np.array([self.init_psi[0,0].real, self.init_psi[1,0].real, self.init_psi[0,0].imag, self.init_psi[1,0].imag])
        # np.array([1实，2实，1虚，2虚])
        
        return self.init_state
    
    
    def reset(self): 
        #训练和测试中
        #在一个新的回合开始时，归位到开始选中的那个点上
        self.state = self.init_state
        self.nstep = 0 

        return self.state

    def step(self, action):


        psi = np.array([self.state[0:int(len(self.state) / 2)] + self.state[int(len(self.state) / 2):int(len(self.state))] * 1j])
        #array([[1实 + 1虚j, 2实 + 2虚j]])
        
        psi = psi.T
        #array([[1实 + 1虚j],
        #        2实 + 2虚j]])

        psi=np.mat(psi) 
        #matrix([[ 1实 + 1虚j],
       #         [ 2实 + 2虚j]])
        
        
        J = 1  # control field strength
        sx = np.mat([[0, 1], [1, 0]], dtype=complex)
        sz = np.mat([[1, 0], [0, -1]], dtype=complex)

        U = np.matrix(np.identity(2, dtype=complex)) 
        
        H =  J *float(action)* sz + 1 * sx
        U = expm(-1j * H * self.dt) 


        psi = U * psi  # next state

        err = 10e-3
        fid = (np.abs(psi.H * self.target) ** 2).item(0).real  
        rwd = 0.1*((10*fid)**3)*(fid<(1-err)) + 5000*(fid>=(1-err))
        #奖励值设成与 10*保真度 成立方关系，从而保证保真度越大时，所获奖励越大


        done =( ((1-fid) < err) or self.nstep>= np.pi/self.dt ) 
        self.nstep +=1  

        #再将量子态的 psi 形式恢复到 state 形式。
        #（因为复数没有梯度可言，神经网络在训练时，需要用实数来进行反向传递）
        psi=np.array(psi)
        psi_T = psi.T
        self.state = np.array(psi_T.real.tolist()[0] + psi_T.imag.tolist()[0])

        return self.state, rwd, done, fid
    
    def step_noise_J(self, action):
        #用于 J 噪声环境中的 step 操作：在选定的动作上增加噪声，进行时间演化


        psi = np.array([self.state[0:int(len(self.state) / 2)] + self.state[int(len(self.state) / 2):int(len(self.state))] * 1j])
        psi = psi.T
        psi=np.mat(psi) 

        J = 1  # control field strength
        sx = np.mat([[0, 1], [1, 0]], dtype=complex)
        sz = np.mat([[1, 0], [0, -1]], dtype=complex)

        U = np.matrix(np.identity(2, dtype=complex)) 
        
        H =  J *(float(action)+self.noise[self.nstep])* sz + 1 * sx

        U = expm(-1j * H * self.dt) 


        psi = U * psi  # next state

        
        err = 10e-3
        fid = (np.abs(psi.H * self.target) ** 2).item(0).real  
        rwd = 0.1*((10*fid)**3)*(fid<(1-err)) + 5000*(fid>=(1-err))

        done =( ((1-fid) < err) or self.nstep>= np.pi/self.dt ) 
        self.nstep +=1  

        psi=np.array(psi)
        psi_T = psi.T
        self.state = np.array(psi_T.real.tolist()[0] + psi_T.imag.tolist()[0])

        return self.state, rwd, done, fid
    
    def step_noise_h(self, action):
        #用于 h 噪声环境中的 step 操作：在选定的动作上增加噪声，进行时间演化

        psi = np.array([self.state[0:int(len(self.state) / 2)] + self.state[int(len(self.state) / 2):int(len(self.state))] * 1j])
        psi = psi.T
        psi=np.mat(psi) 
        
        J = 1  # control field strength
        sx = np.mat([[0, 1], [1, 0]], dtype=complex)
        sz = np.mat([[1, 0], [0, -1]], dtype=complex)

        U = np.matrix(np.identity(2, dtype=complex)) 
        
        H =  J *float(action)* sz + (1+self.noise[self.nstep]) * sx
        U = expm(-1j * H * self.dt) 


        psi = U * psi  # next state

        
        err = 10e-3
        fid = (np.abs(psi.H * self.target) ** 2).item(0).real  
        rwd = 0.1*((10*fid)**3)*(fid<(1-err)) + 5000*(fid>=(1-err))
        done =( ((1-fid) < err) or self.nstep>= np.pi/self.dt ) 
        self.nstep +=1  

        psi=np.array(psi)
        psi_T = psi.T
        self.state = np.array(psi_T.real.tolist()[0] + psi_T.imag.tolist()[0])

        return self.state, rwd, done, fid
    
    def step_noise_h_s(self, action):
        #用于 h 静态噪声环境中的 step 操作：在选定的动作上增加噪声，进行时间演化

        psi = np.array([self.state[0:int(len(self.state) / 2)] + self.state[int(len(self.state) / 2):int(len(self.state))] * 1j])
        psi = psi.T
        psi=np.mat(psi) 
        
        J = 1  # control field strength
        sx = np.mat([[0, 1], [1, 0]], dtype=complex)
        sz = np.mat([[1, 0], [0, -1]], dtype=complex)

        U = np.matrix(np.identity(2, dtype=complex)) 
        
        H =  J *float(action)* sz + (1+self.noise_a) * sx
        U = expm(-1j * H * self.dt) 


        psi = U * psi  # next state

        
        err = 10e-3
        fid = (np.abs(psi.H * self.target) ** 2).item(0).real  
        rwd = 0.1*((10*fid)**3)*(fid<(1-err)) + 5000*(fid>=(1-err))
        done =( ((1-fid) < err) or self.nstep>= np.pi/self.dt ) 
        self.nstep +=1  

        psi=np.array(psi)
        psi_T = psi.T
        self.state = np.array(psi_T.real.tolist()[0] + psi_T.imag.tolist()[0])

        return self.state, rwd, done, fid
    
    def step_noise_J_s(self, action):
        #用于 J 噪声环境中的 step 操作：在选定的动作上增加噪声，进行时间演化


        psi = np.array([self.state[0:int(len(self.state) / 2)] + self.state[int(len(self.state) / 2):int(len(self.state))] * 1j])
        psi = psi.T
        psi=np.mat(psi) 

        J = 1  # control field strength
        sx = np.mat([[0, 1], [1, 0]], dtype=complex)
        sz = np.mat([[1, 0], [0, -1]], dtype=complex)

        U = np.matrix(np.identity(2, dtype=complex)) 
        
        H =  J *(float(action)+self.noise_a)* sz + 1 * sx

        U = expm(-1j * H * self.dt) 


        psi = U * psi  # next state

        
        err = 10e-3
        fid = (np.abs(psi.H * self.target) ** 2).item(0).real  
        rwd = 0.1*((10*fid)**3)*(fid<(1-err)) + 5000*(fid>=(1-err))

        done =( ((1-fid) < err) or self.nstep>= np.pi/self.dt ) 
        self.nstep +=1  

        psi=np.array(psi)
        psi_T = psi.T
        self.state = np.array(psi_T.real.tolist()[0] + psi_T.imag.tolist()[0])

        return self.state, rwd, done, fid