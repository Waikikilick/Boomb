# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 09:29:35 2019
用来实现布洛赫球面上的归零操作 归到 |0>态
此程序可以生成在特定的起点所施加的操作
生成在操作下量子态及其在布洛赫球上位置的变化
@author: Waikikilick
"""

from environment_noise import Env
from Net_dql import DeepQNetwork
import warnings
warnings.filterwarnings('ignore')
from time import *

import numpy as np
np.random.seed(1)


#在布洛赫球上选取一系列训练点
#在 env 文件中会看到这些点是随角度均匀分布的
theta_num = 5  #theta 选取点的数目
#将范围在 [0，pi] 之间的 theta 包含起止点均匀选取 theta_num 个点
varpsi_num = 10 #varpsi 选取点的数目
#将范围在 [0，2*pi) 之间的 varpsi 包含起点但不包含终点均匀选取 varpsi_num 个点
fidelity_list = np.zeros((theta_num, varpsi_num)) 
#记录每个选择测试点在经过操作之后可以达到的最终保真度


#在布洛赫球面上选取一系列测试点
#选取方法是在上面均匀分布的训练点的间隔中，等间距选择 _mul 个点
#训练集和测试集是不交叉的
test_theta_mul = 2
test_varpsi_mul = 4

test_theta_num = test_theta_mul*(theta_num-1) #测试集 test_theta 选取点的数目
test_varpsi_num = varpsi_num*test_varpsi_mul  #      test_varpsi 选取点的数目
test_fidelity_list = np.zeros((test_theta_num, test_varpsi_num))
# #记录每个选择测试点在经过操作之后可以达到的最终保真度


#--------------------------------------------------------------------------------------
#训练部分
def training():
    
    # print('theta:\n',env.theta,'\n\nvarpsi:\n',env.varpsi,'\n')
    #打印出选出来待训练的初态点
    print('\ntraining...')
    
    #根据之前选好的初始点，依次训练神经网络，得到对应的最大最终保真度，并将其记录在矩阵中
    for k in range(theta_num):#按之前生成的 theta 数，依次训练
        for kk in range(varpsi_num):#.....varpsi..........
            print(k,kk)
            env.rrset(k,kk) #在训练完一个选好的训练点后，将初量子态调到下一个训练点上
            global tanxin
            tanxin = 0 #动作选择的 epsilon:当训练时，选择一个训练点后，tanxin = 0 epsilon 设为 0，之后改变 tanxin = 0.5 施加递减的贪心策略；
                        #更换训练点后，tanxin = 0，epsilon 重新选为0，重新执行递减贪心策略
                        #测试时，tanxin = 1，epsilon = 1,不使用贪心策略 ，直接选择值最大的动作
            fid_10 = 0
            ep_max = 100
            for episode in range(ep_max):
                observation = env.reset()

                while True: #
                    
                    action = RL.choose_action(observation,tanxin)
                    observation_, reward, done, fid = env.step(action)
                    RL.store_transition(observation, action, reward, observation_)
                    RL.learn()
                    tanxin = 0.5

                    observation = observation_
                    if done:
                        if episode >= ep_max-11: 
                            #在最后 10 个回合中选择最大的最终保真度进行输出
                            fid_10 = max(fid_10,fid)
                        break  
                        
            fidelity_list[k,kk] = fid_10 #将最大的最终保真度记录到矩阵中
    return fidelity_list
#----------------------------------------------------------------------------------
    
#---------------------------------------------------------------------------------- 
#测试部分(无噪声)
    
def testing():
    #在布洛赫球上测试上面选好的测试点，对训练好的模型进行测试
    
    # print('test_theta:\n',env.test_theta,'\n\ntest_varpsi:\n',env.test_varpsi,'\n')
    #打印出选出来测试点
    print('\ntesting...\n')
    
    global tanxin
    tanxin = 1

    #根据之前选好待测试点，依次输入神经网络，得到对应的最大最终保真度，并将其记录在矩阵中
    for k in range(test_theta_num):#按之前生成的 test_theta 数，依次训练
        for kk in range(test_varpsi_num):#      test_varpsi
            env.trrset(k,kk) #在测试完一个训练点后，将初量子态调到下一个待测试点上
            observation = env.reset()
            test_fid = 0
            
            while True:
                action = RL.choose_action(observation,tanxin)
                observation_, reward, done, fid = env.step(action)
                observation = observation_
                test_fid = max(test_fid,fid) 
                #直接选择在操作过程中最佳保真度作为本回合的保真度
                #因为在计算过程中，达到最佳保真度这一点是可以保证做到的
                
                if done:
                    break
                        
            test_fidelity_list[k,kk] = test_fid #将最大的最终保真度记录到矩阵中
    return test_fidelity_list
#--------------------------------------------------------------------------------


#--------------------------------------------------------------------------------
#测试部分（ J 噪声环境）
    
def testing_noise_J():
    #在布洛赫球上测试上面选好的测试点，对训练好的模型进行测试
    
    # print('test_theta:\n',env.test_theta,'\n\ntest_varpsi:\n',env.test_varpsi,'\n')
    #打印出选出来测试点
    print('\ntesting...noise_J\n')
    
    global tanxin
    tanxin = 1

    #根据之前选好待测试点，依次输入神经网络，得到对应的噪声环境下的最终保真度，并将其记录在矩阵中
    for k in range(test_theta_num):#按之前生成的 test_theta 数，依次训练
        for kk in range(test_varpsi_num):#      test_varpsi
            
            env.trrset(k,kk) #在测试完一个训练点后，将初量子态调到下一个待测试点上
            observation = env.reset()
            action_list = [] #用来保存本回合所采取的动作，用于噪声分析
            fid_list = [] #用来保存本回合中的保真度，选择最大保真度对应的步骤作为后面噪声环境中动作的终止步骤
            while True:
                action = RL.choose_action(observation,tanxin)
                # action = 0 #可以测试没有噪声时，是否和预期一致，检验算法正误
                action_list = np.append(action_list,action)
                observation_, reward, done, fid = env.step(action)
                fid_list = np.append(fid_list,fid)
                observation = observation_
                #这个环节就是完全根据网络的预测来选择动作，并将动作和对应的保真度记录下来
                if done:
                    break
                
            #下面根据记录的保真度，挑出在哪一步保真度最高来进行截取动作（因为有可能达到最高保真度之后，智能体又多走
            #将噪声加到达到最佳保真度之前各步数上
            observation = env.reset()
            fid_list = list(map(float,fid_list))
            max_index = fid_list.index(max(fid_list))
            action_list = action_list[0:max_index+1]
            
            
            #加入噪声
            test_fid_noise = 0
            for action in action_list:
                
                observation_, reward, done, fid = env.step_noise_J(action)
                observation = observation_
                test_fid_noise = fid #选择最后一步的保真度作为本回合的保真度
                
            test_fidelity_list[k,kk] = test_fid_noise #将最终保真度记录到矩阵中
    return test_fidelity_list
#--------------------------------------------------------------------------------

#--------------------------------------------------------------------------------
#测试部分（ h 噪声环境）
    
def testing_noise_h():
    #在布洛赫球上测试上面选好的测试点，对训练好的模型进行测试
    
    # print('test_theta:\n',env.test_theta,'\n\ntest_varpsi:\n',env.test_varpsi,'\n')
    #打印出选出来测试点
    print('\ntesting noise_h...')
    
    global tanxin
    tanxin = 1

    #根据之前选好待测试点，依次输入神经网络，得到在环境噪声下的最终保真度，并将其记录在矩阵中
    for k in range(test_theta_num):#按之前生成的 test_theta 数，依次训练
        for kk in range(test_varpsi_num):#      test_varpsi
            
            env.trrset(k,kk) #在测试完一个训练点后，将初量子态调到下一个待测试点上
            observation = env.reset()
            action_list = [] #用来保存本回合所采取的动作，用于噪声分析
            fid_list = [] #用来保存本回合中的保真度，选择最大保真度对应的步骤作为后面噪声环境中动作的终止步骤
            while True:
                action = RL.choose_action(observation,tanxin)
                # action = 0
                action_list = np.append(action_list,action)
                observation_, reward, done, fid = env.step(action)
                fid_list = np.append(fid_list,fid)
                observation = observation_
                
                if done:
                    break
                
            observation = env.reset()
            fid_list = list(map(float,fid_list))
            max_index = fid_list.index(max(fid_list))
            action_list = action_list[0:max_index+1]
            
            
            test_fid_noise = 0
            for action in action_list:
                
                observation_, reward, done, fid = env.step_noise_h(action)
                observation = observation_
                test_fid_noise = fid 
                
            test_fidelity_list[k,kk] = test_fid_noise 
#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------
#测试部分（ h 静态噪声环境）
    
def testing_noise_h_s():
    #在布洛赫球上测试上面选好的测试点，对训练好的模型进行测试
    
    # print('test_theta:\n',env.test_theta,'\n\ntest_varpsi:\n',env.test_varpsi,'\n')
    #打印出选出来测试点
    print('\ntesting noise_h_s...')
    
    global tanxin
    tanxin = 1

    #根据之前选好待测试点，依次输入神经网络，得到在环境噪声下的最终保真度，并将其记录在矩阵中
    for k in range(test_theta_num):#按之前生成的 test_theta 数，依次训练
        for kk in range(test_varpsi_num):#      test_varpsi
            
            env.trrset(k,kk) #在测试完一个训练点后，将初量子态调到下一个待测试点上
            observation = env.reset()
            action_list = [] #用来保存本回合所采取的动作，用于噪声分析
            fid_list = [] #用来保存本回合中的保真度，选择最大保真度对应的步骤作为后面噪声环境中动作的终止步骤
            while True:
                action = RL.choose_action(observation,tanxin)
                
                action_list = np.append(action_list,action)
                observation_, reward, done, fid = env.step(action)
                fid_list = np.append(fid_list,fid)
                observation = observation_
                
                if done:
                    break
                
            observation = env.reset()
            fid_list = list(map(float,fid_list))
            max_index = fid_list.index(max(fid_list))
            action_list = action_list[0:max_index+1]
            
            
            test_fid_noise = 0
            for action in action_list:
                
                observation_, reward, done, fid = env.step_noise_h_s(action)
                observation = observation_
                test_fid_noise = fid 
                
            test_fidelity_list[k,kk] = test_fid_noise 
    return test_fidelity_list
#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------
#测试部分（ J 静态噪声环境）
    
def testing_noise_J_s():
    #在布洛赫球上测试上面选好的测试点，对训练好的模型进行测试
    
    # print('test_theta:\n',env.test_theta,'\n\ntest_varpsi:\n',env.test_varpsi,'\n')
    #打印出选出来测试点
    print('\ntesting noise_J_s...')
    
    global tanxin
    tanxin = 1

    #根据之前选好待测试点，依次输入神经网络，得到在环境噪声下的最终保真度，并将其记录在矩阵中
    for k in range(test_theta_num):#按之前生成的 test_theta 数，依次训练
        for kk in range(test_varpsi_num):#      test_varpsi
            
            env.trrset(k,kk) #在测试完一个训练点后，将初量子态调到下一个待测试点上
            observation = env.reset()
            action_list = [] #用来保存本回合所采取的动作，用于噪声分析
            fid_list = [] #用来保存本回合中的保真度，选择最大保真度对应的步骤作为后面噪声环境中动作的终止步骤
            while True:
                action = RL.choose_action(observation,tanxin)
                
                action_list = np.append(action_list,action)
                observation_, reward, done, fid = env.step(action)
                fid_list = np.append(fid_list,fid)
                observation = observation_
                
                if done:
                    break
                
            observation = env.reset()
            fid_list = list(map(float,fid_list))
            max_index = fid_list.index(max(fid_list))
            action_list = action_list[0:max_index+1]
            
            
            test_fid_noise = 0
            for action in action_list:
                
                observation_, reward, done, fid = env.step_noise_J_s(action)
                observation = observation_
                test_fid_noise = fid 
                
            test_fidelity_list[k,kk] = test_fid_noise 
    return test_fidelity_list
#--------------------------------------------------------------------------------

#--------------------------------------------------------------------------------
#测试某一特定点，分为两个部分：

#1. 得出该点在布洛赫球上的坐标值、最终保真度，优化脉冲

def testing_point(test_point_theta_ord,test_point_varpsi_ord):
    #在布洛赫球上测试上面选好的测试点，对训练好的模型进行测试
    
    global tanxin
    tanxin = 1
    
    actions = np.zeros((1,50)) - 1 #一共最多需要40次操作，但为了保险，多记录一点也无妨，没施加操作的位置都设为了 -1 以便于识别。
    action_step = 0
    states = np.zeros((50,4))


    #根据之前选好待测试点，依次输入神经网络，得到对应的最大最终保真度，并将其记录在矩阵中
    env.trrset(test_point_theta_ord,test_point_varpsi_ord)#将量子态重置到待测点上
    test_point_theta, test_point_varpsi = \
        env.test_theta[test_point_theta_ord], env.test_varpsi[test_point_varpsi_ord]
        
    print('test_point_theta:\n',test_point_theta,'\n\ntest_point_varpsi:\n',test_point_varpsi,'\n')
    #打印出选出来测试点
    observation = env.reset()
    states[0,:] = observation
    test_point_fid = 0
    
    while True:
        action = RL.choose_action(observation,tanxin)
        actions[0][action_step] = action  
        action_step += 1
        observation_, reward, done, fid = env.step(action)
        observation = observation_
        states[action_step] = observation
        test_point_fid = max(test_point_fid,fid)
        if done:
            break
    return test_point_fid, actions, states

#---------------------------------------------------------------------------

#2.得出该测试点在操作下，量子态的变化以及在布洛赫球上的位置
#输入量子态 states, 输出量子态在布洛赫球上的位置

def positions(states):
    #输入 states 的矩阵，行数为位置数，共 4 列标志着量子态的向量表示 [[1+2j],[3+4j]] 表示为：[1,3,2,4]
    #所以先将 state 表示变为 psi 表示 
    # b 矩阵第一列为 alpha,第二列为 beta
    b = np.zeros((states.shape[0],2),complex) 
    b[:,0] = states[:,0] + states[:,2]*1j
    b[:,1] = states[:,1] + states[:,3]*1j
    alpha = b[:,0]
    beta = b[:,1]
    
    #根据 alpha 和 beta 求直角坐标系下量子态的坐标
    z = 2*(alpha*alpha.conj())-1 #后面表示 z 的列向量中有多余的 -1 量，就是这里的原因。
    x = (beta*np.sqrt(2*(z+1))+beta.conj()*np.sqrt(2*(z+1)))/2
    y = (beta*np.sqrt(2*(z+1))-beta.conj()*np.sqrt(2*(z+1)))/(2*1j)
    
    #positions 矩阵为 位置数*行，3列，分别为 x,y,z
    positions = np.zeros((states.shape[0],3))
    positions[:,0], positions[:,1],positions[:,2] = x,y,z
    return positions
#---------------------------------------------------------------------------------

#---------------------------------------------------------------------------------
#将测试集的保真度从小到大排列出来，来展示保真度分布
def sort_fid(test_fidelity_list):
    sort_fid = []
    for i in range (test_fidelity_list.shape[0]):
        b = test_fidelity_list[i,:]
        sort_fid  = np.append(sort_fid,b)
    sort_fid.sort()
    return sort_fid
#--------------------------------------------------------------------------------


#---------------------------------------------------------------------------------
#主程序部分

if __name__ == "__main__":

    
    dt_=np.pi/40
    noise_a = 0 #噪声的幅值
    env = Env(action_space=list(range(4)),   #允许的动作数 0 ~ 4-1 也就是 4 个
               dt=dt_, theta_num=theta_num , varpsi_num=varpsi_num,
               test_theta_mul=test_theta_mul , test_varpsi_mul=test_varpsi_mul,
               noise_a=noise_a)              
        
    RL = DeepQNetwork(env.n_actions, env.n_features,
              learning_rate=0.0001,
              reward_decay=0.9, 
              e_greedy=0.99,
              replace_target_iter=250,
              memory_size=2000,
              e_greedy_increment=0.001, 
              )
    begin_training = time()
    fidelity_list = training() #训练
    end_training = time()
    training_time = end_training - begin_training
    print('\ntraing_time =',training_time)
    # print("\nFinal_fidelity=\n", fidelity_list,'\n')
    
    
    begin_testing = time()
    test_fidelity_list = testing() #测试
    # test_fidelity_list = testing_noise_J() #测试 J 噪声的影响
    # test_fidelity_list = testing_noise_h() #测试 h 噪声的影响
    end_testing = time()
    testing_time = end_testing - begin_testing
    print('\ntesting_time =',testing_time)
    
    
    # print('\ntest_fidelity_list:\n',test_fidelity_list)
    print('\nmean_test_fidelity:\n',np.mean(test_fidelity_list))
    

    
    #----------------
    #输出单个测试点来得到对其的操作、量子态及在布洛赫球上位置的变化
    # test_point_fid, actions, states = testing_point(-2,-10) #此处（-2，-10）点的意思是挑选上面测试点列表中的对应点来操作
    # print('\ntest_point_fid:\n',test_point_fid,'\ntest_point_actions:\n',actions)
    # print('\npositions:\n',positions(states))
    
    #----------------
    #将测试集的保真度从小到大排列出来
    # print(sort_fid(test_fidelity_list))






#----------------------------------------------------------------------------------
#在测噪声数据时，将以下代码粘贴到 console 更改 env.noise_a 的值就可以得到在对应噪声环境下测试保真度
#而不必再重新训练网络，可以节省大量的时间

# env.noise_a = 0
# print('\nnoise_a =',env.noise_a)
# env.noise = env.noise_a * env.noise_normal
# test_fidelity_list = testing_noise_J() #测试
# print('\nmean_test_fidelity:\n',np.mean(test_fidelity_list))

#----------------------------------------------------------------------------------


# 结果 -----------------------------------------------------------------------------------------------------

    # mean_test_fidelity:
    # 0.9874874461391332
    
    # sort_fid(test_fidelity_list):
    # [0.57653817, 0.69920063, 0.70491641, 0.80807764, 0.88165194,
    #    0.89795443, 0.9066836 , 0.93727376, 0.93867036, 0.9509263 ,
    #    0.95715646, 0.95748538, 0.95910642, 0.96226034, 0.96346267,
    #    0.96378692, 0.96454427, 0.96823265, 0.97039216, 0.97050846,
    #    0.9713286 , 0.97254987, 0.97407873, 0.97529218, 0.97557368,
    #    0.9757027 , 0.97619024, 0.97786442, 0.97922048, 0.98044782,
    #    0.98052501, 0.98331354, 0.98360785, 0.98417219, 0.98524544,
    #    0.98569496, 0.9864285 , 0.98643517, 0.98652997, 0.98678033,
    #    0.98825094, 0.98836257, 0.98910524, 0.98980162, 0.99004475,
    #    0.99005492, 0.99007425, 0.9901374 , 0.99017012, 0.99018441,
    #    0.99020039, 0.99022929, 0.99025076, 0.99031473, 0.99031572,
    #    0.99041271, 0.99049612, 0.99054433, 0.99059671, 0.99060669,
    #    0.99063191, 0.99063277, 0.99064523, 0.99066207, 0.99072649,
    #    0.99080576, 0.99081017, 0.99083682, 0.99087488, 0.99088654,
    #    0.99091265, 0.99093816, 0.99096542, 0.99097521, 0.99098341,
    #    0.99099992, 0.99101266, 0.99101313, 0.99102102, 0.99102502,
    #    0.99104541, 0.99105743, 0.99105915, 0.99107807, 0.99108743,
    #    0.99112278, 0.99116123, 0.99118199, 0.99119542, 0.99126696,
    #    0.99131719, 0.99134609, 0.9913647 , 0.9913795 , 0.99139296,
    #    0.99141278, 0.9914165 , 0.9914375 , 0.99148923, 0.99152454,
    #    0.99152836, 0.99159132, 0.99159729, 0.99163251, 0.99166274,
    #    0.99169728, 0.99180181, 0.99184807, 0.99190143, 0.99206314,
    #    0.99210392, 0.99212126, 0.99213115, 0.99219875, 0.99221228,
    #    0.99223704, 0.9923083 , 0.99234209, 0.9923672 , 0.99242523,
    #    0.99249292, 0.99250395, 0.99252748, 0.99253891, 0.99261432,
    #    0.99261519, 0.99262001, 0.99262076, 0.99262902, 0.99263757,
    #    0.99264494, 0.99266523, 0.99269156, 0.99269325, 0.99275514,
    #    0.99277425, 0.99277425, 0.99284379, 0.99296155, 0.99301424,
    #    0.99305726, 0.99307663, 0.99307723, 0.99308614, 0.99311264,
    #    0.99314347, 0.99320321, 0.99322387, 0.99340536, 0.99347195,
    #    0.99353036, 0.99361332, 0.99362537, 0.99366163, 0.99370354,
    #    0.99373045, 0.99377585, 0.99377903, 0.99378284, 0.9938401 ,
    #    0.99384315, 0.99386081, 0.99388993, 0.99389634, 0.99393355,
    #    0.99394535, 0.99395911, 0.99399431, 0.99403096, 0.99403107,
    #    0.99407444, 0.99410951, 0.99413086, 0.99420474, 0.99422058,
    #    0.99422365, 0.99429527, 0.99433896, 0.99435575, 0.99436647,
    #    0.99438912, 0.99439662, 0.99440574, 0.99446009, 0.99448524,
    #    0.99450309, 0.99450892, 0.99452451, 0.99455516, 0.99455968,
    #    0.99457895, 0.99473604, 0.99477011, 0.99479267, 0.99479522,
    #    0.99491012, 0.99494851, 0.99502019, 0.99505069, 0.99506435,
    #    0.99512213, 0.99513107, 0.99514332, 0.99533009, 0.99534409,
    #    0.99537765, 0.99537954, 0.99538279, 0.9953904 , 0.99540341,
    #    0.99546622, 0.99551129, 0.99554603, 0.99555516, 0.99559425,
    #    0.99562716, 0.9956447 , 0.99564593, 0.99566498, 0.99566979,
    #    0.9957739 , 0.99584681, 0.99593605, 0.99599056, 0.99610616,
    #    0.99610632, 0.99612232, 0.99622643, 0.99629493, 0.99630869,
    #    0.99638495, 0.99642201, 0.99642562, 0.99644169, 0.9964653 ,
    #    0.99649029, 0.99654431, 0.99658425, 0.99668245, 0.99672718,
    #    0.99674618, 0.99674816, 0.99676138, 0.9967627 , 0.99676968,
    #    0.99678978, 0.99685613, 0.99685617, 0.99686658, 0.99688415,
    #    0.99689516, 0.99690167, 0.99692111, 0.99692277, 0.99694832,
    #    0.9970713 , 0.99708118, 0.9971016 , 0.99711513, 0.99716176,
    #    0.99719557, 0.9972775 , 0.99727968, 0.99728019, 0.99741092,
    #    0.9974153 , 0.99741837, 0.99742396, 0.9974351 , 0.99759323,
    #    0.99761321, 0.9976659 , 0.99768706, 0.99771446, 0.99772368,
    #    0.99775849, 0.99778098, 0.99784369, 0.99785467, 0.99786739,
    #    0.99787207, 0.99790482, 0.99790536, 0.99791   , 0.99793075,
    #    0.99802706, 0.99806035, 0.99807208, 0.99811254, 0.99814385,
    #    0.99819772, 0.99827246, 0.9983099 , 0.99835213, 0.99838325,
    #    0.99845207, 0.9984785 , 0.99849118, 0.99850543, 0.99855783,
    #    0.99859491, 0.9986231 , 0.99865306, 0.99888467, 0.99890153,
    #    0.99891659, 0.99900414, 0.99901209, 0.99902576, 0.99911286,
    #    0.99911988, 0.99914198, 0.99922989, 0.99923623, 0.99926191,
    #    0.99936467, 0.99938809, 0.99939355, 0.99939778, 0.99946024]
