#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 15:45:33 2017

@author: chitianqilin
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Mar 26 21:53:42 2017

@author: chitianqilin
"""

import numpy as np
import copy
import matplotlib as mpl
import matplotlib.pyplot as plt
import time
import dill
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.backends.backend_pdf import PdfPages 
from cycler import cycler
from collections import deque
import os

def relu(x):
    return np.maximum(x,0) 
       
class DynamicSynapseArray:
    def __init__(self, NumberOfSynapses = [1,3], Period=None, tInPeriod=None, PeriodVar=None,\
                 Amp=None, WeightersCentre = None, WeightersCentreUpdateRate = 0.000012, WeightersOscilateDecay=0.0000003, \
                ModulatorAmount=0, InitAmp=0.4, t = 0, dt=1, NormalizedWeight=False):
                    # LearningRuleOsci=True, LearningRulePre=False
        
        self.NumberOfSynapses = NumberOfSynapses#[1] 
        
        self.dt = dt
        self.t = t
        self.PeriodCentre=np.ones(NumberOfSynapses).astype(np.float)*Period if Period is not None else 1000+100*(np.random.rand(*NumberOfSynapses)-0.5)
        self.Period = copy.deepcopy(self.PeriodCentre)
        self.tInPeriod=np.ones(NumberOfSynapses).astype(np.float)*tInPeriod  if tInPeriod is not None else np.random.rand(*NumberOfSynapses)*self.Period
        
        self.PeriodVar = np.ones(NumberOfSynapses).astype(np.float)*PeriodVar if PeriodVar is not None else np.ones(NumberOfSynapses).astype(np.float)*0.1
        self.Amp = np.ones(NumberOfSynapses).astype(np.float)*Amp if Amp is not None else np.ones(NumberOfSynapses).astype(np.float)*0.2

        self.WeightersCentre = np.ones(NumberOfSynapses)*WeightersCentre if WeightersCentre is not None else (np.random.rand(*NumberOfSynapses)-0.5)*InitAmp
        self.NormalizedWeight=NormalizedWeight  
        print (self.WeightersCentre)
        print (np.sum(self.WeightersCentre,axis=1))
        if NormalizedWeight:
            self.WeightersCentre /= np.sum(self.WeightersCentre,axis=1)[:,None]
        self.WeightersCentreUpdateRate = np.ones(NumberOfSynapses)*WeightersCentreUpdateRate if WeightersCentreUpdateRate is not None else  np.ones(NumberOfSynapses)* 0.000012

        self.Weighters = self.WeightersCentre + self.Amp*np.sin(self.tInPeriod/self.Period*2*np.pi)
        self.WeightersLast = copy.deepcopy(self.Weighters)
        self.WeightersOscilateDecay = np.ones(NumberOfSynapses)*WeightersOscilateDecay  if WeightersOscilateDecay is not None else  np.ones(NumberOfSynapses) 

        self.ModulatorAmount = np.ones(NumberOfSynapses) *  ModulatorAmount
        self.ZeroCross=np.ones(NumberOfSynapses, dtype=bool)

#        self.LearningRuleOsci=LearningRuleOsci
#        self.LearningRulePre=LearningRulePre
#        self.LearningRulePreFactor=1
#        self.WeightersCentreVar=np.zeros(self.WeightersCentre.shape)

    def StepSynapseDynamics(self, dt,t, ModulatorAmount,PreSynActivity=None):
        
        if dt is None:
            dt = self.dt
        self.t = t
        self.tInPeriod += dt
        self.Weighters = self.WeightersCentre + self.Amp*np.sin(self.tInPeriod/self.Period*2*np.pi)
#        self.WeightersCentre += (self.Weighters-self.WeightersCentre)*relu(ModulatorAmount) *self.WeightersCentreUpdateRate*dt
        self.WeightersCentre += (self.Weighters-self.WeightersCentre)*ModulatorAmount *self.WeightersCentreUpdateRate*dt
#        if self.LearningRuleOsci==True:        
#            self.WeightersCentreVar += (self.Weighters-self.WeightersCentre)*ModulatorAmount *self.WeightersCentreUpdateRate*dt
#        if self.LearningRulePre==True:
#            self.WeightersCentreVar += (self.LearningRulePreFactor*PreSynActivity-self.WeightersCentre)*ModulatorAmount *self.WeightersCentreUpdateRate*dt
#        self.WeightersCentre += self.WeightersCentreVar    

        if self.NormalizedWeight:
            self.WeightersCentre /= np.sum(np.abs(self.WeightersCentre),axis=1)[:,None]

        self.ModulatorAmount=np.ones(self.NumberOfSynapses)*ModulatorAmount
        self.Amp *= np.exp(-self.WeightersOscilateDecay*self.ModulatorAmount*dt)
        self.ZeroCross = np.logical_and(np.less(self.WeightersLast, self.WeightersCentre),np.greater_equal(self.Weighters, self.WeightersCentre))
        self.tInPeriod[self.ZeroCross] = self.tInPeriod[self.ZeroCross]%self.Period[self.ZeroCross]
#        self.Period[self.ZeroCross] += (np.random.rand(*self.NumberOfSynapses)[self.ZeroCross]-0.5)*self.PeriodVar[self.ZeroCross]*self.Period[self.ZeroCross]+(self.PeriodCentre-self.Period)[self.ZeroCross]*0.03
        self.Period[self.ZeroCross] = np.random.normal(loc=self.PeriodCentre[self.ZeroCross], scale=self.PeriodCentre[self.ZeroCross]*0.1)
        self.WeightersLast = self.Weighters
        return self.Weighters
        
    def InitRecording(self):
        self.RecordingState = True
        self.Trace = {'Weighters': deque(),
                        'WeightersCentre' : deque(),
                        'ModulatorAmount' : deque(),
                        'Amp' : deque(),
                        'Period' : deque(),
                        'tInPeriod' : deque(),
                        't': deque()
                        }
    def Recording(self):
        Temp = None
        for key in self.Trace:
            exec("Temp = self.%s" % (key))
            self.Trace[key].append(copy.deepcopy(Temp))

    #%%                  
    def plot(self, path='', savePlots = False, StartTimeRate=0.3, DownSampleRate=10,linewidth =1, FullScale=False, NameStr=None, NeuronID=0):
    #    plt.rc('axes', prop_cycle=(cycler('color',['C0','C1','C2','C3','C4','C5','C6','C7','C8','C9','b','k'])))
        mpl.rcParams['axes.prop_cycle']=cycler('color',['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd','#8c564b','#e377c2','#7f7f7f','#bcbd22','#17becf','b','k'])
    #    mpl.rcParams['axes.prop_cycle']=cycler(color='category20')
#        if Trace is None:
#            Trace = self.Trace
        if NameStr is None:
            NameStr = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
        TracetInS=np.array(self.Trace['t'])[::DownSampleRate].astype(float)/1000
        NumberOfSteps = len(TracetInS)
        if StartTimeRate == 0:
              StartStep = 0
        else:
              StartStep = NumberOfSteps - int(NumberOfSteps*StartTimeRate)
        
        FigureDict={}
        FigureDict['ASynapse'] = plt.figure()
        SynapseID=0
        figure0lines0, = plt.plot(TracetInS, np.array(self.Trace['ModulatorAmount'])[::DownSampleRate, NeuronID, SynapseID],  linewidth= 1)
        figure0lines1, = plt.plot(TracetInS, np.array(self.Trace['WeightersCentre'])[::DownSampleRate, NeuronID, SynapseID], linewidth= 1)
        figure0lines2, = plt.plot(TracetInS, np.array(self.Trace['Weighters'])[::DownSampleRate, NeuronID, SynapseID],  linewidth= 1)
        plt.legend([figure0lines2,figure0lines1, figure0lines0], ['Weight Fluctuation','Fluctuation Centre','Modulator Amount',], loc=4)
        plt.title('Example Dynamics of a Synapse')
#        plt.ylim([-2,2])
        FigureDict['Weighters'] = plt.figure()

        labels = [str(i) for i in range(self.Trace['Weighters'][0].shape[1])]
        figure1lines = plt.plot(TracetInS, np.array(self.Trace['Weighters'])[::DownSampleRate, NeuronID],  label=labels, linewidth= linewidth)
        plt.legend(figure1lines, labels)
        plt.xlabel('Time (s)')
        plt.title('Instantaneous Synaptic Strength')
              
        X=np.array(self.Trace['Weighters'])[StartStep:NumberOfSteps][::DownSampleRate, NeuronID,0]
        Y=np.array(self.Trace['Weighters'])[StartStep:NumberOfSteps][::DownSampleRate, NeuronID,1]
        Z=np.array(self.Trace['Weighters'])[StartStep:NumberOfSteps][::DownSampleRate, NeuronID,2]
        
        FigureDict['2Weighters'] = plt.figure()
        plt.plot(X,Y)
        plt.xlabel('Time (s)')
        plt.title('2 Instantaneous Synaptic Strength')
        plt.xlabel('Instantaneous Synaptic Strength 0')
        plt.ylabel('Instantaneous Synaptic Strength 1') 
        
        FigureDict['3Weighters'] = plt.figure()
        ax = FigureDict['3Weighters'].add_subplot(111, projection='3d')
        ax.plot(X,Y,zs=Z)
        ax.set_xlabel('Instantaneous Synaptic Strength 0')
        ax.set_ylabel('Instantaneous Synaptic Strength 1')
        ax.set_zlabel('Instantaneous Synaptic Strength 2')
        # Create cubic bounding box to simulate equal aspect ratio
        max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max()
        Xb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][0].flatten() + 0.5*(X.max()+X.min())
        Yb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][1].flatten() + 0.5*(Y.max()+Y.min())
        Zb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][2].flatten() + 0.5*(Z.max()+Z.min())
        # Comment or uncomment following both lines to test the fake bounding box:
        for xb, yb, zb in zip(Xb, Yb, Zb):
           ax.plot([xb], [yb], [zb], 'w',linewidth= linewidth)
        if FullScale:
            ax.set_xlim(-1,1)
            ax.set_ylim(-1,1)
            ax.set_zlim(-1,1)    
        
        FigureDict['WeightersCentre'] = plt.figure()
        figure4lines = plt.plot(TracetInS, np.array(self.Trace['WeightersCentre'])[::DownSampleRate, NeuronID], label=labels, linewidth= linewidth)
        plt.legend(figure4lines, labels)
        plt.title('Center of Synaptic Strength Oscillation')
        plt.xlabel('Time (s)')
        
        FigureDict['Period'] = plt.figure()
        figure5lines = plt.plot(TracetInS, np.array(self.Trace['Period'])[::DownSampleRate, NeuronID], label=labels, linewidth= linewidth)
        plt.legend(figure5lines, labels)
        plt.title('Period')   
        plt.xlabel('Time (s)')
        plt.xlabel('Period (s)')
        FigureDict['tInPeriod'] = plt.figure()
        figure6lines = plt.plot(TracetInS, np.array(self.Trace['tInPeriod' ])[::DownSampleRate, NeuronID], label=labels, linewidth= linewidth)
        plt.legend(figure6lines, labels)
        plt.title('tInPeriod' )   
        plt.xlabel('Time (s)')
        plt.xlabel('Period (s)')
        FigureDict['Amp'] = plt.figure()
        figure6lines = plt.plot(TracetInS, np.array(self.Trace['Amp' ])[::DownSampleRate, NeuronID], label=labels, linewidth= linewidth)
        plt.legend(figure6lines, labels)
        plt.title('Amp' )   
        plt.xlabel('Time (s)')

        FigureDict['ModulatorAmount'] = plt.figure()
        figure7lines = plt.plot(TracetInS, np.array(self.Trace['ModulatorAmount'])[::DownSampleRate, NeuronID], label=labels, linewidth= linewidth)
        plt.legend(figure7lines, labels)
        plt.xlabel('Time (s)')
        plt.title('ModulatorAmount') 
    #%
        FigureDict['PoincareMap'] = plt.figure()
        figure8ax1 = FigureDict['PoincareMap'].add_subplot(111)  
        points0,points1 = CrossAnalysis(np.array(self.Trace['Weighters'])[::DownSampleRate, NeuronID, 0], np.array(self.Trace['WeightersCentre'])[::DownSampleRate, NeuronID, 0],np.array(self.Trace['Weighters'])[::DownSampleRate, NeuronID],TracetInS)
        if FullScale:
            figure8ax1.set_xlim(0,1)
            figure8ax1.set_ylim(0,1)
        print('points0')
        print(points0['points'])
        print('points1')
        print(points1['points'])
        
        pointsploted0 = figure8ax1.scatter(points0['points'][:,1],points0['points'][:,2],c=points0['t'], cmap=plt.cm.get_cmap('Greens'), marker=".", edgecolor='none') #c=c, ,  cmap=cm
        pointsploted1 = figure8ax1.scatter(points1['points'][:,1],points1['points'][:,2],c=points1['t'], cmap=plt.cm.get_cmap('Blues'), marker=".", edgecolor='none')
        #plt.legend(figure7lines, labels)
        plt.colorbar(pointsploted0)
        plt.colorbar(pointsploted1)
        plt.title('Poincare map') 
        plt.xlabel('Instantaneous Synaptic Strength 1')
        plt.ylabel('Instantaneous Synaptic Strength 2')
     #% 
#        for key in FigureDict:
#            FigureDict[key].tight_layout()

        if savePlots == True:
            if not os.path.exists(path):
                os.makedirs(path) 
            pp = PdfPages(path+"DynamicSynapse"+NameStr+'.pdf')
            for key in FigureDict:
                FigureDict[key].savefig(pp, format='pdf')
            pp.close()
    #        Figures = {'TraceWeighters':figure1, 'TraceWeighterVarRates':figure2, 'TraceWeighterInDendrite':figure3, '2TraceWeighters':figure4, '3DTraceWeighters':figure5, 'WeightersCentre':figure6, 'Damping':figure7,'EquivalentVolume':figure8,'Poincare map':figure9}
    #        with open(path+"DynamicSynapse"+TimOfRecording+'.pkl', 'wb') as pkl:
    #            dill.dump(Figures, pkl)
    
                
        return FigureDict,ax
#%%
def CrossAnalysis(Oscillate,Reference,OscillateArray,Tracet):
    points0={'t':[],'points':[]}
    points1={'t':[],'points':[]}
    GreaterThanCentre=(Oscillate[0]>Reference[0])
    print(Oscillate[0])
    print(Reference[0])
    for i1 in range(len(Oscillate)):
        
#        print(Oscillates[i1,0])
#        print(References[i1,0])
        if GreaterThanCentre == True:
            if Oscillate[i1]<Reference[i1]:
                #print(GreaterThanCentre)
                #print(Oscillates[i1,0])
                points0['points'].append(OscillateArray[i1])
                points0['t'].append(Tracet[i1])
                GreaterThanCentre = False
        elif GreaterThanCentre ==  False:
            if Oscillate[i1]>Reference[i1]:
                #print (GreaterThanCentre)
                #print(Oscillates[i1,0])
                points1['points'].append(OscillateArray[i1])
                points1['t'].append(Tracet[i1])
                GreaterThanCentre = True
    #c = np.empty(len(m[:,0])); c.fill(megno)
    points0['points']=np.array(points0['points'])
    points1['points']=np.array(points1['points'])
    points0['t']=np.array(points0['t'])
    points1['t']=np.array(points1['t'])
    return points0, points1
    
    
            
def SimulationLoop(ADSRA,dt, NumberOfSteps, Arg0 , Arg1 , phase=0,Index0=0,Index1=0):    
    ADSRA.tauWV = Arg0
    ADSRA.aWV = Arg1
    
    ADSRA.InitRecording(NumberOfSteps)  
    Tracet = np.zeros(NumberOfSteps)           
    for step in range(NumberOfSteps):
    #        WeightersLast = copy.deepcopy(Weighters)
    #        WeighterVarRatesLast = copy.deepcopy(WeighterVarRates)
        ADSRA.StateUpdate()
        ADSRA.StepSynapseDynamics( SimulationTimeInterval,0)
        if  ADSRA.RecordingState:
            ADSRA.Recording()  
            Tracet[step] = step*SimulationTimeInterval
            #%%
        if step%(100000./dt)<1:
           print ('phase=%s,Index0=%d, Index1=%d, tauWV=%s, aWV=%s, step=%s'%(phase,Index0,Index1,ADSRA.tauWV, ADSRA.aWV,step))    

    return ADSRA
            
if __name__=="__main__":
    InintialDS =1
    SingleSimulation = 1
    TimOfRecording=time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    np.random.seed(0)
    if InintialDS:
        
        SimulationTimeLenth = 100*60*1000  
        Period = 20000
        dt = SimulationTimeInterval = 33
        NumberOfSteps = int(SimulationTimeLenth/SimulationTimeInterval)
        PeriodSteps = int(Period/SimulationTimeInterval)
        ModulatorAmount = np.zeros(NumberOfSteps)
        ModulatorAmount[PeriodSteps/2:PeriodSteps]=1
        ModulatorAmount[PeriodSteps*2:PeriodSteps*2+PeriodSteps/2]=1
        NumberOfNeuron=2
        NumberOfSynapses = 3
        WeightersCentre =0# np.ones((NumberOfNeuron,NumberOfSynapses))*0+ 0.4* (np.random.rand(NumberOfNeuron,NumberOfSynapses)-0.5)

        ADSRA=DynamicSynapseArray(NumberOfSynapses = (NumberOfNeuron,NumberOfSynapses), Period=Period, tInPeriod=None, PeriodVar=0.1,\
                 Amp=1, WeightersCentre = WeightersCentre, WeightersCentreUpdateRate = 0.000012, WeightersOscilateDecay=0.0000003) #tInPeriod=None
               

        ADSRA.InitRecording() 
#%%       
    if SingleSimulation: 
#%%     
        for step in range(NumberOfSteps):
            
            ADSRA.StepSynapseDynamics( SimulationTimeInterval,  step*SimulationTimeInterval, 0)

            if  ADSRA.RecordingState:
                ADSRA.Recording()  
            if step % 1000 == 0:
                print('%d of %d steps'%(step,NumberOfSteps))
        
#%%
        FigureDict,ax = ADSRA.plot( path='/media/archive2T/chitianqilin/SimulationResult/DynamicSynapseRandomSin/Plots/',DownSampleRate=1, savePlots=True, linewidth= 0.2, NameStr=TimOfRecording) #path=
#%%
        