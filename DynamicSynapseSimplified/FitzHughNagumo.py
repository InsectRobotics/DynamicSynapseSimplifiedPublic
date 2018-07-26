#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 14:56:07 2018

@author: chitianqilin
"""
import numpy as np
from RK4 import rk4
import matplotlib.pyplot as plt
from collections import deque
import copy

class FHNN:
    def __init__(self, NumberOfNeurons, a=None, b=None, c=None, I=None, V=None, W=None, t=0, scale = 1):
        self.a=a*np.ones(NumberOfNeurons) if a is not None else 0.08*np.ones(NumberOfNeurons)
        self.b=b*np.ones(NumberOfNeurons) if b is not None else 2*np.ones(NumberOfNeurons)
        self.c=c*np.ones(NumberOfNeurons) if c is not None else 0.8*np.ones(NumberOfNeurons)
        self.I=I*np.ones(NumberOfNeurons) if I is not None else 0*np.ones(NumberOfNeurons)
        self.Vn=V*np.ones(NumberOfNeurons) if V is not None else 0*np.ones(NumberOfNeurons)
        self.Wn = W*np.ones(NumberOfNeurons) if W is not None else 0*np.ones(NumberOfNeurons)
        self.Vp = self.V*np.ones(NumberOfNeurons) if not V is None else 0*np.ones(NumberOfNeurons)
        self.Wp = self.W*np.ones(NumberOfNeurons) if not W is None else 0*np.ones(NumberOfNeurons)
        self.t=t
        self.scale = scale
    def Derivative (self, state, inputs, NeuronID=None):
        V, W=state
        I=inputs
        Dv=(V-np.power(V,3)-W + I)*self.scale
        if NeuronID==None:
            DW=(self.a*(self.b*V-self.c*W))*self.scale
        else:
            DW=(self.a[NeuronID]*(self.b[NeuronID]*V-self.c[NeuronID] *W))*self.scale
        return np.array([Dv, DW])
    
    def StepDynamics(self, dt, I):
        self.t += dt
        self.I = I
        self.Vn, self.Wn = rk4(dt, [self.Vp, self.Wp], self.I, self.Derivative)
        assert np.logical_not(np.logical_or(np.any(np.isnan([self.Vn,self.Vp])),np.any(np.isinf([self.Vn,self.Vp])))), \
               "\nself.Vn=" + str(self.Vn) \
               +"\nself.Wn=" + str(self.Wn)\
               +"\nself.Vp=" + str(self.Vp) \
               +"\nself.Wp=" + str(self.Wp)\
               +"\nself.I=" + str(self.I)
        self.Vn[self.Vn>2]=2
        self.Vn[self.Vn<-2]=-2
        self.Wn[self.Wn>2]=2
        self.Wn[self.Wn<-2]=-2     
        return[self.Vn, self.Wn]
    
    def Update(self):
        assert np.logical_not(np.logical_or(np.any(np.isnan([self.Vn,self.Vp])),np.any(np.isinf([self.Vn,self.Vp])))), \
               "\nself.Vn=" + str(self.Vn) \
               +"\nself.Wn=" + str(self.Wn)\
               +"\nself.Vp=" + str(self.Vp) \
               +"\nself.Wp=" + str(self.Wp)
               
        self.Vp, self.Wp = self.Vn, self.Wn

    def UpdateParameters(self,Parameters):
        Parameters=np.array(Parameters)
        self.scale=Parameters[:, 0]
        self.a=Parameters[:, 1]
        self.b=Parameters[:, 2]
        self.c=Parameters[:, 3]
        
    def InitRecording(self):
        self.Trace={
                    'Vn': deque(),
                    'Wn': deque(),
                    't': deque(),
                    'I': deque()
                }
        
    def Recording(self):
        for key in self.Trace:
            exec("self.Trace['%s'].append(copy.deepcopy(self.%s))"%(key, key))
           
    def Plot(self, NeuronID=0, DownSampleRate=1):
        FigureDict = dict()
        FigureDict[str(NeuronID)]=plt.figure()
        Index=np.s_[::DownSampleRate, NeuronID]

        lines=plt.plot(np.array(self.Trace['t'])[::DownSampleRate],\
                          np.vstack((np.array(self.Trace['Vn'])[Index],\
                                    np.array(self.Trace['Wn'])[Index],\
                                    np.array(self.Trace['I'])[Index])).T)
        plt.legend(lines, ['v', 'w', 'I'], loc=4)    
#        lines=plt.plot(np.array(self.Trace['t'])[::DownSampleRate],\
#                          np.vstack((np.array(self.Trace['Vn'])[Index],\
#                                    np.array(self.Trace['Wn'])[Index],\
#                                    np.array(self.Trace['I'])[Index],\
#                                    np.array(self.Trace['Vn'])[Index] - np.array(self.Trace['Wn'])[Index])).T)
#        plt.legend(lines, ['v', 'w', 'I', 'v+w'], loc=4)    
        plt.xlabel('Time (ms)')
        
    def PlotPhasePortrait(self, I, xlim, ylim, fig=None, ax=None, NeuronID=0,DownSampleRate=1):
        if fig == None or ax == None:
            fig, ax = plt.subplots(1,sharex=False )#, figsize=(20, 12)
        Vs=np.linspace(-1.5,1.5)
        colors=['r','c']
        if (type(I) is list and len(I)==2): 
            ax.plot( Vs,Vs-np.power(Vs, 3)+I[0] ,'r-', lw=2, label='v-nullcline (start)' ) #
            ax.plot( Vs,Vs-np.power(Vs, 3)+I[1] ,'c-', lw=2, label='v-nullcline (end)' ) 
        else:
            ax.plot( Vs,Vs-np.power(Vs, 3)+I ,'r-', lw=2, label='v-nullcline' ) 
        print (self.c[NeuronID])
        ax.plot( Vs,self.b[NeuronID]/self.c[NeuronID]*Vs ,'b-', lw=2, label='w-nullcline' ) #
        Vspace=np.linspace(xlim[0],xlim[1], num=30)
        Uspace=np.linspace(ylim[0],ylim[1], num=20)
        Vstep=Vspace[1]-Vspace[0]
        Ustep=Uspace[1]-Uspace[0]
        if (type(I) is list and len(I)==2):
            for i1 in range(len(I)):
                V1 , U1  = np.meshgrid(Vspace+(float(i1)*Vstep/len(I)), Uspace+(float(i1)*Ustep/len(I))) 
                DV1, DU1=self.Derivative([V1, U1],I[i1], NeuronID=0)
                M = np.hypot(DV1, DU1)
#                M=M+1
#                logM=np.log(M)
#                RatioM=logM / M
#                DV1Ratioed=DV1*RatioM
#                DU1Ratioed=DU1*RatioM
#                DV1Normal=DV1Ratioed#/np.max(DV1Ratioed)
#                DU1Normal=DU1Ratioed#/np.max(DU1Ratioed)
#                ratioV=Vstep/np.max(np.abs(DV1Normal))*0.5
#                ratioU=Ustep/np.max(np.abs(DU1Normal))*0.5
#                
#                DV1Scaled=DV1Ratioed*ratioV*ratioU
#                DU1Scaled=DU1Ratioed*ratioV*ratioU
#
                #ax.quiver(V1 , U1, DV1, DU1, M, color=colors[i1], edgecolors=(colors[i1]),width=0.002, scale=100)#pivot='mid')
#                ax.quiver(V1 , U1, DV1Scaled, DU1Scaled, color=colors[i1], edgecolors=(colors[i1]),width=0.002, scale=100)#pivot='mid')
                ax.quiver(V1 , U1, DV1, DU1, color=colors[i1], edgecolors=(colors[i1]),width=0.002,  angles='xy')#scale=100,pivot='mid')

        else:
            V1 , U1  = np.meshgrid(Vspace, Uspace) 
            DV1, DU1=self.Derivative([V1, U1],I, NeuronID=0)
            M = np.hypot(DV1, DU1)
#            M=M+1
#            logM=np.log(M)
#            RatioM=logM / M
#            DV1Ratioed=DV1*RatioM
#            DU1Ratioed=DU1*RatioM
#            DV1Normal=DV1Ratioed#/np.max(DV1Ratioed)
#            DU1Normal=DU1Ratioed#/np.max(DU1Ratioed)
#            ratioV=Vstep/np.max(np.abs(DV1Normal))*0.5
#            ratioU=Ustep/np.max(np.abs(DU1Normal))*0.5
#            
#            DV1Scaled=DV1Ratioed*ratioV*ratioU
#            DU1Scaled=DU1Ratioed*ratioV*ratioU
#            ax.quiver(V1 , U1, DV1Scaled, DU1Scaled, (M), width=0.002, scale=100)#pivot='mid')
            ax.quiver(V1 , U1,DV1, DU1, M,   width=0.002,  angles='xy')#scale=100,pivot='mid')
        #ax.legend(bbox_to_anchor=(0.6, 0.2), loc=2, borderaxespad=0.,prop={'size':12})
        Index=np.s_[::DownSampleRate, NeuronID]
        ax.plot(np.array(self.Trace['Vn'])[Index], np.array(self.Trace['Wn'])[Index], 'b-', label='Trajectory')
        ax.legend(prop={'size':12}, loc=0)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.grid()
        plt.xlabel('V')
        plt.ylabel('w')
        return [fig, ax]
    
if __name__ == "__main__":

    T = 10000 #ms
    TimeList, dt = np.linspace(0, T, 1000, retstep=True)
    NumberOfNeurons=2
    AFHNN = FHNN(NumberOfNeurons,scale = 0.02)
    AFHNN.InitRecording()
    I=np.array([(TimeList/T-0.5)*4])
    for step in range(len(TimeList)):
        AFHNN.StepDynamics(dt, I[:,step])   
        AFHNN.Recording()
        AFHNN.Update()
    AFHNN.Plot(0)
    fig, ax=AFHNN.PlotPhasePortrait([I[0][0],I[0][-1]], xlim=np.array([-1.5,1.5]), ylim=np.array([-1+I[0].min(),1+I[0].max()]))
    ax.plot()