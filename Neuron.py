import enum
import numpy as np
from hyperopt import hp
import Utils.GraphUtils as gu
import Models.Fiuri.Connection as con
import Utils.GraphUtils as gu


class Neuron:

#Pensar un poco como es la inicializacion
    def __init__(self, nname):
        self.neuronName = nname
        self.state= None
        #learnable param
        self.threshold=0
        self.testThreshold=0

        #estado interno no aprendible, es el estado neuronal
        self.internalstate=0
        #estado de salida no aprendible, es el estado neuronal
        self.outputstate=0
        #buffered values, needed to not depend on the evaluation order
        self.bufferedInternalState=0
        self.bufferedOutputState=0

        #learnable
        self.decayfactor=0
        self.testDecayfactor=0


        self.lastActivationTrace = []
        self.stateToTrace = {'TH': [], 'DF': []}


    def __str__(self):
        return "[neu: %s, InState: %s, Threshold: %s, OutState: %s, DF: %s]\n" % (
            self.neuronName, self.internalstate,self.threshold,self.outputstate,self.decayfactor)

    def __repr__(self):
        return "[neu: %s, InState: %s, Threshold: %s]" % (
            self.neuronName, self.internalstate, self.threshold)

    def __eq__(self, other):
        if not isinstance(other, Neuron):
            # don't attempt to compare against unrelated types
            return NotImplemented
        return self.neuronName == other.neuronName

    def getName(self):
        return self.neuronName

    # initialized in Quiescent stare
    def initialize(self, th, inst,oust,df):
        #self.internalstate=0
        #self.bufferState=NeuronState.Quiescent
        self.threshold = th
        self.testThreshold = th
        self.internalstate=inst
        self.outputstate=oust
        self.testDecayfactor=df
        self.decayfactor=df
        self.lastActivationTrace = []
        self.bufferedInternalState=inst
        self.bufferedOutputState=oust


    def getStateToTrace(self):
        return self.stateToTrace

    def updateStateToTrace(self):
        self.stateToTrace['TH'].append(self.testThreshold)
        self.stateToTrace['DF'].append(self.testDecayfactor)
        #print('passed by the conection's updateStateToTrace with values for vleak %s'%(self.stateToTrace['Vleak']))

    def graphVariableTraces(self,folder):
        gu.graphVarTraces(self.stateToTrace,folder,self.neuronName)

    def resetLastActivationTrace(self):
        self.lastActivationTrace=[]
    def getLastActivaionTrace(self):
        return self.lastActivationTrace
    def recordActivationTrace(self):
        self.lastActivationTrace.append(str(round(self.internalstate,3)))

    def setInternalState(self, st):
        self.internalstate = st
    def getInternalState(self):
        return self.internalstate

    def setOutputState(self, st):
        self.outputstate = st
    def getOutputState(self):
        return self.outputstate

#    def commitComputation(self):
#        self.state=self.bufferState

    def setThreshold(self, th):
        self.threshold = th
    def getThreshold(self):
        return self.threshold
    def setTestThreshold(self, th):
        self.testThreshold = th
    def getTestThreshold(self):
        return self.testThreshold

    def setDecayFactor(self, df):
        self.decayfactor = df
    def getDecayFactor(self):
        return self.decayfactor
    def setTestDecayFactor(self, df):
        self.testDecayfactor = df
    def getTestDecayFactor(self):
        return self.testDecayfactor


    # in this model we don't have voltages, we have states but the method needs to be named in this way to maintain
    # the same interface with the other models.
    def computeVnext(self, dendriticConnections):
        #currState = self.internalstate
        currInfluence=0
        #denConnAmount= len(dendriticConnections)
        for dc in dendriticConnections:
            sourceOutState = dc.getSource().getOutputState()
            if dc.isType(con.ConnectionType.ChemEx):
                currInfluence = currInfluence + dc.getTestWeight()*sourceOutState
            elif dc.isType(con.ConnectionType.ChemIn):
                currInfluence = currInfluence - dc.getTestWeight()*sourceOutState
            else:
                if(sourceOutState<self.internalstate):
                    currInfluence = currInfluence - dc.getTestWeight() * sourceOutState
                elif(sourceOutState>self.internalstate):
                    currInfluence = currInfluence + dc.getTestWeight() * sourceOutState
        currState=self.internalstate+currInfluence
        if currState < -10:
            currState = -10
        elif currState > 10:
            currState = 10
        #here is where we establish the equivalence between internal and output states
        if currState > self.testThreshold:
            #may be this should be without -self.threshold term
            self.bufferedOutputState=currState-self.testThreshold
            self.bufferedInternalState=currState-self.testThreshold
        elif currState==self.internalstate:
            self.bufferedOutputState=0
            self.bufferedInternalState=self.internalstate-self.testDecayfactor
        else:
            self.bufferedOutputState=0
            self.bufferedInternalState=currState


    def commitComputation(self):
        self.internalstate = self.bufferedInternalState
        self.outputstate=self.bufferedOutputState

    #check if still used
    def isSameAs(self,neu2):
        return self.testThreshold==neu2.testThreshold and self.testDecayfactor==neu2.testDecayfactor and self.internalstate==neu2.internalstate and self.outputstate==neu2.outputstate

#needed for Random Seek
    def commitNoise(self):
        self.threshold = self.testThreshold
        self.decayfactor=self.testDecayfactor

    def revertNoise(self):
        self.testThreshold = self.threshold
        self.testDecayfactor=self.decayfactor

#needed for PYGAD
    def getComponentOfIndividualForPYGAD(self):
        return [self.testThreshold,self.testDecayfactor]

#needed for HyperOpt
#REVISAR LIMITES
    def getVariablesForHyperOpt(self):
        return {
                #self.neuronName+'_tth':hp.uniform(self.neuronName+'_tth',0.0,1.0),
                self.neuronName+'_tth':hp.uniform(self.neuronName+'_tth',0.0,1.0),
                self.neuronName + '_tdf': hp.uniform(self.neuronName + '_tdf', 0, 0.5),
        }






def loadNeuron(xmlNeu):
    neuToReturn = Neuron(xmlNeu.attrib['name'])
    neuToReturn.threshold=float(xmlNeu.attrib['th'])
    neuToReturn.df=float(xmlNeu.attrib['df'])
    neuToReturn.threshold=float(xmlNeu.attrib['internalstate'])
    neuToReturn.df=float(xmlNeu.attrib['outputstate'])
    neuToReturn.revertNoise()
    return neuToReturn




