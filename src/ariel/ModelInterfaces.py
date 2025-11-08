import xml.etree.ElementTree as ET
import numpy as np
#import np.random as rng


distortions={}
variances={}

class BinaryInterface:

    def __init__(self, name, posNeu, negNeu, type):
        self.name = name
        self.maxState =20
        self.minState = -20
        self.valleyVal = 0
        self.minValue = 0
        self.maxValue=0
        self.value = 0
        self.positiveNeuron = posNeu
        self.negativeNeuron = negNeu
        self.type = type

    def __str__(self):
        return "<%s,%s,%s,%s,%s>\n" % (
            self.name, self.type, self.positiveNeuron.getName(), self.negativeNeuron.getName(), self.value)

    def __repr__(self):
        return "[pos: %s, neg: %s, value: %s]" % (
            self.positiveNeuron, self.negativeNeuron, self.value)

    def reset(self):
        self.value = 0

    def getName(self):
        return  self.name

    def setMaxValue(self,max):
        self.maxValue = max

    def setMinValue(self,min):
        self.minValue = min


    def setValue(self, val):
        self.value = val

    # def resetFired(self):
    #     self.negativeNeuron.resetFired()
    #     self.positiveNeuron.resetFired()

    #Was named sync in the original paper (for I&F)
    def feedNN(self):
        if self.type == 'IN':
            if self.value>=self.valleyVal:
                corVal = self.value/self.maxValue
                posPot = (self.maxState-self.minState)*corVal+self.minState
                self.positiveNeuron.setInternalState(posPot)
                self.positiveNeuron.setOutputState(posPot)
                self.negativeNeuron.setInternalState(self.minState)
                self.negativeNeuron.setOutputState(self.minState)
                #print (self.value,posPot,self.minPotencial)
            else:
                corVal = self.value/(-self.minValue)
                negPot = (self.maxState-self.minState)*(-corVal)+self.minState
                self.positiveNeuron.setInternalState(self.minState)
                self.positiveNeuron.setOutputState(self.minState)
                self.negativeNeuron.setInternalState(negPot)
                self.negativeNeuron.setOutputState(negPot)
                #print(self.value, self.minPotencial,negPot)

    def getFeedBackNN(self):
        if self.type == 'OUT':
            #se podria probar tambien con el output state en lugar del internal
            negSt = self.negativeNeuron.getInternalState()
            posSt = self.positiveNeuron.getInternalState()
            retVal1 = self.bounded_affine(self.minState, 0, self.maxState, self.maxValue, posSt)
            retVal2 = self.bounded_affine(self.minState, 0, self.maxState, -self.minValue, negSt)
            #print ('retValPos:%s,retValNeg:%s=%s, pot[%s,%s]'%(retVal1,retVal2,retVal1-retVal2,posPot,negPot))
            return retVal1-retVal2

    def bounded_affine(self, xmin, ymin, xmax, ymax, x):
        a = (ymax - ymin) / (xmax - xmin)
        d = ymin - a * xmin
        y = a * x + d
        if y > ymax:
            y = ymax
        elif y < ymin:
            y = ymin
        return y



    # def clone(self):
    #     toRet=BinaryInterface(self.name,self.positiveNeuron,self.negativeNeuron,self.type)
    #     toRet.valleyVal=self.valleyVal
    #     toRet.value=self.value
    #     toRet.minValue=self.minValue
    #     toRet.maxValue=self.maxValue
    #     return toRet

    def isSameAs(self,interface2):
        isSame=True
        isSame=isSame and (self.value==interface2.value)
        isSame=isSame and (self.minValue==interface2.minValue)
        isSame=isSame and (self.maxValue==interface2.maxValue)
        isSame=isSame and (self.valleyVal==interface2.valleyVal)
        return isSame

def loadConnection(xmlMI,nn):
    interfacesToReturn={}
    for child in xmlMI:
        miToReturn = BinaryInterface(child.attrib['name'],nn.getNeuron(child.attrib['positiveNeuron']),nn.getNeuron(child.attrib['negativeNeuron']),child.attrib['type'])
        miToReturn.valleyVal=float(child.attrib['valleyVal'])
        miToReturn.value=float(child.attrib['value'])
        miToReturn.minValue=float(child.attrib['minVal'])
        miToReturn.maxValue=float(child.attrib['maxVal'])
        miToReturn.minState=float(child.attrib['minState'])
        miToReturn.maxState=float(child.attrib['maxState'])
        interfacesToReturn[miToReturn.name]=miToReturn
    return interfacesToReturn

####################################################
#interfaces for openGym Environment

def envObsToModelObs(envObs):
    observations = []
    for i in range(0, 2):
        observations.append(float(0))
    observations[0] = float(envObs[0])
    observations[1] = float(envObs[1])
    return observations

def modActionToEnvAction(modAction):
    actions = np.zeros(1)
    actions[0] = modAction
    return actions

#interfaces for HyperOpt
def getSpaceForHyperOpt(model):
    return model.neuralnetwork.getSpaceForHyperOpt()


def putVariablesFromHyperOpt(model,vars):
    #uncomment when to learn neuron parameters is needed
    for neuName in model.neuralnetwork.getNeuronNames():
        deltaTH=vars[neuName+'_tth']
        model.neuralnetwork.setNeuronTestThresholdOfName(neuName, deltaTH)
        deltaDF=vars[neuName+'_tdf']
        model.neuralnetwork.setNeuronTestDecayFactorOfName(neuName, deltaDF)
    for idx in range(0,len(model.neuralnetwork.connections)):
        deltaWeight=vars['con_'+str(idx)+'_w']
        model.neuralnetwork.setConnectionTestWeightOfIdx(idx, deltaWeight)


#Interfaces for PyGad


def getNumGenes():
    #only connections
    #return 52
    return 96
    #connections and neurons

def getGeneType():
    return float

def getGeneSpace():
    #for connections and neurons uncomment the commented lines
    return [
        {'low': 0.0, 'high': 1.0}, {'low': 0.00, 'high': 1.00},
        {'low': 0.0, 'high': 1.0}, {'low': 0.0, 'high': 1.0},
        {'low': 0.0, 'high': 1.0}, {'low': 0.0, 'high': 1.0},
        {'low': 0.0, 'high': 1.0}, {'low': 0.0, 'high': 1.0},
        {'low': 0.0, 'high': 1.0}, {'low': 0.0, 'high': 1.0},
        {'low': 0.0, 'high': 1.0}, {'low': 0.0, 'high': 1.0},
        {'low': 0.0, 'high': 1.0}, {'low': 0.0, 'high': 1.0},
        {'low': 0.0, 'high': 1.0}, {'low': 0.0, 'high': 1.0},
        {'low': 0.0, 'high': 1.0}, {'low': 0.0, 'high': 1.0},
        {'low': 0.0, 'high': 1.0}, {'low': 0.0, 'high': 1.0},
        {'low': 0.0, 'high': 1.0}, {'low': 0.0, 'high': 1.0},
        {'low': 0.0, 'high': 0.5},
        {'low': 0.0, 'high': 0.5},
        {'low': 0.0, 'high': 0.5},
        {'low': 0.0, 'high': 0.5},
        {'low': 0.0, 'high': 0.5},
        {'low': 0.0, 'high': 0.5},
        {'low': 0.0, 'high': 0.5},
        {'low': 0.0, 'high': 0.5},
        {'low': 0.0, 'high': 0.5},
        {'low': 0.0, 'high': 0.5},
        {'low': 0.0, 'high': 0.5},
        {'low': 0.0, 'high': 0.5},
        {'low': 0.0, 'high': 0.5},
        {'low': 0.0, 'high': 0.5},
        {'low': 0.0, 'high': 0.5},
        {'low': 0.0, 'high': 0.5},
        {'low': 0.0, 'high': 0.5},
        {'low': 0.0, 'high': 0.5},
        {'low': 0.0, 'high': 0.5},
        {'low': 0.0, 'high': 0.5},
        {'low': 0.0, 'high': 0.5},
        {'low': 0.0, 'high': 0.5},
        {'low': 0.0, 'high': 0.5},
        {'low': 0.0, 'high': 0.5},
        {'low': 0.0, 'high': 0.5},
        {'low': 0.0, 'high': 0.5},
            ]


def getIndividualForPYGAD(model):
    return model.neuralnetwork.getIndividualForPYGAD()


def putIndividualFromPYGAD(model,indiv):
    #for connections only
    #putIndividualFromPYGAD_OnlyConnMode(model, indiv)
    #for connections and neurons
    putIndividualFromPYGAD_ConnNeuMode(model,indiv)


def putIndividualFromPYGAD_OnlyConnMode(model,indiv):
    index = 0
    neuronRandomIndexedNames = model.getNeuronRandomIndexedNames()
    for i in range(0, len(indiv)):
        if i in [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48,
                 50]:
            weight = indiv[i]
            model.neuralnetwork.setConnectionTestWeightOfIdx(index, weight)
        if i in [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47, 49,
                 51]:
            sigma = indiv[i]
            model.neuralnetwork.setConnectionTestSigmaOfIdx(index, sigma)
            index = index + 1

def putIndividualFromPYGAD_ConnNeuMode(model, indiv):
    index = 0
    neuronRandomIndexedNames = model.getNeuronRandomIndexedNames()
    for i in range(0, len(indiv)):
        if i in [0, 2, 4, 6, 8, 10, 12, 14, 15, 18, 20]:
            neuronName = neuronRandomIndexedNames[index]
            paramTH = indiv[i]
            model.neuralnetwork.setNeuronTestThresholdOfName(neuronName, paramTH)
        if i in [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21]:
            neuronName = neuronRandomIndexedNames[index]
            paramDF = indiv[i]
            model.neuralnetwork.setNeuronTestDecayFactorOfName(neuronName, paramDF)
            index = index + 1
        if i in [22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46,
                 47]:
            weight = indiv[i]
            model.neuralnetwork.setConnectionTestWeightOfIdx(index - 11, weight)
            index = index + 1


def randonizeModel(model):
    model.addNoise('Weight', 1.0, 26)
    model.addNoise('TH', 0.0, 11)
    model.addNoise('DF', 0.0, 11)

#Interfaces for Random Seek


def setInitialDistortions():
        global distortions
        distortions['weight']=15
        distortions['th']=8
        distortions['df']=8

def setInitialVariance():
        global variances
        variances['weight']=1.0
        variances['th'] = 0.49
        variances['df'] = 0.025



def setBaseDistortions():
    global distortions
    distortions['weight']=6
    distortions['th'] = 6
    distortions['df'] = 6


def setStepVariance():
        global variances
        variances['weight']=np.random.uniform(0.01, 0.8)
        variances['th']=np.random.uniform(0.05, 0.49)
        variances['df']=np.random.uniform(0.002, 0.025)

def setDecresedDistortions():
    global distortions
    if (distortions['weight'] > 4):
        distortions['weight'] -= 1
    if (distortions['th'] > 2):
        distortions['th'] -= 1
    if (distortions['df'] > 2):
        distortions['df'] -= 1




def setIncresedDistortions():
    global distortions
    if (distortions['weight'] < 13):
        distortions['weight'] += 1
    if (distortions['th'] < 6):
        distortions['th'] += 1
    if (distortions['df'] < 6):
        distortions['df'] += 1




def randonize(model):
    global distortions
    global variances
    model.addNoise('Weight', variances['weight'], distortions['weight'])
    model.addNoise('TH', variances['th'], distortions['th'])
    model.addNoise('DF', variances['df'], distortions['df'])

def randozieAll(model):
    model.addNoise('Weight', 1, 26)
    model.addNoise('TH', 0.49, 11)
    model.addNoise('DF', 0.025, 11)

