from . import Neuron as neu
from . import Connection as con
import matplotlib.pyplot as plt

class NeuralNetwork:

    def __init__(self, name):
        self.name = name
        self.neurons = {}
        self.connections = []

    def __str__(self):
        return "[nombre: %s, neuronas: %s, conexiones: %s]" % (
        self.name, " ".join(str(c) for c in self.neurons.values())," ".join(str(c) for c in self.connections))
        # return "[neu: %s, estado: %s, conexiones: %s]" % (self.neuronName,self.neuronState,len(self.neuronConnections))

    def __repr__(self):
        return "[nombre: %s,  neuronas: %s,conexiones: %s]" % (
        self.name, self.countNeurons(), self.countConnections)

    def updateStateToTrace(self):
        for neu in self.neurons.values():
            neu.updateStateToTrace()
        for con in self.connections:
            con.updateStateToTrace()

    def graphVariableTraces(self,folder):
        for neu in self.neurons.values():
            neu.graphVariableTraces(folder)
        for con in self.connections:
            con.graphVariableTraces(folder)


    def getNeuron(self, nname):
         return self.neurons[nname]
    def getNeurons(self):
         return self.neurons.values()
    def getNeuronNames(self):
         return self.neurons.keys()

    # def getNeuronStateOfName(self,name):
    #     return self.neurons[name].getState()
    def resetActivationTraces(self):
        for neu in self.neurons.values():
            neu.resetLastActivationTrace()

    def recordActivationTraces(self):
        for neu in self.neurons.values():
            neu.recordActivationTrace()

    def resetAllNeurons(self):
        for n in self.neurons.values():
            n.setInternalState(0.0)
            #n.setOutputState(0.0)

    def countNeurons(self):
        return len(self.neurons)


    def countConnections(self, type=None):
        count = 0
        if type==None:
            count=len(self.connections)
        else:
            for conn in self.connections:
                if conn.isType(type):
                    count = count + 1
        return count



    def doSimulationStep(self):
        for neu in self.neurons.values():
            neu.computeVnext(self.getDendriticConnectionsFor(neu))
        for neu in self.neurons.values():
             neu.commitComputation()
        #printNeurons([])

    def printNeurons(self,nlist):
        print('-----------------------------------------------------')
        result=''
        if len(nlist)==0:
            for neu in self.getNeurons():
                result=result+neu.getName()+':'+str(self.neuralnetwork.getNeuronStateOfName(neu.getName()))+','
        else:
            for nename in nlist:
                print(nename+':'+str(self.neuralnetwork.getNeuronStateOfName(nename)))
        print(result)


    def getDendriticConnectionsFor(self,targetNeuron):
        dendriticConnections = [c for c in self.connections if c.getTarget() == targetNeuron]
        return dendriticConnections

    def getConnectionIdx(self,idx):
        return self.connections[idx]
    def getConnections(self):
        return self.connections


    def getConnectionTestWeightOfIdx(self,idx):
        return self.connections[idx].getTestWeight()

    def setConnectionTestWeightOfIdx(self,idx,val):
        return self.connections[idx].setTestWeight(val)



    def getNeuronTestThresholdOfName(self,name):
        return self.neurons[name].getTestThreshold()
    def setNeuronTestThresholdOfName(self,name,val):
        return self.neurons[name].setTestThreshold(val)


    def getNeuronTestDecayFactorOfName(self,name):
        return self.neurons[name].getTestDecayFactor()
    def setNeuronTestDecayFactorOfName(self,name,val):
        return self.neurons[name].setTestDecayFactor(val)


    def resetActivationTraces(self):
        for neu in self.neurons.values():
            neu.resetLastActivationTrace()

    def writeToGraphs(self,folder):
        times=[]
        values=[]
        for neu in self.neurons.values():
            values=neu.getLastActivaionTrace()
            neu.resetLastActivationTrace()
            times=[i for i in range(0, len(values))]
            fig=plt.figure()
            plt.plot(times, values, 'ro', label='state')
            plt.ylabel('State')
            plt.xlabel('Episode step')
            plt.legend()
            plt.title('Neural State for %s'%(neu.getName()))
            fig.savefig( folder+'/'+neu.getName()+'.png', bbox_inches='tight')
        #logfile.write("CONNECTIONS"+'\n')
        #for con in self.connections:
        #    logfile.write(str(con) + '\n')




    def getConnectionSize(self):
        return len(self.connections)

    def getNeuronsSize(self):
        return len(self.neurons)

    def getInternalStateOfName(self,name):
        return self.neurons[name].getInternalState()


    def writeToFile(self,logfile):
        logfile.write("NEURONS"+'\n')
        for neu in self.neurons.values():
            logfile.write(str(neu) + '\n')
        logfile.write("CONNECTIONS"+'\n')
        for con in self.connections:
            logfile.write(str(con) + '\n')


    #needed for PyGAD
    def getIndividualForPYGAD(self):
        currList = []
        for neu in self.neurons.values():
            for nc in neu.getComponentOfIndividualForPYGAD():
                currList.append(nc)
        for con in self.connections:
            for nc in con.getComponentOfIndividualForPYGAD():
                currList.append(nc)
        return currList

    #needed for HyperOpt
    def getSpaceForHyperOpt(self):
        currDic={}
        varDic = {}
        for neu in self.neurons.values():
            varDic=neu.getVariablesForHyperOpt()
            for key,value in varDic.items():
                currDic[key]=value
        for index in range(0,len(self.connections)):
            con=self.connections[index]
            varDic=con.getSpaceForHyperOpt(index)
            for key,value in varDic.items():
                currDic[key]=value
        return currDic

    # needed for Random Seek

    def commitNoise(self):
        for neu in self.neurons.values():
            neu.commitNoise()
        for con in self.connections:
            con.commitNoise()

    def revertNoise(self):
        for neu in self.neurons.values():
            neu.revertNoise()
        for con in self.connections:
            con.revertNoise()

    def isSameAs(self,neuralNetwork2):
        sameNeurons=True
        sameConnections=True
        for key in self.neurons:
            sameNeurons=sameNeurons or (self.neurons[key]).isSameAs(neuralNetwork2.neurons[key])
        for index  in range(0,len(self.connections)):
            sameConnections = sameConnections or (self.connections[index]).isSameAs(neuralNetwork2.connections[index])
        return [sameNeurons,sameConnections]



def loadNeuralNetwork(xmlNn):
     nnToReturn = NeuralNetwork(xmlNn.attrib['name'])
     #namesForRandomIndexes = []
     for child in xmlNn:
         if child.tag == 'neuron':
             n = neu.loadNeuron(child)
             nnToReturn.neurons[n.getName()] = n
             #namesForRandomIndexes.append(n.getName())
         if child.tag == 'connection':
              c = con.loadConnection(child,nnToReturn)
              nnToReturn.connections.append(c)
     return nnToReturn

