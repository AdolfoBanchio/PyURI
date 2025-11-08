from . import NeuralNetwork as nN
from . import ModelInterfaces as mi
import numpy as np
import xml.etree.ElementTree as ET
from xml.dom import minidom
from . import Environment as eut

#aca iran los AddNoise, el update y capaz que aLGUNA OTRA COSA
np.random.seed(423)
neuronRandomIndexedNames = []



class Model:
    def __init__(self,name):
        self.name = name
        self.neuralnetwork = None
        self.interfaces = {}
        self.generationGraphModeFolder = ''
        self.generationGraphModeCounter=0

    def setRandomIndexesList(self,list):
        global neuronRandomIndexedNames
        neuronRandomIndexedNames=list

    def load(self,fromFile=''):
        if fromFile=='':
            print('the harcoded loading functionality was removed.')
        else:
            self.loadFromFile(fromFile)

    def __str__(self):
        return """
        RED:
        %s
        INTERFACES:
        %s
        """ % (self.neuralnetwork," ".join(str(i) for i in self.interfaces.values()))
    def __repr__(self):
        return """RED:
            %s
        INTERFACES:
            %s
            """ % (self.neuralnetwork, " ".join(str(i) for i in self.interfaces))


    def setGenerationGraphModeFolderOn(self,folder):
        self.generationGraphModeFolder=folder
        #same ammount of steps so the generation graph is self reseting
        self.generationGraphModeCounter=10

    def setGenerationGraphModeFolderOff(self):
        self.generationGraphModeFolder= ''

    def updateStateToTrace(self):
        return self.neuralnetwork.updateStateToTrace()

    def graphVariableTraces(self,folder):
        self.neuralnetwork.graphVariableTraces(folder)


    def getInterface(self,name):
        return self.interfaces[name]
    def getInterfaces(self):
        return self.interfaces.values()

    def getNeuron(self,name):
        return self.neuralnetwork.getNeuron(name)

    def getName(self):
        return self.name

    def setName(self,name):
        self.name=name

#Reset
    def Reset(self):
        self.neuralnetwork.resetAllNeurons()
        for intf in self.interfaces.values():
            intf.reset()

    def runPulse(self):
        simmulationSteps = 2
        for step in range(0, simmulationSteps):
            randomX, randomV = eut.getRandomObservation("MouCarCon")
            self.interfaces['IN1'].setValue(randomX)
            self.interfaces['IN2'].setValue(randomV)
            self.interfaces['IN1'].feedNN()
            self.interfaces['IN2'].feedNN()
            self.neuralnetwork.doSimulationStep()
            ret = self.interfaces['OUT1'].getFeedBackNN()
        #print("x:%s, v:%s, a:%s"%(randomX,randomV,ret))



#Update(observations,0.01,10)..
#Update(observations,deltaT,simmulationSteps):
#mas que un update esto es un runConnectome ya que toma las entradas (observaciones)
# y corre el connectoma con estas entradas a un paso de deltaT por simmulationSteps pasos
#se mantiene el nombre mientras se use el mismo codigo del autor del paper
    def Update(self,observations,mode=None,doLog=False):
        #deltaT=0.1
        #self.printNeurons([])
        ret=0
        simmulationSteps = 2
        if doLog:
            networkLog = open('log/oneEpisodeModel.log', 'a')
            networkLog.write("----Update(%s,%s,%s)" % (observations,simmulationSteps) + '\n')
            #networkLog.write('\t' + "Pre state" + '\n')
        """ if self.generationGraphModeCounter != 0:
             gut.graphSnapshot('FI',self, 0, True) """
        for step in range(0,simmulationSteps-1):
            #seteo valores de las entradas
            self.interfaces['IN1'].setValue(observations[0])
            self.interfaces['IN2'].setValue(observations[1])
            #transformo y cargo las variables x y v a la red
#            self.printNeurons(['AVM','AVD','AVA','ALM'])
            #print(observations)
            self.interfaces['IN1'].feedNN()
            self.interfaces['IN2'].feedNN()
            #print('entries for step: '+str(step))
            #self.printNeurons([])
            # hago DoSimulationStep(deltaT)
            #if doLog:
                #self.neuralnetwork.writeToFile(networkLog)
            #gut.graphSnapshot('FI', self, step+100 , True)
            self.neuralnetwork.doSimulationStep()
            #gut.graphSnapshot('FI', self, step + 101, True)
            #print('after dusimulation: ')
            #self.printNeurons([])
            # recupero el valor de output
            ret = ret + self.interfaces['OUT1'].getFeedBackNN()
            """ if self.generationGraphModeCounter!=0:
                #this generations is in extended mode, so voltages will be in node labels
                gut.graphSnapshot('FI',self,step+1,True)
                self.generationGraphModeCounter=self.generationGraphModeCounter-1
             """
            if doLog:
                #self.neuralnetwork.writeToFile(networkLog)
                networkLog.write('\t' + "Return:%s"%(ret) + '\n')
                networkLog.write('\t' + "------------------------------------" + '\n')
                networkLog.flush()
        if mode == 'debug':
            self.neuralnetwork.recordActivationTraces()
        #return el valor de output
        return ret

    #to debug on haimovici
    def printNeurons(self,nlist):
        print('-----------------------------------------------------')
        result=''
        if len(nlist)==0:
            for neu in self.neuralnetwork.getNeurons():
                result=result+neu.getName()+':'+str(self.neuralnetwork.getNeuronStateOfName(neu.getName()))+','
        else:
            for nename in nlist:
                print(nename+':'+str(self.neuralnetwork.getNeuronStateOfName(nename)))
        print(result)

    def writeToFile(self,logfile):
        logfile.write("NEURAL NETWORK" + '\n')
        self.neuralnetwork.writeToFile(logfile)
        logfile.write("INTERFACES" + '\n')
        for inter in self.interfaces.values():
            logfile.write(str(inter) + '\n')

    def isSameAs(self,model2):
        return self.neuralnetwork.isSameAs(model2.neuralnetwork)



    def loadFromFile(self,fromFile):
        #neuronRandomIndexedNames=[]
        mydoc = ET.parse(fromFile)
        xmlRootModelElement = mydoc.getroot()
        for child in xmlRootModelElement:
            if child.tag == 'NeuralNetwork':
               loaded =  nN.loadNeuralNetwork(child)
               self.neuralnetwork = loaded
               #neuronRandomIndexedNames=loaded[1]
               for n in self.neuralnetwork.getNeurons():
                   neuronRandomIndexedNames.append(n.getName())
            if child.tag == 'Interfaces':
               inter = mi.loadConnection(child,self.neuralnetwork)
               self.interfaces=inter


    def getModelInterface(self):
        return mi

    def getNeuronRandomIndexedNames(self):
        global neuronRandomIndexedNames
        return neuronRandomIndexedNames


#needed for RandomSeek

    # CommitNoise()
    def commitNoise(self):
        #se comitea sobre todas, capaz que para hacerlo mas rapido hay que comitear sobre las que fueron modif nomas
        self.neuralnetwork.commitNoise()

    # UndoNoise
    def revertNoise(self):
        #se comitea sobre todas, capaz que para hacerlo mas rapido hay que comitear sobre las que fueron modif nomas
        self.neuralnetwork.revertNoise()


    def addNoise(self,parameter,varianza,samplesAmmount):
      #try:
        mu, sigma = 0, varianza
        #uniformDistribuitedValues=None
        #normalDistribuitedValues=None
        if (parameter == 'TH'):
            normalDistribuitedValues = np.random.uniform(0.0, 1.0, samplesAmmount)
            #normalDistribuitedValues = np.random.normal(mu, sigma, samplesAmmount)
            uniformDistribuitedValues = np.random.randint(0, self.neuralnetwork.getNeuronsSize(), samplesAmmount)
            # print('sigma:%s'%(sigma))
            #print(normalDistribuitedValues)
        elif parameter == 'DF':
            normalDistribuitedValues = np.random.uniform(0.0, 0.5, samplesAmmount)
            #normalDistribuitedValues = np.random.normal(mu, sigma, samplesAmmount)
            uniformDistribuitedValues = np.random.randint(0, self.neuralnetwork.getNeuronsSize(), samplesAmmount)
            #print('inside the addNoise %s for vleak: %s' % (samplesAmmount,uniformDistribuitedValues))
        elif parameter == 'Weight':
            normalDistribuitedValues = np.random.uniform(0.0, 1.0, samplesAmmount)
            #normalDistribuitedValues = np.random.normal(mu, sigma, samplesAmmount)
            uniformDistribuitedValues = np.random.randint(0, self.neuralnetwork.getConnectionSize(), samplesAmmount)
            #print('inside the addNoise %s for weight: %s'%(samplesAmmount,uniformDistribuitedValues))
        self.noiseParam(samplesAmmount,parameter,uniformDistribuitedValues,normalDistribuitedValues)
      #except:
      #    print(parameter)



    def noiseParam(self, samplesAmmount, parameter, whiches, values):
        for i in range(0, samplesAmmount):
            which = whiches[i]
            x = values[i]
            if parameter == 'Weight':
                #print("conection index %s: " %(self.neuralnetwork.getConnectionIdx(which)))
                #newValue = self.neuralnetwork.getConnectionTestWeightOfIdx(which) + x
                newValue=x
                if newValue < 0:
                    newValue = 0
                elif newValue > 10.0:
                    newValue = 10.0
                self.neuralnetwork.setConnectionTestWeightOfIdx(which, newValue)
                #print("conection index %s: " %(self.neuralnetwork.getConnectionIdx(which)))
            elif parameter == 'TH':
                #print("neuron index %s: " %(self.neuralnetwork.getNeuron(neuronRandomIndexedNames[which])))
                #newValue = self.neuralnetwork.getNeuronTestThresholdOfName(neuronRandomIndexedNames[which]) + x
                newValue = x
                if newValue < 0.0:
                    newValue = 0.0
                elif newValue > 10.0:
                    newValue = 10.0
                self.neuralnetwork.setNeuronTestThresholdOfName(neuronRandomIndexedNames[which],newValue)
                #print("neuron index %s: " %(self.neuralnetwork.getNeuron(neuronRandomIndexedNames[which])))
            elif parameter == 'DF':
                #print("neurona indice %s: " %(self.neuralnetwork.getNeuron(neuronRandomIndexedNames[which])))
                #newValue = self.neuralnetwork.getNeuronTestR1OfName(neuronRandomIndexedNames[which]) + x
                newValue = x
                if newValue < 0.0:
                    newValue = 0.0
                elif newValue > 10.0:
                    newValue = 10.0
                self.neuralnetwork.setNeuronTestDecayFactorOfName(neuronRandomIndexedNames[which], newValue)
                #print("neuron index %s: " %(self.neuralnetwork.getNeuron(neuronRandomIndexedNames[which])))







