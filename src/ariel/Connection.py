import enum
from hyperopt import hp


class Connection:

    def __init__(self, contype, source, target, weight):
        self.connType = contype
        self.connSource = source
        self.connTarget = target
        self.connWeight = weight
        self.testWeight = weight
        self.stateToTrace={'Weight':[]}

    def __str__(self):
        return "<Con from:%s, To: %s, W: %s, TestW: %s> \n" % (self.connSource.getName(), self.connTarget.getName(), self.connWeight, self.testWeight)

    def __repr__(self):
        return "<Con from:%s, To: %s, W: %s, TestW: %s>" % (self.connSource.getName(), self.connTarget.getName(), self.connWeight, self.testWeight)

    def setWeight(self, w):
        self.connWeight = w

    def getWeight(self):
        return self.connWeight

    def setTestWeight(self, w):
        self.testWeight = w

    def getTestWeight(self):
        return self.testWeight

    def isType(self, type):
        return type == self.connType

    def updateStateToTrace(self):
        self.stateToTrace['Weight'].append(self.testWeight)
        #print('passed by the conection's updateStateToTrace with values for weight %s' %(self.stateToTrace['Weight']))

    def getSource(self):
        return self.connSource

    def setSource(self,src):
        self.connSource=src

    def getTarget(self):
        return self.connTarget
    def setTarget(self,tar):
        self.connTarget=tar



    def isSameAs(self,con2):
        return self.testWeight==con2.testWeight

#needed for PyGAD
    def getComponentOfIndividualForPYGAD(self):
        return [self.testWeight]

#needed for HyperOpt
    def getSpaceForHyperOpt(self,index):
        namew='con_'+str(index)+'_w'
        return {
            namew: hp.uniform(namew, 0.0, 1.0),
        }

#needed for Random Seek
    def commitNoise(self):
        self.connWeight = self.testWeight


    def revertNoise(self):
        self.testWeight = self.connWeight



def loadConnection(xmlCon,nn):
    connToReturn = Connection(connTypeFromStr(xmlCon.attrib['connType']),nn.getNeuron(xmlCon.attrib['connSource']),nn.getNeuron(xmlCon.attrib['connTarget']),float(xmlCon.attrib['connWeight']))
    connToReturn.testWeight=float(xmlCon.attrib['testWeight'])
    return connToReturn




class ConnectionType(enum.Enum):
    ChemIn = 1
    ChemEx = 2
    SGJ = 3
    AGJ = 4

def connTypeFromStr(str):
    if str=='ConnectionType.ChemIn':
        return ConnectionType.ChemIn
    if str=='ConnectionType.ChemEx':
        return ConnectionType.ChemEx
    if str=='ConnectionType.AGJ':
        return ConnectionType.AGJ
    if str=='ConnectionType.SGJ':
        return ConnectionType.SGJ
