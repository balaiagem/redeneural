from os import close
from pybrain3.tools.shortcuts import buildNetwork
from pybrain3.datasets import SupervisedDataSet
from pybrain3.supervised.trainers import BackpropTrainer
from pybrain3.structure.modules import SoftmaxLayer
from pybrain3.structure.modules import SigmoidLayer


rede = buildNetwork(6,100,100,1)
base = SupervisedDataSet(6,1)

arquivo = open('base.txt','r')
arquivo.seek(0,0)
for linha in arquivo.readlines():
    print(linha)
    l = [float(x) for x in linha.strip().split(',') if x!='']
    indata = tuple(l[:6])
    outdata = tuple(l[6:])
    print(indata)
    print(outdata)
    base.addSample(indata,outdata)


treinamento = BackpropTrainer(rede,dataset=base,learningrate=0.01,momentum=0.3)

for i in range(1,3000):
    erro = treinamento.train()
    if i % 1000 == 0:
        print("erro %s" %erro)

print(rede.activate([20,20,20,20,20,20]))