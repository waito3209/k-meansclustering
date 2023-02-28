import matplotlib.pyplot as plt
import numpy as np
import math
import copy
import time
import random
def kmean(data:np.array,g,expode:int,viewintervial:int,initrangea,initrangeb,path = 'data',stoptime = 2):
    #([x1,x2,x3..])
    no_data = np.shape(data)[0]
    no_feature= np.shape(data)[1]
    print(f"no feature : {no_feature}")
    print(f"no data    : {no_data}")

    def init(x_):
        t = []
        for i in range(no_feature):
            t.append(random.randint(initrangea,initrangeb))
        return t

    def cal(x_,y_):
        t = 0
        for k in range(no_feature):
            t+=(x_[k]-y_[k])**2
        return math.sqrt(t)
    output = [ init(x) for x in range(g)]
    initvalue = copy.deepcopy(output)
    print(output)
    data_class = np.full((no_data,),-1)

    for i in range(no_data):
        temp = [ cal(output[x],data[i]) for x in range(g)]
        data_class[i] = np.argmin(np.asarray(temp))
    def plot(data:np.array,label:list,no_class:int,show:bool):
        assert np.shape(data)[0]==len(label)
        result= []
        for m in range(no_class):
            result.append([])
        for n in range(len(label)):
            result[label[n]].append(data[n])
        for nn in result:
            nnt=np.asarray(nn)
            try:
                x = nnt[:, 0]
                y = nnt[:, 1]
                if show:
                    plt.scatter(x, y)
            except:
                pass
        if show:
            plt.xlim([initrangea,initrangeb])
            plt.ylim([initrangea, initrangeb])
            ax = plt.gca()
            ax.set_aspect('equal', adjustable='box')
            plt.show(block=False)
            plt.pause(stoptime)
            plt.close()
            print()
        else:
            print('>',end='')
        return result
    #plot(data,data_class,g)

    for e in range(expode):
        for i in range(no_data):
            temp = [cal(output[x], data[i]) for x in range(g)]
            data_class[i] = np.argmin(np.asarray(temp))
        normtabel=plot(data,data_class,g,show=e%viewintervial==0)

        for nn in range(len(normtabel)):
            nnt = np.asarray(normtabel[nn])
            for f in range(no_feature):
                output[nn][f] = np.average(np.asarray(nnt[:, f]))



    # print final picture
    result = []
    for m in range(g):
        result.append([])
    for n in range(len(data_class)):
        result[data_class[n]].append(data[n])
    for nn in result:
        nnt = np.asarray(nn)
        x = nnt[:, 0]
        y = nnt[:, 1]

        plt.scatter(x, y)
    maxleng=[]
    for i in range(g):
        maxleng.append(0)
    for i in range(len(data)):
        tem = math.sqrt(sum([(data[i,x]-output[data_class[i]][x])**2 for x in range(no_feature)]))
        maxleng[data_class[i]]= tem  if tem   > maxleng[data_class[i]] else maxleng[data_class[i]]






    initvalue = np.asarray(initvalue)
    output = np.asarray(output)
    plt.scatter(initvalue[:,0], initvalue[:,1], marker='o',c='b',s=40)
    plt.scatter(output[:, 0], output[:, 1], marker='x',c='r',s=40)
    plt.xlim([initrangea, initrangeb])
    plt.ylim([initrangea, initrangeb])
    print(max(maxleng))
    plt.title(f"K-mean clust max :{max(maxleng)}")
    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')
    plt.savefig(f'{path}/{time.strftime("%Y%m%d-%H%M%S")}')
    return output




