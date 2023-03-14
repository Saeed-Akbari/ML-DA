""" @author: Saeed  """

import os, os.path

import matplotlib as matplt
import matplotlib.pyplot as plt
from matplotlib import animation

import numpy as np

import random
random.seed(10)

def contourPlot(X, Y, phi, barRange, fileName='filename', figSize=(14,7)):

    fig, axs = plt.subplots(1,1,figsize=figSize)

    axs.contour(X, Y, phi, barRange, linewidths=0.5, colors='k')
    cntr1 = axs.contourf(X, Y, phi, barRange, cmap="RdBu_r")
    cb = fig.colorbar(cntr1, ax=axs, shrink=0.8, orientation='vertical')
    axs.set_aspect('equal')
    fig.tight_layout()
    fig.savefig(fileName, bbox_inches = 'tight', pad_inches = 0.1, dpi = 400)

    fig.clear(True)

def contourSubPlot(X, Y, phi1, phi2, phi3, phi4, phi5, phi6, phi7, phi8, time, figTimeTest,\
                    testStartTime, barRange, fileName='filename', figSize=(14,7)):

    #if ra < ra_max:
    #    cs = axs[2].contour(X,Y,th,20,vmin=-0, vmax=1,colors='black')
    #cs = axs[2].imshow(th.T,extent=[0, 2, 0, 1], origin='lower',
    #        interpolation='bicubic',cmap='RdBu_r', alpha=1.0)

    fig, axs = plt.subplots(4,2,figsize=figSize)
    phi = [phi1, phi2, phi3, phi4, phi5, phi6, phi7, phi8]
    i = 0
    for ax in axs.flat:
        #ax.contour(X, Y, phi[i], barRange, linewidths=0.5, colors='k')
        cntr1 = ax.contourf(X, Y, phi[i], barRange, cmap="RdBu_r")
        #cs = ax.contour(X,Y,phi[i],20,vmin=-0, vmax=1,colors='black')
        #cs = ax.imshow(phi[i].T,extent=[0, 2, 0, 1], origin='lower',
        #        interpolation='bicubic',cmap='RdBu_r', alpha=1.0)
        ax.set_aspect('equal')
        if i%2==0:
            ax.set_ylabel('$y$')
        if i-len(axs.flat) > -3:
            ax.set_xlabel('$x$')
        i = i + 1
    cb1 = fig.colorbar(cntr1, ax=axs, shrink=0.8, orientation='vertical')

    axs[0][0].set_title(r'$t={:.0f}$'.format(time[figTimeTest[0]+testStartTime]))
    axs[0][1].set_title(r'$t={:.0f}$'.format(time[figTimeTest[1]+testStartTime]))
    #fig.colorbar(cs, ax=axs, shrink=0.8, orientation='vertical')
    
    plt.text(-3.3, 4.35, r'\bf{FOM}', va='center',fontsize=16)
    plt.text(-3.15, 3.08, r'\bf{TP}', va='center',fontsize=16)
    #plt.text(-11.5, 4.85, r'\bf{($r=2$)}', va='center',fontsize=18)
    plt.text(-3.15, 1.78, r'\bf{ML}', va='center',fontsize=16)
    plt.text(-3.45, 0.5, r'\bf{ML-DA}', va='center',fontsize=16)

    fig.tight_layout()
    #fig.subplots_adjust(wspace=0, hspace=0)
    fig.savefig(fileName, bbox_inches = 'tight', pad_inches = 0.1, dpi = 400)
    fig.clear(True)

def animationGif(X, Y, phi, fileName='filename', figSize=(14,7)):

    fig = plt.figure(figsize=figSize)

    plt.xticks([])
    plt.yticks([])
    
    def animate(i):
        cont = plt.contourf(X,Y,phi[:,:,i],120,cmap='jet')
        return cont  
    
    anim = animation.FuncAnimation(fig, animate, frames=50)
    fig.tight_layout()
    # anim.save('animation.mp4')
    writergif = animation.PillowWriter(fps=10)
    anim.save(fileName+'.gif',writer=writergif)

    fig.clear(True)

def plotPODcontent(RICd, AEmode, dirPlot, myCont):

    fig = plt.figure()
        
    nrplot = 2 * np.min(np.argwhere(RICd>myCont))
    index = np.arange(1,nrplot+1)
    newRICd = [0, *RICd]
    plt.plot(range(len(newRICd)), newRICd, 'k')
    x1 = 1
    x2 = AEmode
    y1 = RICd[x2-1]
    y2 = RICd[x2-1]
    plt.plot([x1,x2], [y1,y2], color='k', linestyle='--')
    xave = np.exp(0.75*np.log(x1) + 0.25*np.log(x2))
    plt.text(xave, y2+0.4, r'$'+str(np.round(RICd[x2-1],decimals=2))+'\%$',fontsize=10)
    plt.fill_between(index[:x2], RICd[:x2], RICd[0],alpha=0.6,color='orange')

    x1 = AEmode
    x2 = AEmode
    y1 = RICd[0]
    y2 = RICd[x2-1]
    plt.plot([x1,x2], [y1,y2], color='k', linestyle='--')
    yave = np.exp(0.5*np.log(y1) + 0.5*np.log(y2))
    plt.text(x2+0.1, yave, r'r=$'+str(x2)+'$',fontsize=10)

    x1 = 1
    x2 = np.min(np.argwhere(RICd>myCont))
    y1 = RICd[x2-1]
    y2 = RICd[x2-1]
    plt.plot([x1,x2], [y1,y2], color='k', linestyle='--')
    xave = np.exp(0.5*np.log(x1) + 0.5*np.log(x2))
    plt.text(xave, y2+0.4, r'$'+str(np.round(RICd[x2-1],decimals=2))+'\%$',fontsize=10)
    plt.fill_between(index[AEmode-1:x2], RICd[AEmode-1:x2], RICd[0],alpha=0.2,color='blue')

    x1 = np.min(np.argwhere(RICd>myCont))
    x2 = np.min(np.argwhere(RICd>myCont))
    y1 = RICd[0]
    y2 = RICd[x2-1]
    plt.plot([x1,x2], [y1,y2], color='k', linestyle='--')
    yave = np.exp(0.5*np.log(y1) + 0.5*np.log(y2))
    plt.text(x2+2, yave, r'r=$'+str(x2)+'$',fontsize=10)
    
    plt.xscale("log")
    plt.xlabel(r'\bf POD index ($k$)')
    plt.ylabel(r'\bf RIC ($\%$)')
    plt.gca().set_ylim([RICd[0], RICd[-1]+3])
    plt.gca().set_xlim([1, x2*2])

    plt.savefig(dirPlot + '/content.pdf', dpi = 500, bbox_inches = 'tight')
    fig.clear(True)
    #fig, ax = plt.subplots()
    #ax.clear()

def plot1(figNum, epochs, loss, valLoss, label1, label2, label3, label4, plotTitle, fileName):

    fig = plt.figure(figNum)
    plt.plot(epochs, loss)
    plt.plot(epochs, valLoss)
    plt.xlabel(label3)
    plt.ylabel(label4)
    plt.legend([label1, label2])
    plt.title(plotTitle)
    plt.savefig(fileName)
    fig.clear(True)
    
    fig, ax = plt.subplots()
    ax.clear()

def plot(figNum, epochs, loss, valLoss, label1, label2, plotTitle, fileName):

    fig = plt.figure(figNum)
    plt.semilogy(epochs, loss, 'b', label=label1)
    plt.semilogy(epochs, valLoss, 'r', label=label2)
    plt.title(plotTitle)
    plt.legend()
    plt.savefig(fileName)
    fig.clear(True)

    fig, ax = plt.subplots()
    ax.clear()

def subplot(figNum, epochs, predData, trueData, label1, label2, label3, label4, plotTitle, fileName, px, py):

    fig, axs = plt.subplots(px, py, figsize=(18, 9))
    fig.suptitle(plotTitle)

    i = 0
    for ax in axs.flat:
        ax.plot(epochs, trueData[:, i])
        ax.plot(epochs, predData[:, i])
        #if i%px == 0:
        #    ax.set_ylabel(label4)
        #if i%py == 1:
        #    ax.set_xlabel(label3)
        if i == 0:
            ax.legend([label1, label2], loc=0, prop={'size': 6})
        i = i + 1

    plt.savefig(fileName)
    fig.clear(True)

from matplotlib.ticker import FormatStrFormatter
import matplotlib as mpl
mpl.rc('text', usetex=True)
mpl.rcParams['text.latex.preamble']=r"\usepackage{amsmath}"
#mpl.rcParams['text.latex.preamble'] = r'\boldmath'
font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 12}
mpl.rc('font', **font)


def subplotMode(epochsTrain, epochs, trueData, testData,\
                trueLabel, testLabel, fileName, px, py, n):

    #fig, axs = plt.subplots(px, py, figsize=(18,14))
    fig, axs = plt.subplots(px*py, 1, figsize=(18,14))

    i = px*py*n

    for ax in axs.flat:
        ax.plot(epochs,testData[:, i], 'b', label=r'\bf{{{}}}'.format(testLabel), linewidth = 3)
        ax.plot(epochs,trueData[:, i], 'g-.', label=r'\bf{{{}}}'.format(trueLabel), linewidth = 3)
        ax.axvspan(epochsTrain[0], epochsTrain[-1], color='khaki', alpha=0.4, lw=0)
        ax.set_xlim([epochsTrain[0], epochs[-1]])
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        ax.set_ylabel(r'$R_{}(t)$'.format(i+1), labelpad=5)
        i = i + 1

    axs.flat[-1].set_xlabel('Time (s)',fontsize=22)
    axs.flat[-2].set_xlabel('Time (s)',fontsize=22)
    #axs.flat[0].legend(loc="center", bbox_to_anchor=(0.5,1.25),ncol =2,fontsize=22)
    fig.legend(labels=[testLabel, trueLabel], loc="center", bbox_to_anchor=(0.51,1.02),ncol =2,fontsize=22)
    fig.subplots_adjust(hspace=0.5)

    fig.tight_layout()
    plt.savefig(fileName, dpi = 500, bbox_inches = 'tight')
    fig.clear(True)


def subplotModeDA(epochs, trueData, mlData, daData,\
                trueLabel, mlLabel, daLabel, fileName, px, py):

    #fig, axs = plt.subplots(px, py, figsize=(18,14))
    fig, axs = plt.subplots(px*py, 1, figsize=(18,14))

    i = 0

    for ax in axs.flat:
        ax.plot(epochs,daData[:, i], 'r--', label=r'\bf{{{}}}'.format(daLabel), linewidth = 3)
        ax.plot(epochs,mlData[:, i], 'b', label=r'\bf{{{}}}'.format(mlLabel), linewidth = 3)
        ax.plot(epochs,trueData[:, i], 'g-.', label=r'\bf{{{}}}'.format(trueLabel), linewidth = 3)
        #ax.axvspan(epochsTrain[0], epochsTrain[-1], color='khaki', alpha=0.4, lw=0)
        #ax.set_xlim([epochsTrain[0], epochs[-1]])
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        #ax.set_ylabel(r'$R_{}(t)$'.format(i+1), labelpad=5)
        ax.set_ylabel(r'$R_{}(t)$'.format(i+1),fontsize=22, labelpad=5)
        ax.tick_params(axis='both', which='major', labelsize=20)
        i = i + 1

    axs.flat[-1].set_xlabel('Time (s)',fontsize=22)
    #axs.flat[0].legend(loc="center", bbox_to_anchor=(0.5,1.25),ncol =2,fontsize=22)
    fig.legend(labels=[daLabel, mlLabel, trueLabel], loc="center", bbox_to_anchor=(0.51,1.02),ncol =3,fontsize=22)
    fig.subplots_adjust(hspace=0.5)

    fig.tight_layout()
    plt.savefig(fileName, dpi = 500, bbox_inches = 'tight')
    fig.clear(True)



def subplotModeAE(epochsTrain, epochsTest, trueProjct, AEdata,\
                trueLabel, AELabel, fileName, px, py):

    fig, axs = plt.subplots(px, py, figsize=(18,14))

    i = 0
    
    for ax in axs.flat:
        ax.plot(epochsTest,AEdata[:, i], label=r'\bf{{{}}}'.format(AELabel), linewidth = 3)
        ax.plot(epochsTest,trueProjct[:, i], ':', label=r'\bf{{{}}}'.format(trueLabel), linewidth = 3)
        ax.axvspan(epochsTrain[0], epochsTrain[-1], color='khaki', alpha=0.4, lw=0)
        ax.set_xlim([epochsTrain[0], epochsTest[-1]])
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        ax.set_ylabel(r'$r_{}(t)$'.format(i+1), labelpad=5)
        i = i + 1

    axs.flat[-1].set_xlabel(r'$t$',fontsize=22)
    axs.flat[0].legend(loc="center", bbox_to_anchor=(0.5,1.25),ncol =4,fontsize=15)
    fig.subplots_adjust(hspace=0.5)

    plt.savefig(fileName, dpi = 500, bbox_inches = 'tight')
    fig.clear(True)

def subplotProbe(epochsTest, epochsTrain, trueData, testData, FOMData,\
                    trueLabel, testLabel, FOMLabel, fileName, px, py, var):

    #fig, axs = plt.subplots(px, py, figsize=(18, 14))
    fig, axs = plt.subplots(px*py, 1, figsize=(18,14))

    FOMData = FOMData.reshape(FOMData.shape[0], FOMData.shape[1]*FOMData.shape[2])
    testData = testData.reshape(testData.shape[0], testData.shape[1]*testData.shape[2])
    trueData = trueData.reshape(trueData.shape[0], trueData.shape[1]*trueData.shape[2])

    i = 0
    for ax in axs.flat:
        ax.plot(epochsTest,FOMData[:, i], 'k', label=r'\bf{{{}}}'.format(FOMLabel), linewidth = 3)
        ax.plot(epochsTest,testData[:, i], 'r--', label=r'\bf{{{}}}'.format(testLabel), linewidth = 3)
        ax.plot(epochsTest,trueData[:, i], 'g-.', label=r'\bf{{{}}}'.format(trueLabel), linewidth = 3)
        ax.axvspan(epochsTrain[0], epochsTrain[-1], color='khaki', alpha=0.4, lw=0)

        ax.set_xlim([epochsTrain[0], epochsTest[-1]])
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        ax.set_ylabel(r'$\{}$'.format(var)+r'$_{}$'.format(i+1), labelpad=5)
        i = i + 1

    
    axs.flat[-1].set_xlabel('Time (s)',fontsize=22)
    axs.flat[-2].set_xlabel('Time (s)',fontsize=22)
    #axs.flat[0].legend(loc="center", bbox_to_anchor=(0.5,1.25),ncol =4,fontsize=15)
    fig.legend(labels=[FOMLabel, testLabel, trueLabel], loc="center", bbox_to_anchor=(0.51,1.02),ncol =3,fontsize=22)
    fig.subplots_adjust(hspace=0.5)

    fig.tight_layout()
    plt.savefig(fileName, dpi = 500, bbox_inches = 'tight')
    fig.clear(True)



def plotMode(figNum, epochsTest, epochsTrain, trueData, testData, trainData,\
                trueLabel, testLabel, trainLabel, fileName, px, py):

    fig, ax = plt.subplots(px, py, figsize=(18,8))

    i = 0

    ax.plot(epochsTrain,trainData[:, i], label=r'\bf{{{}}}'.format(trainLabel), linewidth = 3)
    ax.plot(epochsTest,testData[:, i], label=r'\bf{{{}}}'.format(testLabel), linewidth = 3)
    ax.plot(epochsTest,trueData[:, i], ':', label=r'\bf{{{}}}'.format(trueLabel), linewidth = 3)
    #ax.plot(epochsTest,trueData[:, i], 'o', markerfacecolor="None", markevery = 4,\
    #        label=r'\bf{{{}}}'.format(trueLabel), linewidth = 3)
    #ax.plot(epochs[ind_m],w[i,:], 'o', fillstyle='none', \
    #        label=r'\bf{Observation}', markersize = 8, markeredgewidth = 2)
    #ax.plot(epochs,ua[i,:], '--', label=r'\bf{Analysis}', linewidth = 3)
    ax.axvspan(epochsTrain[0], epochsTrain[-1], color='y', alpha=0.4, lw=0)
    #if i % 2 == 0:
    #    ax.set_ylabel(r'$z_{}(t)$'.format(i+1), labelpad=5)
    #else:
    #    ax.set_ylabel(r'$z_{}(t)$'.format(i+1), labelpad=-12)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax.set_ylabel(r'$z_{}(t)$'.format(i+1), labelpad=5)
    i = i + 1

    ax.set_xlabel(r'$t$',fontsize=22)
    ax.legend(loc="center", bbox_to_anchor=(0.5,1.25),ncol =4,fontsize=15)
    fig.subplots_adjust(hspace=0.5)

    plt.savefig(fileName, dpi = 500, bbox_inches = 'tight')
    fig.clear(True)


def subplotProbeDA(t_test, FOMData, trueData, mlData, daData,\
                FOMLabel, trueLabel, mlLabel, daLabel, fileName, var):

    fig, axs = plt.subplots(4, 1, figsize=(18, 14))

    FOMData = FOMData.reshape(FOMData.shape[0]*FOMData.shape[1], FOMData.shape[2])
    trueData = trueData.reshape(trueData.shape[0]*trueData.shape[1], trueData.shape[2])
    mlData = mlData.reshape(mlData.shape[0]*mlData.shape[1], mlData.shape[2])
    daData = daData.reshape(daData.shape[0]*daData.shape[1], daData.shape[2])
    

    i = 0
    for ax in axs.flat:
        ax.plot(t_test,daData[i], 'r--', label=r'\bf{{{}}}'.format(daLabel), linewidth = 3)
        ax.plot(t_test,mlData[i], 'b', label=r'\bf{{{}}}'.format(mlLabel), linewidth = 3)
        ax.plot(t_test,trueData[i], 'g-.', label=r'\bf{{{}}}'.format(trueLabel), linewidth = 3)
        ax.plot(t_test,FOMData[i], 'k:', label=r'\bf{{{}}}'.format(FOMLabel), linewidth = 3)

        ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        ax.set_ylabel(r'$\{}$'.format(var)+r'$_{}$'.format(i+1),fontsize=22, labelpad=5)
        ax.tick_params(axis='both', which='major', labelsize=20)
        i = i + 1

    
    axs.flat[-1].set_xlabel('Time (s)',fontsize=22)
    axs.flat[0].legend(loc="center", bbox_to_anchor=(0.5,1.25),ncol =4,fontsize=22)
    fig.subplots_adjust(hspace=0.5)

    fig.tight_layout()
    plt.savefig(fileName, dpi = 500, bbox_inches = 'tight')
    fig.clear(True)