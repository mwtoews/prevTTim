from pylab import *
from ttim import *

ml = ModelMaq(kaq=[1.0,5.0],z=[3,2,1,0],c=[10.],Saq=[0.03,0.03],Sll=[0.001],tmin=.001,tmax=100.0,M=20)

MscreenLineSink(ml,-5., -5.0, -5.,-4.0, 0.0, [1,2])
MscreenLineSink(ml,-5., -4.0, -5.,-3.0, 0.0, [1,2])
MscreenLineSink(ml,-5., -3.0, -5.,-2.0, 0.0, [1,2])
MscreenLineSink(ml,-5., -2.0, -5.,-1.0, 0.0, [1,2])
MscreenLineSink(ml,-5., -1.0, -5., 0.0, 0.0, [1,2])
MscreenLineSink(ml,-5.,  0.0, -5., 1.0, 0.0, [1,2])
MscreenLineSink(ml,-5.,  1.0, -5., 2.0, 0.0, [1,2])
MscreenLineSink(ml,-5.,  2.0, -5., 3.0, 0.0, [1,2])
MscreenLineSink(ml,-5.,  3.0, -5., 4.0, 0.0, [1,2])
MscreenLineSink(ml,-5.,  4.0, -5., 5.0, 0.0, [1,2])

w1=Well(ml,0., 0., 0.1, -10., 1)
ml.solve()

#######################################################
# following lines create figures in manual, Modflow 
# output read from data files *.dat
####################################################
#
#t = 10**linspace(-2,1.69897000434,100)
#
#figure(1)
#h = ml.head(-5,4.9,t)
#ncrack=loadtxt('NCrack.dat')
#semilogx(t,h[0], 'b')
#semilogx(t,h[1], 'g')
#semilogx(ncrack[:,0], ncrack[:,1]-3.,'b+')
#semilogx(ncrack[:,0], ncrack[:,2]-3.,'g+')
#legend(('ttim, upper', 'ttim, lower', 'MODFLOW, upper', 'MODFLOW, lower'), loc=2)
#xlabel('time')
#ylabel('minus drawdown')
#title('north end of crack')
#show()
#
#t = 10**linspace(-2,1.69897000434,100)
#
#figure(2)
#h = ml.head(-5,0,t)
#ccrack=loadtxt('CCrack.dat')
#semilogx(t,h[0], 'b')
#semilogx(t,h[1], 'g')
#semilogx(ccrack[:,0], ccrack[:,1]-3.,'b+')
#semilogx(ccrack[:,0], ccrack[:,2]-3.,'g+')
#legend(('ttim, upper', 'ttim, lower', 'MODFLOW, upper', 'MODFLOW, lower'), loc=2)
#xlabel('time')
#ylabel('minus drawdown')
#title('center of crack')
#show()
#
#t = 10**linspace(-2,1.69897000434,100)
#
#figure(3)
#h = ml.head(-5,-4.9,t)
#scrack=loadtxt('SCrack.dat')
#semilogx(t,h[0], 'b')
#semilogx(t,h[1], 'g')
#semilogx(scrack[:,0], scrack[:,1]-3.,'b+')
#semilogx(scrack[:,0], scrack[:,2]-3.,'g+')
#legend(('ttim, upper', 'ttim, lower', 'MODFLOW, upper', 'MODFLOW, lower'), loc=2)
#xlabel('time')
#ylabel('minus drawdown')
#title('south end of crack')
#show()
#
#crackxs=loadtxt('Crackxst10.dat')
#xsection(ml, -10.0, 10.0, 0, 0, 201., 9.69, [1,2])
#plot(crackxs[:,0]-90., crackxs[:,1]-3.,'b+')
#plot(crackxs[:,0]-90., crackxs[:,2]-3.,'g+')
#legend(('ttim, upper', 'ttim, lower', 'MODFLOW, upper', 'MODFLOW, lower'), loc=2)
#xlabel('distance')
#ylabel('minus drawdown')
#title('t=10.0')
#show()