from pylab import *
from ttim2 import *

ml = ModelMaq(kaq=[1.0,5.0],z=[3,2,1,0],c=[10.],Saq=[0.03,0.03],Sll=[0.001],tmin=.001,tmax=100.0,M=20)
xy = zip(-5*ones(11),linspace(-5,5,11))
ZeroMscreenLineSinkString(ml,xy=xy,layers=[1,2])
Well(ml, xw=0, yw=0, rw=0.1, tsandQ=[(0,-10.)], layers=1)
ml.solve()

######################################################
# following lines create figures in manual, Modflow 
# output read from data files *.dat in ttim directory
###################################################

import os
ttimdir = os.path.dirname( os.path.abspath( __file__ ) ) + '/'  # abspath includes filename. just dirname doesn't work when you are in the directory

t = 10**linspace(-2,1.69897000434,100)

figure(figsize=(14,10))

subplot(221)
h = ml.head(-5,4.9,t)
ncrack=loadtxt(ttimdir+'NCrack.dat')
semilogx(t,h[0], 'b')
semilogx(t,h[1], 'g')
semilogx(ncrack[:,0], ncrack[:,1]-3.,'b+')
semilogx(ncrack[:,0], ncrack[:,2]-3.,'g+')
legend(('ttim, upper', 'ttim, lower', 'MODFLOW, upper', 'MODFLOW, lower'), loc=2)
xlabel('time')
ylabel('minus drawdown')
title('north end of crack')

t = 10**linspace(-2,1.69897000434,100)

subplot(222)
h = ml.head(-5,0,t)
ccrack=loadtxt(ttimdir+'CCrack.dat')
semilogx(t,h[0], 'b')
semilogx(t,h[1], 'g')
semilogx(ccrack[:,0], ccrack[:,1]-3.,'b+')
semilogx(ccrack[:,0], ccrack[:,2]-3.,'g+')
legend(('ttim, upper', 'ttim, lower', 'MODFLOW, upper', 'MODFLOW, lower'), loc=2)
xlabel('time')
ylabel('minus drawdown')
title('center of crack')

t = 10**linspace(-2,1.69897000434,100)

subplot(223)
h = ml.head(-5,-4.9,t)
scrack=loadtxt(ttimdir+'SCrack.dat')
semilogx(t,h[0], 'b')
semilogx(t,h[1], 'g')
semilogx(scrack[:,0], scrack[:,1]-3.,'b+')
semilogx(scrack[:,0], scrack[:,2]-3.,'g+')
legend(('ttim, upper', 'ttim, lower', 'MODFLOW, upper', 'MODFLOW, lower'), loc=2)
xlabel('time')
ylabel('minus drawdown')
title('south end of crack')

subplot(224)
crackxs=loadtxt(ttimdir+'Crackxst10.dat')
#xsection(ml, -10.0, 10.0, 0, 0, 201., 9.69, [1,2])
xsection(ml, -10.0, 10.0, 0, 0, 201., 10, [1,2], newfig=False)
plot(crackxs[:,0]-90., crackxs[:,1]-3.,'b+')
plot(crackxs[:,0]-90., crackxs[:,2]-3.,'g+')
#legend(('ttim, upper', 'ttim, lower', 'MODFLOW, upper', 'MODFLOW, lower'), loc=2)
xlabel('distance')
ylabel('minus drawdown')
title('t=10.0')
show()