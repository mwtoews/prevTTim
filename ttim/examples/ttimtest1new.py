from pylab import *
from ttim2 import *

ml = ModelMaq(kaq=[1.0,5.0],z=[3,2,1,0],c=[10.],Saq=[0.3,0.01],Sll=[0.001],tmin=0.001,tmax=1000000.0,M=15)
w1 = Well(ml,xw=0,yw=0,rw=1e-5,tsandQ=[(0,1)],layers=[1])
ml.solve()

############################################
# following lines create figures in manual, 
# MLU output read in from data files *.fth in ttim directory
############################################

import os
ttimdir = os.path.dirname( os.path.abspath( __file__ ) ) + '/'  # abspath includes filename. just dirname doesn't work when you are in the directory

t0 = logspace(-3,4,50)
h0 = ml.head(.2,0,t0)
r0=loadtxt(ttimdir+'x0y0.fth')

figure(figsize=(14,10))

subplot(221)
semilogx(t0,-h0[0],'b+')
semilogx(t0,-h0[1],'g+')
semilogx(r0[:,0], -r0[:,1],'b')
semilogx(r0[:,0], -r0[:,2],'g')
legend(('ttim, upper', 'ttim, lower', 'mlu, upper', 'mlu, lower'), loc=2)
xlabel('time')
ylabel('drawdown')
title('r=0.2')

t1 = 10**linspace(-2,5,50)
h1 = ml.head(1,0,t1)
r1=loadtxt(ttimdir+'x1y0.fth')

subplot(222)
semilogx(t1,-h1[0],'b+')
semilogx(t1,-h1[1],'g+')
semilogx(r1[:,0], -r1[:,1],'b')
semilogx(r1[:,0], -r1[:,2],'g')
legend(('ttim, upper', 'ttim, lower', 'mlu, upper', 'mlu, lower'),loc=2)
xlabel('time')
ylabel('drawdown')
title('r=1.0')

t5 = 10**linspace(-1,6,50)
h5 = ml.head(5,0,t5)
r5=loadtxt(ttimdir+'x5y0.fth')

subplot(223)
semilogx(t5,-h5[0],'b+')
semilogx(t5,-h5[1],'g+')
semilogx(r5[:,0], -r5[:,1],'b')
semilogx(r5[:,0], -r5[:,2],'g')
legend(('ttim, upper', 'ttim, lower', 'mlu, upper', 'mlu, lower'),loc=2)
xlabel('time')
ylabel('drawdown')
title('r=5.0')
 
t10 = 10**linspace(-1,6,50)
h10 = ml.head(10,0,t10)
r10=loadtxt(ttimdir+'x10y0.fth')

subplot(224)
semilogx(t10,-h10[0],'b+')
semilogx(t10,-h10[1],'g+')
semilogx(r10[:,0], -r10[:,1],'b')
semilogx(r10[:,0], -r10[:,2],'g')
legend(('ttim, upper', 'ttim, lower', 'mlu, upper', 'mlu, lower'),loc=2)
xlabel('time')
ylabel('drawdown')
title('r=10')
show()
