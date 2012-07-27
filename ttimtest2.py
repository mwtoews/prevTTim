from pylab import *
from ttim import *

#
# MODEL MLW - multiple wells to model linesink
#

mlw = ModelMaq(kaq=[1.0,5.0],z=[3,2,1,0],c=[1000.],Saq=[0.3,0.003],Sll=[0.001],tmin=.01,tmax=1000000.0,M=20)
xlist = linspace(-4.5,4.5,10)
for x in xlist:
    DischargeWell(mlw, xw=x, yw=0, rw=.1, tsandQ=[(0,1)], layers=[1])
mlw.solve()

#
# MODEL MLLS - linesink model
#

mlls = ModelMaq(kaq=[1.0,5.0],z=[3,2,1,0],c=[1000.],Saq=[0.3,0.003],Sll=[0.001],tmin=.01,tmax=1000000.0,M=20)
LineSink(mlls, x1=-5, y1=0, x2=5, y2=0, tsandQ=[(0,10)], layers=1)
mlls.solve()

#
# graphical comparison of results
#

t=10**linspace(-2,4,50)
hw = mlw.head(0,0,t)
hls = mlls.head(0,0,t)

figure(figsize=(14,10))

subplot(221)
semilogx(t,-hls[0], 'b')
semilogx(t,-hls[1],'g')
semilogx(t,-hw[0], 'b+')
semilogx(t,-hw[1], 'g+')
legend(('linesink, upper', 'linesink, lower', 'wells, upper', 'wells, lower'), loc=2)
xlabel('time')
ylabel('drawdown')
title('x=0, y=0')

t=10**linspace(-1,5,50)
hw = mlw.head(0,2,t)
hls = mlls.head(0,2,t)

subplot(222)
semilogx(t,-hls[0], 'b')
semilogx(t,-hls[1],'g')
semilogx(t,-hw[0], 'b+')
semilogx(t,-hw[1], 'g+')
legend(('linesink, upper', 'linesink, lower', 'wells, upper', 'wells, lower'), loc=2)
xlabel('time')
ylabel('drawdown')
title('x=0, y=2')

t = 10**linspace(0,6,50)
hw = mlw.head(10,0,t)
hls = mlls.head(10,0,t)

subplot(223)
semilogx(t,-hls[0], 'b')
semilogx(t,-hls[1],'g')
semilogx(t,-hw[0], 'b+')
semilogx(t,-hw[1], 'g+')
legend(('linesink, upper', 'linesink, lower', 'wells, upper', 'wells, lower'), loc=2)
xlabel('time')
ylabel('drawdown')
title('x=10, y=0')

subplot(224)
timcontour(mlw,-6, 6, 60, -3, 3, 60, [-3.5, 0.0, .25], 40, 1,'r', newfig=False)
timcontour(mlls,-6, 6, 60, -3, 3, 60, [-3.5, 0.0, .25], 40, 1,'b', newfig=False)
title('contour plot')
axis('scaled')
show()