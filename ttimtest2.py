from pylab import *
from ttim import *
#
# MODEL MLW - multiple wells to model linesink
#
mlw = ModelMaq(kaq=[1.0,5.0],z=[3,2,1,0],c=[1000.],Saq=[0.3,0.003],Sll=[0.001],tmin=.01,tmax=1000000.0,M=20)
w1 = Well(mlw,-4.5,0,.1,1,[1])
w2 = Well(mlw,-3.5,0,.1,1,[1])
w3 = Well(mlw,-2.5,0,.1,1,[1])
w4 = Well(mlw,-1.5,0,.1,1,[1])
w5 = Well(mlw,-0.5,0,.1,1,[1])
w10 = Well(mlw,4.5,0,.1,1,[1])
w9 = Well(mlw,3.5,0,.1,1,[1])
w8 = Well(mlw,2.5,0,.1,1,[1])
w7 = Well(mlw,1.5,0,.1,1,[1])
w6 = Well(mlw,0.5,0,.1,1,[1])
mlw.solve()
#
# MODEL MLLS - linesink model
#
mlls = ModelMaq(kaq=[1.0,5.0],z=[3,2,1,0],c=[1000.],Saq=[0.3,0.003],Sll=[0.001],tmin=.01,tmax=1000000.0,M=20)
LineSink(mlls, -5.0, 0.0, 5.0, 0.0, 1.0, 1)
mlls.solve()
#
# graphical comparison of results
#
t=10**linspace(-2,4,50)
hw = mlw.head(0,0,t)
hls = mlls.head(0,0,t)

figure(1)
semilogx(t,-hls[0], 'b')
semilogx(t,-hls[1],'g')
semilogx(t,-hw[0], 'b+')
semilogx(t,-hw[1], 'g+')
legend(('linesink, upper', 'linesink, lower', 'wells, upper', 'wells, lower'), loc=2)
xlabel('time')
ylabel('drawdown')
title('x=0.0, y=0.0')
show()

t=10**linspace(-1,5,50)
hw = mlw.head(0,2,t)
hls = mlls.head(0,2,t)

figure(2)
semilogx(t,-hls[0], 'b')
semilogx(t,-hls[1],'g')
semilogx(t,-hw[0], 'b+')
semilogx(t,-hw[1], 'g+')
legend(('linesink, upper', 'linesink, lower', 'wells, upper', 'wells, lower'), loc=2)
xlabel('time')
ylabel('drawdown')
title('x=0.0, y=2.0')
show()

t = 10**linspace(0,6,50)
hw = mlw.head(10,0,t)
hls = mlls.head(10,0,t)

figure(3)
semilogx(t,-hls[0], 'b')
semilogx(t,-hls[1],'g')
semilogx(t,-hw[0], 'b+')
semilogx(t,-hw[1], 'g+')
legend(('linesink, upper', 'linesink, lower', 'wells, upper', 'wells, lower'), loc=2)
xlabel('time')
ylabel('drawdown')
title('x=10.0, y=0.0')
show()

pycontour(mlw,-6, 6, 60, -3, 3, 60, [-3.5, 0.0, .25], 40, 1,'r')
pycontour(mlls,-6, 6, 60, -3, 3, 60, [-3.5, 0.0, .25], 40, 1,'b', newfig=False)
axis('scaled')
show()