from pylab import *
from ttim import *

kaq = 1.0 * np.ones(11)
z = np.arange(11.0,-1,-1); z[0] = 10.01
S = 0.001 * np.ones(11); S[0] = 0.1
#
T = 10.0; Saq = S[-1] * 10
sig = Saq / S[0]
print 'sigma equals ',sig
ts = 10**np.linspace(-1,5,40)
t = ts * Saq * 10**2 / T
#
ml = Model3D(kaq,z,S,kzoverkh=1,phreatictop=True,tmin=t[0],tmax=t[-1],M=20)
w = Well(ml,0,0,.1,np.ones(10),range(2,12))
#
ml.solve()
h = ml.potential(10,0,t)
h = -h*4*np.pi*T/10.0  # Make dimensionless

#plot(log10(ts),log10(h[-1]),'b+')