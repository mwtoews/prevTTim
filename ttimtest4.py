from pylab import *
import matplotlib.image as mpimg
from ttim import *

######################################################
# Neuman figure read from neuman.png in ttim directory
###################################################

import os
ttimdir = os.path.dirname( os.path.abspath( __file__ ) ) + '/'  # abspath includes filename. just dirname doesn't work when you are in the directory

# Figure
plt.figure()
img = mpimg.imread(ttimdir+'neuman.png')
ax = plt.imshow(np.flipud(img),origin='lower',extent=(-1,5,-2,1))
ax = ax.get_axes()
plt.draw()
b = ax.get_position()
ax = plt.axes(b)
ax.patch.set_facecolor('None')

for Ss in [1e-3,1e-4,1e-5,1e-6,1e-7]:
    kaq = 1.0 * np.ones(11)
    z = np.arange(11.0,-1,-1); z[0] = 10.01
    S = Ss * np.ones(11); S[0] = 0.1
    #
    T = 10.0; Saq = S[-1] * 10
    sig = Saq / S[0]
    print 'sigma equals ',sig
    ts = np.logspace(-1,5,40)
    t = ts * Saq * 10**2 / T
    #
    ml = Model3D(kaq,z,S,kzoverkh=1,phreatictop=True,tmin=t[0],tmax=t[-1],M=20)
    w = DischargeWell(ml, xw=0, yw=0, rw=.1, tsandQ=[(0,1)], layers=range(2,12))
    #
    ml.solve()
    h = ml.potential(10,0,t)
    h = -h*4*np.pi*T/10.0  # Make dimensionless
    plot(log10(ts),log10(h[-1]),'+')
    
xlabel('$Tt/(Sr^2)$',fontsize=14)
ylabel('$4\pi Ts/Q$',fontsize=14)

show()