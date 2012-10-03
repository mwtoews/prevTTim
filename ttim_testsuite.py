import unittest
from ttim import *
import numpy as np

class TTimTest(unittest.TestCase):
    def test_wellbore_storage(self):
        print 'test_wellbore_storage'
        ml = ModelMaq(kaq=[10,5],z=[4,2,1,0],c=[100],Saq=[1e-3,1e-4],Sll=[1e-6],tmin=.01,tmax=10)
        rw = 0.5; rc = 2*rw
        Qw = 5.0
        w = Well(ml,0,0,rw=rw,tsandQ=[(0.0,Qw)],res=1.0,rc=rc,layers=[1,2],wbstype='pumping')
        ml.solve()
        t = np.logspace(-2,1,10)
        Qnet = -Qw + sum(w.strength(t),0)
        volume_change_inside_well = w.headinside(t,derivative=1)*np.pi*rc**2
        for i in w.pylayers:
            np.testing.assert_almost_equal(Qnet,volume_change_inside_well[i],decimal=3)
    def test_ho_linesink_qxqy(self):
        print 'test_ho_linesink_qxqy'
        ml = ModelMaq(kaq=[4,5],z=[4,2,1,0],c=[100],Saq=[1e-1,1e-4],Sll=[1e-6],tmin=1,tmax=10,M=20)
        ls2 = HeadLineSinkHo(ml,order=5,layers=[1,2])
        ml.solve()
        d = 1e-3
        x = 2.0
        y = 3.0
        t = 2.5
        h1 = ml.head(x-d,y,t)[:,0]
        h2 = ml.head(x+d,y,t)[:,0]
        qxnum = (h1-h2)/(2.0*d) * ml.aq.T
        h3 = ml.head(x,y-d,t)[:,0]
        h4 = ml.head(x,y+d,t)[:,0]
        qynum = (h3-h4)/(2.0*d) * ml.aq.T
        qx,qy = ml.discharge(x,y,t)
        print 'qx    ',qx[:,0]
        print 'qxnum ',qxnum
        print 'qy    ',qy[:,0]
        print 'qynum ',qynum
        np.testing.assert_almost_equal(qx[:,0],qxnum,decimal=3)
        np.testing.assert_almost_equal(qx[:,0],qxnum,decimal=3)
    def test_ho_linedoublet_qxqy(self):
        print 'test_ho_linedoublet_qxqy'
        ml = ModelMaq(kaq=[4,5],z=[4,2,1,0],c=[100],Saq=[1e-1,1e-4],Sll=[1e-6],tmin=1,tmax=10,M=20)
        ld1 = LineDoubletHoBase(ml,order=5,type='g')
        ml.solve()
        d = 1e-3
        x = 2.0
        y = 3.0
        p1 = ld1.potinf(x-d,y)
        p2 = ld1.potinf(x+d,y)
        qxnum = (p1-p2)/(2.0*d)
        p3 = ld1.potinf(x,y-d)
        p4 = ld1.potinf(x,y+d)
        qynum = (p3-p4)/(2.0*d)
        qx,qy = ld1.disinf(x,y)
        #print 'qx    ',qx[:,0]
        #print 'qxnum ',qxnum
        #print 'qy    ',qy[:,0]
        #print 'qynum ',qynum
        np.testing.assert_almost_equal(qx,qxnum,decimal=3)
        np.testing.assert_almost_equal(qx,qxnum,decimal=3)
    def test_slug_well(self):
        print 'test_slug_well'
        kgs = np.loadtxt('kgs.txt')
        ml = Model3D(kaq=1,z=np.linspace(4,1,31),Saq=1e-5,kzoverkh=.1,phreatictop=False,tmin=5e-5/86400,tmax=8e5/86400)
        w = Well(ml,0,0,rw=0.029,tsandQ=[(0.0,-np.pi*.029**2)],rc=0.029,layers=range(11,21),wbstype='slug')        
        ml.solve()
        t = kgs[:,0]/86400
        h = w.headinside(t)
        np.testing.assert_almost_equal(h[0],kgs[:,1],decimal=3)
        
#
#if __name__ == '__main__':
#    unittest.main(verbosity=2)
    
if __name__ == '__main__':
    import sys
    from unittest.runner import TextTestRunner
    unittest.main(testRunner=TextTestRunner(stream=sys.stderr))
    
