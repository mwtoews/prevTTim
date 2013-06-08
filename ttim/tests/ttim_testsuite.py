import unittest
from ttim import *
import numpy as np

class TTimTest(unittest.TestCase):
    #def test_wellbore_storage(self):
    #    print 'test_wellbore_storage'
    #    ml = ModelMaq(kaq=[10,5],z=[4,2,1,0],c=[100],Saq=[1e-3,1e-4],Sll=[1e-6],tmin=.01,tmax=10)
    #    rw = 0.5; rc = 2*rw
    #    Qw = 5.0
    #    w = Well(ml,0,0,rw=rw,tsandQ=[(0.0,Qw)],res=1.0,rc=rc,layers=[1,2],wbstype='pumping')
    #    ml.solve()
    #    t = np.logspace(-2,1,10)
    #    Qnet = -Qw + sum(w.strength(t),0)
    #    volume_change_inside_well = w.headinside(t,derivative=1)*np.pi*rc**2
    #    for i in w.pylayers:
    #        np.testing.assert_almost_equal(Qnet,volume_change_inside_well[i],decimal=3)
    #def test_uni_linesink_qxqy(self):
    #    print 'test_uni_linesink_qxqy'
    #    ml = ModelMaq(kaq=[4,5],z=[4,2,1,0],c=[100],Saq=[1e-1,1e-4],Sll=[1e-6],tmin=1,tmax=10,M=20)
    #    ls2 = HeadLineSink(ml,layers=[1,2])
    #    ml.solve()
    #    d = 1e-3
    #    x = 2.0
    #    y = 3.0
    #    t = 2.5
    #    h1 = ml.head(x-d,y,t)[:,0]
    #    h2 = ml.head(x+d,y,t)[:,0]
    #    qxnum = (h1-h2)/(2.0*d) * ml.aq.T
    #    h3 = ml.head(x,y-d,t)[:,0]
    #    h4 = ml.head(x,y+d,t)[:,0]
    #    qynum = (h3-h4)/(2.0*d) * ml.aq.T
    #    qx,qy = ml.discharge(x,y,t)
    #    print 'qx    ',qx[:,0]
    #    print 'qxnum ',qxnum
    #    print 'qy    ',qy[:,0]
    #    print 'qynum ',qynum
    #    np.testing.assert_almost_equal(qx[:,0],qxnum,decimal=3)
    #    np.testing.assert_almost_equal(qx[:,0],qxnum,decimal=3)
    #def test_ho_linesink_qxqy(self):
    #    print 'test_ho_linesink_qxqy'
    #    ml = ModelMaq(kaq=[4,5],z=[4,2,1,0],c=[100],Saq=[1e-1,1e-4],Sll=[1e-6],tmin=1,tmax=10,M=20)
    #    ls2 = HeadLineSinkHo(ml,order=0,layers=[1,2])
    #    ml.solve()
    #    d = 1e-3
    #    x = 2.0
    #    y = 3.0
    #    t = 2.5
    #    h1 = ml.head(x-d,y,t)[:,0]
    #    h2 = ml.head(x+d,y,t)[:,0]
    #    qxnum = (h1-h2)/(2.0*d) * ml.aq.T
    #    h3 = ml.head(x,y-d,t)[:,0]
    #    h4 = ml.head(x,y+d,t)[:,0]
    #    qynum = (h3-h4)/(2.0*d) * ml.aq.T
    #    qx,qy = ml.discharge(x,y,t)
    #    print 'qx    ',qx[:,0]
    #    print 'qxnum ',qxnum
    #    print 'qy    ',qy[:,0]
    #    print 'qynum ',qynum
    #    np.testing.assert_almost_equal(qx[:,0],qxnum,decimal=3)
    #    np.testing.assert_almost_equal(qx[:,0],qxnum,decimal=3)
    #def test_linesink_string_qxqy(self):
    #    print 'test_linesink_string_qxqy'
    #    ml = ModelMaq(kaq=[4,5],z=[4,2,1,0],c=[100],Saq=[1e-1,1e-4],Sll=[1e-6],tmin=1,tmax=10,M=20)
    #    lss = HeadLineSinkString(ml,xy=[(-2,-2),(0,-1),(3,-1)],layers=[1,2])
    #    ml.solve()
    #    d = 1e-3
    #    x = 2.0
    #    y = 3.0
    #    t = 2.5
    #    h1 = ml.head(x-d,y,t)[:,0]
    #    h2 = ml.head(x+d,y,t)[:,0]
    #    qxnum = (h1-h2)/(2.0*d) * ml.aq.T
    #    h3 = ml.head(x,y-d,t)[:,0]
    #    h4 = ml.head(x,y+d,t)[:,0]
    #    qynum = (h3-h4)/(2.0*d) * ml.aq.T
    #    qx,qy = ml.discharge(x,y,t)
    #    print 'qx    ',qx[:,0]
    #    print 'qxnum ',qxnum
    #    print 'qy    ',qy[:,0]
    #    print 'qynum ',qynum
    #    np.testing.assert_almost_equal(qx[:,0],qxnum,decimal=3)
    #    np.testing.assert_almost_equal(qx[:,0],qxnum,decimal=3)
    #def test_ho_linedoublet_qxqy(self):
    #    print 'test_ho_linedoublet_qxqy'
    #    ml = ModelMaq(kaq=[4,5],z=[4,2,1,0],c=[100],Saq=[1e-1,1e-4],Sll=[1e-6],tmin=1,tmax=10,M=20)
    #    ld1 = LineDoubletHoBase(ml,order=5,type='g')
    #    ml.solve()
    #    d = 1e-3
    #    x = 2.0
    #    y = 3.0
    #    p1 = ld1.potinf(x-d,y)
    #    p2 = ld1.potinf(x+d,y)
    #    qxnum = (p1-p2)/(2.0*d)
    #    p3 = ld1.potinf(x,y-d)
    #    p4 = ld1.potinf(x,y+d)
    #    qynum = (p3-p4)/(2.0*d)
    #    qx,qy = ld1.disinf(x,y)
    #    #print 'qx    ',qx[:,0]
    #    #print 'qxnum ',qxnum
    #    #print 'qy    ',qy[:,0]
    #    #print 'qynum ',qynum
    #    np.testing.assert_almost_equal(qx,qxnum,decimal=3)
    #    np.testing.assert_almost_equal(qx,qxnum,decimal=3)
    #def test_slug_well(self):
    #    print 'test_slug_well'
    #    kgs = np.loadtxt('kgs.txt')
    #    ml = Model3D(kaq=1,z=np.linspace(4,1,31),Saq=1e-5,kzoverkh=.1,phreatictop=False,tmin=5e-5/86400,tmax=8e5/86400)
    #    w = Well(ml,0,0,rw=0.029,tsandQ=[(0.0,-np.pi*.029**2)],rc=0.029,layers=range(11,21),wbstype='slug')        
    #    ml.solve()
    #    t = kgs[:,0]/86400
    #    h = w.headinside(t)
    #    np.testing.assert_almost_equal(h[0],kgs[:,1],decimal=3)
    #def test_leaky_wall(self):
    #    print 'test_leaky_wall'
    #    ml = ModelMaq(kaq=[3,4,5],z=[10,6,4,2,1,0],c=[200,100],Saq=[1e-1,1e-3,1e-4],Sll=[1e-5,1e-6],tmin=0.1,tmax=10,M=20)
    #    w = DischargeWell(ml,0,20,.1,layers=[1])
    #    ld1 = LeakyLineDoublet(ml,x1=-20,y1=0,x2=20,y2=0,res=4,order=2,layers=[1,2])  # Along x-axis
    #    ld2 = LeakyLineDoublet(ml,x1=20,y1=0,x2=40,y2=20,res=4,order=2,layers=[2,3])  # 45 degree angle
    #    ml.solve()
    #    t = [3]
    #    qx1,qy1 = ml.discharge(ld1.xc[0],ld1.yc[0],t,ld1.layers)
    #    hmin1 = ml.head(ld1.xcneg[0],ld1.ycneg[0],t,ld1.layers)
    #    hplus1 = ml.head(ld1.xc[0],ld1.yc[0],t,ld1.layers)
    #    qy1num = ml.aq.Haq[ld1.pylayers][:,np.newaxis] * (hmin1-hplus1) / ld1.res
    #    print 'qy1 ',qy1
    #    print 'qy1num ',qy1num
    #    qx2,qy2 = ml.discharge(ld2.xc[1],ld2.yc[1],t,ld2.layers)
    #    qnorm = -qx2 * np.cos(np.pi/4) + qy2 * np.sin(np.pi/4)
    #    hmin2 = ml.head(ld2.xcneg[1],ld2.ycneg[1],t,ld2.layers)
    #    hplus2 = ml.head(ld2.xc[1],ld2.yc[1],t,ld2.layers)
    #    qnum = ml.aq.Haq[ld2.pylayers][:,np.newaxis] * (hmin2-hplus2) / ld2.res
    #    print 'qnorm ',qnorm
    #    print 'qnum ',qnum
    #    np.testing.assert_almost_equal(qy1,qy1num,decimal=3)
    #    np.testing.assert_almost_equal(qnorm,qnum,decimal=3)
    def test_circ_inhom_qxqy(self):
        ml = ModelMaq(kaq=[4,5],z=[4,2,1,0],c=[100],Saq=[1e-3,1e-4],Sll=[1e-6],tmin=1,tmax=10,M=1)
        w = DischargeWell(ml,xw=0,yw=0,rw=.1,tsandQ=[0,5.0],layers=1)
        c1a = CircInhomDataMaq(ml,0,0,10,[10,2],[4,2,1,0],[200],[2e-3,2e-4],[1e-5])
        c1 = CircInhom(ml,0,0,10,order=5)
        ml.initialize()
        x,y = 2.0,3.0
        d = 1e-3
        qx1,qy1 = c1.disinf(x,y) 
        qxnum1 = (c1.potinf(x-d,y)-c1.potinf(x+d,y))/(2*d)
        qynum1 = (c1.potinf(x,y-d)-c1.potinf(x,y+d))/(2*d)
        print 'qx first term    ',qx1[0,0,0]
        print 'qxnum first term ',qxnum1[0,0,0]
        print 'qy first term    ',qy1[0,0,0]
        print 'qynum first term ',qynum1[0,0,0]
        x,y = 12.0,3.0
        d = 1e-3
        qx2,qy2 = c1.disinf(x,y) 
        qxnum2 = (c1.potinf(x-d,y)-c1.potinf(x+d,y))/(2*d)
        qynum2 = (c1.potinf(x,y-d)-c1.potinf(x,y+d))/(2*d)
        print 'qx first term    ',qx2[c1.Nparam/2,0,0]
        print 'qxnum first term ',qxnum2[c1.Nparam/2,0,0]
        print 'qy first term    ',qy2[c1.Nparam/2,0,0]
        print 'qynum first term ',qynum2[c1.Nparam/2,0,0]
        np.testing.assert_allclose(qx1,qxnum1,rtol=1e-5,atol=1e-12)
        np.testing.assert_allclose(qy1,qynum1,rtol=1e-5,atol=1e-12)
        np.testing.assert_allclose(qx2,qxnum2,rtol=1e-5,atol=1e-12)
        np.testing.assert_allclose(qy2,qynum2,rtol=1e-5,atol=1e-12)
    #def test_circ_inhom_with_well(self):
    #    ml = ModelMaq(kaq=[4,5],z=[4,2,1,0],c=[100],Saq=[1e-3,1e-4],Sll=[1e-6],tmin=1,tmax=10,M=20)
    #    w = DischargeWell(ml,xw=5,yw=0,rw=.1,tsandQ=[0,5.0],layers=1)
    #    c1a = CircInhomDataMaq(ml,0,0,10,[10,2],[4,2,1,0],[200],[2e-3,2e-4],[1e-5])
    #    c1 = CircInhom(ml,0,0,10,order=8)
    #    ml.solve()       
    #    a = c1.thetacp
    #    d = 1e-4
    #    x1 = (10-d)*np.cos(a)
    #    y1 = (10-d)*np.sin(a)
    #    x2 = (10+d)*np.cos(a)
    #    y2 = (10+d)*np.sin(a)
    #    h1 = ml.headalongline(x1,y1,2)
    #    h2 = ml.headalongline(x2,y2,2)
    #    hc = ml.headalongline(c1.xc,c1.yc,2)
    #    qx1,qy1 = np.zeros((2,c1.Ncp)),np.zeros((2,c1.Ncp))
    #    qx2,qy2 = np.zeros((2,c1.Ncp)),np.zeros((2,c1.Ncp))
    #    for i in range(c1.Ncp):
    #        qx1[:,i][:,np.newaxis],qy1[:,i][:,np.newaxis] = ml.discharge(x1[i],y1[i],2)
    #        qx2[:,i][:,np.newaxis],qy2[:,i][:,np.newaxis] = ml.discharge(x2[i],y2[i],2)
    #    qn1 = qx1*np.cos(a) + qy1*np.sin(a)
    #    qn2 = qx2*np.cos(a) + qy2*np.sin(a)
    #    np.testing.assert_allclose(h1,h2,rtol=1e-4,atol=1e-8)
    #    np.testing.assert_allclose(qn1,qn2,rtol=1e-4,atol=1e-8)
    #def test_ellip_inhom_qxqy(self):
    #    ml = ModelMaq(kaq=[4,5],z=[4,2,1,0],c=[100],Saq=[1e-3,1e-4],Sll=[1e-6],tmin=1,tmax=10,M=1)
    #    w = DischargeWell(ml,xw=0,yw=0,rw=.1,tsandQ=[0,5.0],layers=1)
    #    e1a = EllipseInhomDataMaq(ml,0,0,along=2.0,bshort=1.0,angle=0.0,kaq=[10,2],z=[4,2,1,0],c=[200],Saq=[2e-3,2e-4],Sll=[1e-5])
    #    e1 = EllipseInhom(ml,0,0,along=2.0,bshort=1.0,angle=0.0,order=5)
    #    ml.initialize()
    #    x,y = 0.3,0.2
    #    d = 1e-3
    #    qx1,qy1 = e1.disinf(x,y) 
    #    qxnum1 = (e1.potinf(x-d,y)-e1.potinf(x+d,y))/(2*d)
    #    qynum1 = (e1.potinf(x,y-d)-e1.potinf(x,y+d))/(2*d)
    #    print 'qx first term    ',qx1[0,0,0]
    #    print 'qxnum first term ',qxnum1[0,0,0]
    #    print 'qy first term    ',qy1[0,0,0]
    #    print 'qynum first term ',qynum1[0,0,0]
    #    x,y = 3.0,0.2
    #    d = 1e-3
    #    qx2,qy2 = e1.disinf(x,y) 
    #    qxnum2 = (e1.potinf(x-d,y)-e1.potinf(x+d,y))/(2*d)
    #    qynum2 = (e1.potinf(x,y-d)-e1.potinf(x,y+d))/(2*d)
    #    print 'qx first term    ',qx2[e1.Nparam/2,0,0]
    #    print 'qxnum first term ',qxnum2[e1.Nparam/2,0,0]
    #    print 'qy first term    ',qy2[e1.Nparam/2,0,0]
    #    print 'qynum first term ',qynum2[e1.Nparam/2,0,0]
    #    np.testing.assert_allclose(qx1,qxnum1,rtol=1e-5,atol=1e-8)
    #    np.testing.assert_allclose(qy1,qynum1,rtol=1e-5,atol=1e-8)
    #    np.testing.assert_allclose(qx2,qxnum2,rtol=1e-5,atol=1e-8)
    #    np.testing.assert_allclose(qy2,qynum2,rtol=1e-5,atol=1e-8)
    #def test_ellip_inhom_with_well(self):
    #    ml = ModelMaq(kaq=[4,5],z=[4,2,1,0],c=[100],Saq=[1e-3,1e-4],Sll=[1e-6],tmin=1,tmax=10,M=20)
    #    w = DischargeWell(ml,xw=.5,yw=0,rw=.1,tsandQ=[0,5.0],layers=1)
    #    e1a = EllipseInhomDataMaq(ml,0,0,along=2.0,bshort=1.0,angle=0.0,kaq=[10,2],z=[4,2,1,0],c=[200],Saq=[2e-3,2e-4],Sll=[1e-5])
    #    e1 = EllipseInhom(ml,0,0,along=2.0,bshort=1.0,angle=0.0,order=5)
    #    ml.solve()       
    #    h1,h2 = np.zeros((2,e1.Ncp)), np.zeros((2,e1.Ncp))
    #    qn1,qn2 = np.zeros((2,e1.Ncp)), np.zeros((2,e1.Ncp))
    #    for i in range(e1.Ncp):
    #        h1[:,i] = ml.head(e1.xc[i],e1.yc[i],2,aq=e1.aqin)[:,0]
    #        h2[:,i] = ml.head(e1.xc[i],e1.yc[i],2,aq=e1.aqout)[:,0]
    #        qx1,qy1 = ml.discharge(e1.xc[i],e1.yc[i],2,aq=e1.aqin)
    #        qx2,qy2 = ml.discharge(e1.xc[i],e1.yc[i],2,aq=e1.aqout)
    #        a = e1a.outwardnormalangle(e1.xc[i],e1.yc[i])
    #        qn1[:,i] = qx1[:,0]*np.cos(a) + qy1[:,0]*np.sin(a)
    #        qn2[:,i] = qx2[:,0]*np.cos(a) + qy2[:,0]*np.sin(a)
    #    np.testing.assert_allclose(h1,h2,rtol=1e-4,atol=1e-8)
    #    np.testing.assert_allclose(qn1,qn2,rtol=1e-3,atol=1e-8)



#
#if __name__ == '__main__':
#    unittest.main(verbosity=2)
    
if __name__ == '__main__':
    import sys
    from unittest.runner import TextTestRunner
    unittest.main(testRunner=TextTestRunner(stream=sys.stderr))
    
