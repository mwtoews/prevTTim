'''
Copyright (C), 2010, Mark Bakker.
TTim is distributed under the MIT license
'''

import numpy as np
import matplotlib.pyplot as plt
from bessel import *
from invlap import *

def ModelMaq(kaq=[1],z=[1,0],c=[],Saq=[],Sll=[],topboundary='imp',phreatictop=False,tmin=1,tmax=10,M=20):
    kaq = np.atleast_1d(kaq).astype('d')
    Naq = len(kaq)
    z = np.atleast_1d(z).astype('d')
    c = np.atleast_1d(c).astype('d')
    Saq = np.atleast_1d(Saq).astype('d')
    Sll = np.atleast_1d(Sll).astype('d')
    H = z[:-1] - z[1:]
    if topboundary[:3] == 'imp':
        assert len(z) == 2*Naq, 'Error: Length of z needs to be ' + str(2*Naq)
        assert len(c) == Naq-1, 'Error: Length of c needs to be ' + str(Naq-1)
        assert len(Saq) == Naq, 'Error: Length of Saq needs to be ' + str(Naq)
        assert len(Sll) == Naq-1, 'Error: Length of Sll needs to be ' + str(Naq-1)
        T = kaq * H[::2]
        Saq = Saq * H[::2]
        if phreatictop: Saq[0] = Saq[0] / H[0]
        Sll = Sll * H[1::2]
        c = np.hstack((np.nan,c))
        Sll = np.hstack((np.nan,Sll))
    else: # leaky layer on top
        assert len(z) == 2*Naq+1, 'Error: Length of z needs to be ' + str(2*Naq+1)
        assert len(c) == Naq, 'Error: Length of c needs to be ' + str(Naq)
        assert len(Saq) == Naq, 'Error: Length of Saq needs to be ' + str(Naq)
        assert len(Sll) == Naq, 'Error: Length of Sll needs to be ' + str(Naq)
        T = kaq * H[1::2]
        Saq = Saq * H[1::2]
        Sll = Sll * H[::2]
        if phreatictop and (topboundary[:3]=='lea'): Sll[0] = Sll[0] / H[0]
    return TimModel(T,c,Saq,Sll,topboundary,tmin,tmax,M)
    
def Model3D(kaq=[1,1,1],z=[4,3,2,1],Saq=[0.3,0.001,0.001],kzoverkh=[.1,.1,.1],phreatictop=True,tmin=1,tmax=10,M=20):
    kaq = np.atleast_1d(kaq).astype('d')
    Naq = len(kaq)
    z = np.atleast_1d(z).astype('d')
    Saq = np.atleast_1d(Saq).astype('d')
    kzoverkh = np.atleast_1d(kzoverkh).astype('d')
    if len(kzoverkh) == 1: kzoverkh = kzoverkh * np.ones(Naq)
    H = z[:-1] - z[1:]
    T = kaq * H
    c = 0.5 * H[:-1] / ( kzoverkh[:-1] * kaq[:-1] ) + 0.5 * H[1:] / ( kzoverkh[1:] * kaq[1:] )
    Saq = Saq * H
    if phreatictop: Saq[0] = Saq[0] / H[0]
    c = np.hstack((np.nan,c))
    Sll = 1e-20 * np.ones(len(c))
    return TimModel(T,c,Saq,Sll,'imp',tmin,tmax,M)

class TimModel:
    def __init__(self,T=[1,1],c=[np.nan,100],Saq=[0.3,0.003],Sll=[1e-3],topboundary='imp',tmin=1,tmax=10,M=20):
        self.elementList = []
        self.tmin = float(tmin)
        self.tmax = float(tmax)
        self.M = M
        self.aq = Aquifer(self,T,c,Saq,Sll,topboundary)
        self.compute_laplace_parameters()
	self.name = 'ml'
        bessel.initialize()
    def __repr__(self):
	return 'Model'
    def addElement(self,el):
        self.elementList.append(el)
    def compute_laplace_parameters(self):
        itmin = int(np.floor(np.log10(self.tmin)))
        itmax = int(np.ceil(np.log10(self.tmax)))
        self.tintervals = np.arange(itmin,itmax+1)
        self.tintervals = 10.0**self.tintervals
        alpha = 0.0
        tol = 1e-9
        self.Nin = itmax - itmin
        run = np.arange(2*self.M+1)  # so there are 2M+1 terms in Fourier series expansion
        self.p = []
        self.gamma = []
        for i in range(self.Nin):
            T = self.tintervals[i+1] * 2.0
            gamma = alpha - np.log(tol) / (T/2.0)
            p = gamma + 1j * np.pi * run / T
            self.p.extend( p.tolist() )
            self.gamma.append(gamma)
        self.p = np.array(self.p)
        self.gamma = np.array(self.gamma)
        self.Np = len(self.p)
        self.Npin = 2 * self.M + 1
        self.aq.initialize()
    def potential(self,x,y,t,aq=None):
        '''returns array of potentials of len(t)
        t must be ordered and tmin <= t <= tmax'''
	if aq is None: aq = self.aq.findAquiferData(x,y)
        t = np.atleast_1d(t)
        pot = np.zeros((aq.Naq,self.Np),'D')
        for e in self.elementList:
            pot += e.potential(x,y,aq)
        #pot = np.sum( [ e.potential(x,y,aq) for e in self.elementList ], 0 )  # slower for 30 elements
        pot = np.sum( pot * aq.eigvec, 1 )
        rv = np.zeros((aq.Naq,len(t)))
        it = 0
        for n in range(self.Nin):
            if n == self.Nin-1:
                tp = t[ (t >= self.tintervals[n]) & (t <= self.tintervals[n+1]) ]
            else:
                tp = t[ (t >= self.tintervals[n]) & (t < self.tintervals[n+1]) ]
            Nt = len(tp)
            if Nt > 0:  # if all values zero, don't do the inverse transform
                for i in range(aq.Naq):
                    if np.abs( pot[i,n*self.Npin] ) > 1e-20:
                        rv[i,it:it+Nt] = invlaptrans.invlap( tp, self.tintervals[n], self.tintervals[n+1], pot[i,n*self.Npin:(n+1)*self.Npin], self.gamma[n], self.M, Nt )
                it = it + Nt
        return rv
    def head(self,x,y,t,aq=None):
        if aq is None: aq = self.aq.findAquiferData(x,y)
        pot = self.potential(x,y,t,aq)
        for i in range(aq.Naq):
            pot[i] = aq.potentialToHead(pot[i],i)
        return pot
    def phi(self,x,y,aq=None):
        '''array of complex potentials'''
	if aq is None: aq = self.aq.findAquiferData(x,y)
        pot = np.zeros((aq.Naq,self.Np),'D')
        for e in self.elementList:
            pot += e.potential(x,y,aq)
        pot = np.sum( pot * aq.eigvec, 1 )
        return pot
    def phihead(self,x,y,aq=None):
        '''array of complex heads'''
	if aq is None: aq = self.aq.findAquiferData(x,y)
        pot = np.zeros((aq.Naq,self.Np),'D')
        for e in self.elementList:
            pot += e.potential(x,y,aq)
        pot = np.sum( pot * aq.eigvec, 1 )
        return pot / aq.T[:,np.newaxis]
    def headgrid(self,x1,x2,nx,y1,y2,ny,t):
        xg,yg = np.meshgrid( np.linspace(x1,x2,nx), np.linspace(y1,y2,ny) ) 
        t = np.atleast_1d(t)
        hphi = np.zeros( ( ml.aq.Naq, ml.Np, ny, nx ), 'd' )
        for irow in range(ny):
            for jcol in range(nx):
                hphi[:,:,irow,jcol] = ml.phihead(xg[irow,jcol], yg[irow,jcol])
        # Contour
        # Colors
        h = np.zeros( (ml.aq.Naq,len(t),ny,nx) )
        for irow in range(ny):
            for jcol in range(nx):
                for k in range(ml.aq.Naq):
                    h[k,:,irow,jcol] = ml.inverseLapTran(hphi[k,:,irow,jcol],t)
        return h
    def headalongline(self,x,y,t):
        xg,yg = np.atleast_1d(x),np.atleast_1d(y)
        nx = len(xg)
        if len(yg) == 1:
            yg = yg * np.ones(nx)
        t = np.atleast_1d(t)
        hphi = np.zeros( ( ml.aq.Naq, ml.Np, nx ), 'd' )
        for i in range(nx):
            hphi[:,:,i] = ml.phihead(xg[i], yg[i])
        h = np.zeros( (ml.aq.Naq,len(t),nx) )
        for i in range(nx):
            for k in range(ml.aq.Naq):
                h[k,:,i] = ml.inverseLapTran(hphi[k,:,i],t)
        return h
    def inverseLapTran(self,pot,t):
        '''returns array of potentials of len(t)
        t must be ordered and tmin <= t <= tmax'''
        t = np.atleast_1d(t)
        rv = np.zeros(len(t))
        it = 0
        for n in range(self.Nin):
            if n == self.Nin-1:
                tp = t[ (t >= self.tintervals[n]) & (t <= self.tintervals[n+1]) ]
            else:
                tp = t[ (t >= self.tintervals[n]) & (t < self.tintervals[n+1]) ]
            Nt = len(tp)
            if Nt > 0:  # if all values zero, don't do the inverse transform
                if np.abs( pot[n*self.Npin] ) > 1e-20:
                    rv[it:it+Nt] = invlaptrans.invlap( tp, self.tintervals[n], self.tintervals[n+1], pot[n*self.Npin:(n+1)*self.Npin], self.gamma[n], self.M, Nt )
                it = it + Nt
        return rv
    def solve(self,printmat = 0,sendback=0):
        '''Compute solution'''
	# Initialize elements
	self.aq.initialize()
	for e in self.elementList:
	    e.initialize()
	# Compute number of equations
	self.Neq = np.sum( [e.Nunknowns for e in self.elementList] )
	print 'self.Neq ',self.Neq
	if self.Neq == 0:
	    print 'No unknowns. Solution complete'
	    return
        mat = np.empty( (self.Neq,self.Neq,self.Np), 'D' )
        rhs = np.empty( (self.Neq,self.Np), 'D' )
        ieq = 0
        for e in self.elementList:
	    if e.Nunknowns > 0:
		mat[ ieq:ieq+e.Nunknowns, :, : ], rhs[ ieq:ieq+e.Nunknowns, : ] = e.equation()
		ieq += e.Nunknowns
        if printmat:
            print 'mat ',mat
            print 'rhs ',rhs
	for i in range( self.Np ):
	    sol = np.linalg.solve( mat[:,:,i], rhs[:,i] )
	    icount = 0
	    for e in self.elementList:
		if e.Nunknowns == 1:
		    e.parameters[0,i] = sol[icount]
		    icount += e.Nunknowns
		elif e.Nunknowns > 1:
		    e.parameters[:,i] = sol[icount:icount+e.Nunknowns]
		    icount += e.Nunknowns
        print 'solution complete'
	if sendback:
	    return sol
        return
    def check(self,full_output=False):
	maxerror = 0.0; maxelement = None
        for e in self.elementList:
            error = e.check(full_output)
	    if error > maxerror:
		maxerror = error
		maxelement = e
	print 'Maximum error '+str(maxerror)
	print 'Occurs at element: '+str(maxelement)
        
class Aquifer:
    def __init__(self,model,T,c,Saq,Sll,topboundary):
	self.model = model
        self.Naq = len(T)
        self.T = np.atleast_1d(T).astype('d')
        self.c = np.atleast_1d(c).astype('d')
        self.Saq = np.atleast_1d(Saq).astype('d')
        self.Sll = np.atleast_1d(Sll).astype('d')
        self.Sll[self.Sll<1e-20] = 1e-20 # Cannot be zero
        self.topboundary = topboundary[:3]
        self.D = self.T / self.Saq
        self.inhomList = []
    def __repr__(self):
        return 'Aquifer T: ' + str(self.T)
    def initialize(self):
        self.eigval = []
        self.eigvec = []
        self.coef = []
        b = np.diag(np.ones(self.Naq))
        for p in self.model.p:
            w,v = self.compute_lab_eigvec(p)
            v = v.T # eigenvectors are stored as rows
            index = np.argsort( abs(w) )[::-1]
            w = w[index]; v = v[index,:]
            self.eigval.append(w); self.eigvec.append(v)
            self.coef.append( np.linalg.solve( v.T, b ).T )
        self.eigval = np.array(self.eigval);
        self.eigval = self.eigval.T
        self.lab = 1.0 / np.sqrt(self.eigval)
        self.lab2 = self.lab.copy(); self.lab2.shape = (self.Naq,self.model.Nin,self.model.Npin)
        self.eigvec = np.array(self.eigvec)
        self.eigvec = self.eigvec.T
        self.coef = np.array(self.coef).T
        # coef[:,ipylayer,np] are the coefficients if the element is in ipylayer belonging to Laplace parameter number np
    def initialize_old(self):
        self.eigval = []
        self.eigvec = []
        self.coef = []
        b = np.diag(np.ones(self.Naq))
        for p in self.model.p:
            w,v = self.compute_lab_eigvec(p)
            self.eigval.append(w)
            self.eigvec.append(v.T)
            self.coef.append( np.linalg.solve( v, b ).T )
        self.eigval = np.array(self.eigval);
        self.eigval = self.eigval.T
        self.lab = 1.0 / np.sqrt(self.eigval)
        self.lab2 = self.lab.copy(); self.lab2.shape = (self.Naq,self.model.Nin,self.model.Npin)
        self.eigvec = np.array(self.eigvec)
        self.eigvec = self.eigvec.T
        self.coef = np.array(self.coef).T
        # coef[:,ipylayer,np] are the coefficients if the element is in ipylayer belonging to Laplace parameter number np
    def findAquiferData(self,x,y):
        return self
    def headToPotential(self,h,pylayer):
        return h * self.T[pylayer]
    def potentialToHead(self,p,pylayer):
        return p / self.T[pylayer]
    def compute_lab_eigvec(self,p):
        sqrtpSc = np.sqrt( p * self.Sll * self.c )
        a = sqrtpSc / np.tanh(sqrtpSc)
        b = sqrtpSc / np.sinh(sqrtpSc)
        if (self.topboundary == 'sem') or (self.topboundary == 'lea'): dzero = sqrtpSc[0] * np.tanh( sqrtpSc[0] )
        #d0 = np.zeros(self.Naq,'D')
        #d0[1:-1] = p / self.D[1:-1] + a[1:-1] / ( self.c[1:-1] * self.T[1:-1] ) + a[2:] / ( self.c[2:] * self.T[1:-1] )
        #d0[-1] = p / self.D[-1] + a[-1] / ( self.c[-1] * self.T[-1] )
        #d0[0] = p / self.D[0] + a[1] / ( self.c[1] * self.T[0] )
        
        d0 = p / self.D
        d0[:-1] += a[1:] / (self.c[1:] * self.T[:-1])
        d0[1:]  += a[1:] / (self.c[1:] * self.T[1:])
        if self.topboundary == 'lea':
            d0[0] += dzero / ( self.c[0] * self.T[0] )
        elif self.topboundary == 'sem':
            d0[0] += a[0] / ( self.c[0] * self.T[0] )
            
        dm1 = -b[1:] / (self.c[1:] * self.T[:-1])
        dp1 = -b[1:] / (self.c[1:] * self.T[1:])
        A = np.diag(dm1,-1) + np.diag(d0,0) + np.diag(dp1,1)
        w,v = np.linalg.eig(A)
        return w,v
    
class Element:
    def __init__(self,model,Nparam,Nunknowns):
        self.model = model
        self.aq = None
	self.Nparam = Nparam  # Number of parameters
	self.Nunknowns = Nunknowns
	self.parameters = np.zeros( (self.Nparam,self.model.Np), 'D' )
    def initialize(self):
	'''Initialize element'''
	pass
    def potinf(self,x,y,aq=None):
	'''Returns 2D complex array of size (Naq,Np) or (Nparam,Naq,Np) if Nparam > 1'''
      	raise 'Must overload Element.potinf()'
    def potential(self,x,y,aq=None):
	'''Returns complex array of size (Naq,Np)'''
	if aq is None: aq = self.model.aq.findAquiferData(x,y)
        pot = self.potinf(x,y,aq)
        rv = self.parameters[0] * pot[0]
        for i in range(1,self.Nparam):
            rv += self.parameters[i] * pot[i]
        return rv
    def potinflayer(self,x,y,pylayer=0,aq=None):
	'''pylayer can be scalar, list, or array. returns array of size (Np, len(pylayer))'''
        if aq is None: aq = self.model.aq.findAquiferData( x, y )
        pot = self.potinf(x,y,aq)
        pylayer = np.atleast_1d(pylayer)
        rv = np.empty( (self.Nparam, len(pylayer), self.model.Np), 'D' )
        for i in range(self.Nparam):
            rv[i,:] = np.sum( pot[i] * aq.eigvec, 1 )[pylayer,:]
        rv = rv.swapaxes(0,1) # As the first axes needs to be the number of layers
        return rv
    def potentiallayer(self,x,y,pylayer=0,aq=None):
	'''Returns complex array of size (Np) or (len(pylayer),Np)'''
	if aq is None: aq = self.model.aq.findAquiferData(x,y)
        pot = self.potential(x,y,aq)
        phi = np.sum( pot * aq.eigvec, 1 )
        return phi[pylayer,:]
    def strengthinflayer(self):
        '''returns array of strengths of size (Nparam,Np)'''
        dis = self.dischargeinf()
        rv = np.empty( (self.Nparam, self.model.Np), 'D' )
        # the dischargeinf is already computed in such a way that dis[i] gives zero discharge in all layers except for self.pylayer[i]
        for i in range(self.Nparam):
            rv[i,:] = np.sum( dis[i] * self.aq.eigvec[self.pylayer[i]], 0 )
        return rv
    def strength(self,t):
        '''returns array of strengths of len(t) t must be ordered and tmin <= t <= tmax'''
        dis = self.dischargeinf()
        Qdis = self.parameters[0] * dis[0]
        for i in range(1,self.Nparam):
            Qdis += self.parameters[i] * dis[i]
        Qdis = np.sum( Qdis * self.aq.eigvec, 1 )
        t = np.atleast_1d(t)
        ml = self.model
        rv = np.zeros((self.aq.Naq,len(t)))
        it = 0
        for n in range(ml.Nin):
            if n == self.model.Nin-1:
                tp = t[ (t >= ml.tintervals[n]) & (t <= ml.tintervals[n+1]) ]
            else:
                tp = t[ (t >= ml.tintervals[n]) & (t < ml.tintervals[n+1]) ]
            Nt = len(tp)
            if Nt > 0:  # if all values zero, don't do the inverse transform
                for i in range(self.aq.Naq):
                    if np.abs( Qdis[i,n*ml.Npin:(n+1)*ml.Npin] ).min() > 1e-18:
                        rv[i,it:it+Nt] = invlaptrans.invlap( tp, ml.tintervals[n], ml.tintervals[n+1], Qdis[i,n*ml.Npin:(n+1)*ml.Npin], ml.gamma[n], ml.M, Nt )
                it = it + Nt
        return rv
    def layout(self):
	return '','',''
    def check(self,full_output=False):
        '''Prints data to verify solution'''
	if full_output:
            print 'Given element has no unknown parameters'
	return 0.0

class HeadEquation:
    def equation(self):
	'''Mix-in class that returns matrix rows for head-specified conditions
	Returns matrix part Np rows, Neq columns, complex'''
	mat = np.empty( (self.Nunknowns,self.model.Neq,self.model.Np), 'D' )
	rhs = np.empty( (self.Nunknowns, self.model.Np), 'D' )
        for i in range(self.Nunknowns):
            rhs[i,:] = self.pc[i] / self.model.p
	ieq = 0
	for e in self.model.elementList:
	    if e.Nunknowns > 0:
		mat[:,ieq:ieq+e.Nunknowns,:] = e.potinflayer(self.xc,self.yc,self.pylayer)
		ieq += e.Nunknowns
	    else:
		rhs -= e.potentiallayer(self.xc,self.yc,self.pylayer)
	return mat, rhs
    
class MscreenEquation:
    def equation(self):
	'''Mix-in class that returns matrix rows for multi-aquifer element with
        total given discharge and uniform but unknown head
	Returns matrix part Np rows, Neq columns, complex'''
	mat = np.zeros( (self.Nunknowns,self.model.Neq,self.model.Np), 'D' ) # Important to set to zero for some of the equations
	rhs = np.zeros( (self.Nunknowns, self.model.Np), 'D' )
        rhs[-1,:] = self.Qtot
	ieq = 0
	for e in self.model.elementList:
	    if e.Nunknowns > 0:
                head = e.potinflayer(self.xc,self.yc,self.pylayer) / self.aq.T[self.pylayer][:,np.newaxis,np.newaxis]
		mat[:-1,ieq:ieq+e.Nunknowns,:] = head[:-1,:,:] - head[1:,:,:]
                if e == self:
                    mat[-1,ieq:ieq+e.Nunknowns,:] = 1.0
		ieq += e.Nunknowns
	    else:
                head = e.potentiallayer(self.xc,self.yc,self.pylayer) / self.aq.T[self.pylayer][:,np.newaxis]
		rhs[:-1,:] -= head[:-1,:] - head[1:,:]
	return mat, rhs
    
class HconnEquation:
    def equation(self):
	'''Mix-in class that returns matrix rows for multi-aquifer element with
        which provides a connection between the layers with a specified resistance, but total net discharge is zero
	Returns matrix part Np rows, Neq columns, complex'''
	mat = np.zeros( (self.Nunknowns,self.model.Neq,self.model.Np), 'D' ) # Important to set to zero for some of the equations
	rhs = np.zeros( (self.Nunknowns, self.model.Np), 'D' )
        disinf = self.strengthinflayer()
	ieq = 0
	for e in self.model.elementList:
	    if e.Nunknowns > 0:
                head = e.potinflayer(self.xc,self.yc,self.pylayer) / self.aq.T[self.pylayer][:,np.newaxis,np.newaxis]
		mat[:-1,ieq:ieq+e.Nunknowns,:] = head[:-1,:,:] - head[1:,:,:]
                if e == self:
                    for i in range(self.Nunknowns-1):
                        mat[i,ieq:ieq+i+1,:] -= self.res[i] * disinf[i]
                    mat[-1,ieq:ieq+e.Nunknowns,:] = 1.0
		ieq += e.Nunknowns
	    else:
                head = e.potentiallayer(self.xc,self.yc,self.pylayer) / self.aq.T[self.pylayer][:,np.newaxis]
		rhs[:-1,:] -= head[:-1,:] - head[1:,:]
	return mat, rhs
    
class Well(Element):
    def __init__(self,model,xw=0,yw=0,rw=0.1,Q=0,layer=1):
        self.xw = float(xw); self.yw = float(yw); self.rw = float(rw)
	self.Q = np.atleast_1d(Q).astype('d')
        self.layer = np.atleast_1d(layer); self.pylayer = self.layer - 1
        if (len(self.Q) == 1) and (len(self.layer) > 1): self.Q = self.Q * np.ones(len(self.layer))
        Element.__init__(self,model,len(self.layer),0)
	self.name = 'Well'
        self.model.addElement(self)
    def __repr__(self):
	return self.name + ' at ' + str((self.xw,self.yw))
    def initialize(self):
	self.aq = self.model.aq.findAquiferData(self.xw,self.yw)
        self.coef = np.empty( (self.Nparam,self.aq.Naq,self.model.Nin,self.model.Npin), 'D' )
        for i in range(self.Nparam):
            self.parameters[i,:] = self.Q[i] 
            c = self.aq.coef[:,self.pylayer[i],:]
            c.shape = (self.aq.Naq,self.model.Nin,self.model.Npin)
            self.coef[i] = c
        self.coef2 = self.coef.reshape(self.Nparam,self.aq.Naq,self.model.Np) # has shape Nparam,Naq,Np
    def potinf(self,x,y,aq=None):
	'''Can be called with only one x,y value'''
        if aq is None: aq = self.model.aq.findAquiferData( x, y )
        rv = np.zeros((self.Nparam,aq.Naq,self.model.Nin,self.model.Npin),'D')
        if aq == self.aq:
            r = np.sqrt( (x-self.xw)**2 + (y-self.yw)**2 )
            pot = np.zeros(self.model.Npin,'D')
            if r < self.rw: r = self.rw  # If at well, set to at radius
            for i in range(self.aq.Naq):
                for j in range(self.model.Nin):
                    if r / abs(self.aq.lab2[i,j,0]) < 20.0:
                        #bessel.besselk0v( x-self.xw, y-self.yw, self.aq.lab2[i,j,:], pot)
                        bessel.k0besselv( r / self.aq.lab2[i,j,:], pot )
                        l = j*self.model.Npin
                        #for k in range(self.Nparam):
                        #    rv[k,i,j,:] = -1.0 / (2*np.pi*self.model.p[l:l+self.model.Npin]) * self.coef[k,i,j,:] * pot
                        rv[:,i,j,:] = -1.0 / (2*np.pi*self.model.p[l:l+self.model.Npin]) * self.coef[:,i,j,:] * pot
        rv.shape = (self.Nparam,aq.Naq,self.model.Np)
        return rv
    def dischargeinf(self):
        rv = np.zeros((self.Nparam,self.aq.Naq,self.model.Np),'D')
        for i in range(self.aq.Naq):
            rv[:,i,:] = 1.0 / self.model.p * self.coef2[:,i,:]
        return rv
    def layout(self):
        return 'point',self.xw,self.yw
    
class LineSink(Element):
    def __init__(self,model,x1=-1,y1=0,x2=1,y2=0,S=0,layer=1,addtomodel=True):
        self.x1 = float(x1); self.y1 = float(y1); self.x2 = float(x2); self.y2 = float(y2);
	self.S = np.atleast_1d(S).astype('d')
        self.layer = np.atleast_1d(layer); self.pylayer = self.layer - 1
        if (len(self.S) == 1) and (len(self.layer) > 1): self.S = self.S * np.ones(len(self.layer))
        Element.__init__(self,model,len(self.layer),0)
	self.name = 'LineSink'
        if addtomodel: self.model.addElement(self)
        self.xa,self.ya,self.xb,self.yb,self.np = np.zeros(1),np.zeros(1),np.zeros(1),np.zeros(1),np.zeros(1,'i')  # needed to call bessel.circle_line_intersection
    def __repr__(self):
	return self.name + ' from ' + str((self.x1,self.y1)) +' to '+str((self.x2,self.y2))
    def write(self):
	return self.name + '( '+str(self.model.name)+', x1=' +str(self.x1) +', y1=' + str(self.y1) +', x2=' +str(self.x2) +\
		', y2=' + str(self.y2) + ', S=' + str(self.S) + ')\n'
    def initialize(self):
	self.xc = 0.5*(self.x1+self.x2); self.yc = 0.5*(self.y1+self.y2)
        self.z1 = self.x1 + 1j*self.y1; self.z2 = self.x2 + 1j*self.y2
	self.aq = self.model.aq.findAquiferData(self.xc,self.yc)
        self.coef = np.empty( (self.Nparam,self.aq.Naq,self.model.Nin,self.model.Npin), 'D' )
        for i in range(self.Nparam):
            self.parameters[i,:] = self.S[i]  # Since only 1 parameter, float is ok
            c = self.aq.coef[:,self.pylayer[i],:]
            c.shape = (self.aq.Naq,self.model.Nin,self.model.Npin)
            self.coef[i] = c
        self.coef2 = self.coef.reshape(self.Nparam,self.aq.Naq,self.model.Np) # has shape Nparam,Naq,Np
    def potinf(self,x,y,aq=None):
	'''Can be called with only one x,y value'''
        if aq is None: aq = self.model.aq.findAquiferData( x, y )
        rv = np.zeros((self.Nparam,aq.Naq,self.model.Nin,self.model.Npin),'D')
        if aq == self.aq:
            pot = np.zeros(self.model.Npin,'D')
            for i in range(self.aq.Naq):
                for j in range(self.model.Nin):
                    bessel.circle_line_intersection(self.z1,self.z2,x+y*1j,20.0*abs(self.model.aq.lab2[i,j,0]),self.xa,self.ya,self.xb,self.yb,self.np)
                    if self.np > 0:
                        za = complex(self.xa,self.ya); zb = complex(self.xb,self.yb) # f2py has problem returning complex arrays -> fixed in new numpy
                        l = j*self.model.Npin
                        bessel.bessellsv(x,y,za,zb,self.aq.lab2[i,j,:],pot)
                        for k in range(self.Nparam):
                            rv[k,i,j,:] = self.coef[k,i,j,:] / self.model.p[l:l+self.model.Npin] * pot
        rv.shape = (self.Nparam,aq.Naq,self.model.Np)
        return rv
    def dischargeinf(self):
        rv = np.zeros((self.Nparam,self.aq.Naq,self.model.Np),'D')
        for i in range(self.aq.Naq):
            rv[:,i,:] = 1.0 / self.model.p * self.coef2[:,i,:]
        return rv
    def layout(self):
        return 'line', [self.x1,self.x2], [self.y1,self.y2]
        
class LineSinkDitch(Element):
    def __init__(self,model,x=[0,1,2],y=[0,1,2],coverw=0.0,Q=1,layer=1):
        Element.__init__(self,model,len(x)-1,len(x)-1)
        self.x = np.array(x); self.y = np.array(y)
        self.L = np.sqrt( (self.x[1:]-self.x[:-1])**2 + (self.y[1:]-self.y[:-1])**2 )
	self.xc = 0.5*(self.x[1:]+self.x[:-1]); self.yc = 0.5* (self.y[1:]+self.y[:-1])
	self.Nunknowns = len(self.x) - 1
	self.coverw = coverw
	self.Q = float(Q)
        self.layer = np.atleast_1d(layer); self.pylayer = self.layer * np.ones(self.Nparam,'i') - 1
	self.name = 'LineSinkDitch'
	self.lsList = []
	for i in range(self.Nunknowns):
	    self.lsList.append( LineSink(model,x[i],y[i],x[i+1],y[i+1],0.0,self.layer,False) )
        self.model.addElement(self)
    def __repr__(self):
	return self.name + ' from ' + str((self.x[0],self.y[0])) +' to '+str((self.x[-1],self.y[-1]))
    def write(self):
	return self.name + '( '+str(self.model.name)+', x=' +str(self.x.tolist()) +', y=' + str(self.y.tolist()) +\
		', coverw=' + str(self.coverw) + ', Q=' + str(self.Q.tolist()) + ')\n'
    def initialize(self):
	for ls in self.lsList: ls.initialize()
        self.aq = self.model.aq.findAquiferData(self.xc[0],self.yc[0])
    def potinf(self,x,y,aq=None):
        '''Returns array (Nunknowns,Nperiods)'''
        if aq is None: aq = self.model.aq.findAquiferData(x,y)
        rv = np.zeros( (self.Nparam,aq.Naq,self.model.Np), 'D' )
	for i in range(self.Nparam):
	    rv[i] = self.lsList[i].potinf(x,y,aq)
	return rv
    def dischargeinf(self):
        rv = np.zeros((self.Nparam,self.aq.Naq,self.model.Np),'D')
        for i in range(self.Nparam):
            rv[i] = self.lsList[i].dischargeinf()
        return rv
    def equation(self):
	mat = np.empty( (self.Nunknowns,self.model.Neq,self.model.Np), 'D' )
	rhs = np.zeros( (self.Nunknowns, self.model.Np), 'D' )
	# First fill matrix with regular resistance line-sink equations
	for icp in range(self.Nunknowns):
	    ieq = 0
	    for e in self.model.elementList:
		if e.Nunknowns > 0:
		    mat[icp,ieq:ieq+e.Nunknowns,:] = e.potinflayer(self.xc[icp],self.yc[icp],self.pylayer[icp])
		    if e == self:
			mat[icp,ieq+icp,:] -= self.lsList[icp].aq.T[self.pylayer[icp]] * self.coverw
		    ieq += e.Nunknowns
		else:
		    rhs[icp,:] -= e.potentiallayer(self.xc[icp],self.yc[icp],self.pylayer[icp])
	# Subtract row i+1 from row i
	for icp in range(self.Nunknowns-1):
	    mat[icp,:,:] -= mat[icp+1,:,:]
	    rhs[icp,:] -= rhs[icp+1,:]
	# Set last equations to be sum of discharges equals Q
	mat[-1,:,:] = 0.0; rhs[-1,:] = 0.0
	ieq = 0
	for e in self.model.elementList:
	    if e == self:
		mat[-1,ieq:ieq+self.Nunknowns,:] = self.strengthinflayer()
		rhs[-1,:] = self.Q / self.model.p
	    ieq += e.Nunknowns
    	return mat, rhs
    
class MscreenWell(Well,MscreenEquation):
    def __init__(self,model,xw=0,yw=0,rw=0.1,Qtot=0,layer=1):
        Well.__init__(self,model,xw,yw,rw,0.0,layer)
	self.Nunknowns = self.Nparam
        self.xc = self.xw + self.rw; self.yc = self.yw # To make sure the point is always the same for all elements
        self.Qtot  = Qtot
	self.name = 'MscreenWell'
    def initialize(self):
        Well.initialize(self)
    def check(self,full_output=False):
        h = self.model.phi(self.xc,self.yc) / self.aq.T[:,np.newaxis]
        maxerror = np.amax( np.abs( h[self.pylayer[:-1],:] - h[self.pylayer[1:],:] ) )
        maxerror = np.amax( maxerror, np.amax( np.abs( np.sum(self.parameters,0) - self.Qtot ) ) )
        print 'Error in Q ',np.amax( np.abs( np.sum(self.parameters,0) - self.Qtot ) )
	if full_output:
	    print self.name+' with control point at '+str((self.xc,self.yc))+' max error ',maxerror
	return maxerror

class HconnWell(Well,HconnEquation):
    def __init__(self,model,xw=0,yw=0,rw=0.1,res=0,layer=1):
        Well.__init__(self,model,xw,yw,rw,0.0,layer)
	self.Nunknowns = self.Nparam
        self.xc = self.xw + self.rw; self.yc = self.yw # To make sure the point is always the same for all elements
        self.res = np.atleast_1d(res)
	self.name = 'HconnWell'
    def initialize(self):
        Well.initialize(self)
        if len(self.res) == 1: self.res = self.res[0] * np.ones(self.Nparam-1)
        assert len(self.res) == self.Nparam-1, 'Length of res needs to be ' + str(len(Nparam)-1)
    def check(self,full_output=False):
        h = self.model.phi(self.xc,self.yc) / self.aq.T[:,np.newaxis]
        delh = h[:-1,:] - h[1:,:]
        disinf = self.strengthinflayer()
        maxerror = 0.0
        Q = np.zeros(self.model.Np,'D')
        for i in range(self.Nparam-1):
            Q += self.parameters[i,:]
            maxerror = np.amax( np.amax( np.abs( delh[i,:] - self.res[i] * Q ) ), maxerror )
        maxerror = np.amax( maxerror, np.amax( np.abs( np.sum(self.parameters,0) ) ) )
	if full_output:
	    print self.name+' with control point at '+str((self.xc,self.yc))+' max error ',maxerror
	return maxerror
    
class MscreenLineSink(LineSink,MscreenEquation):
    def __init__(self,model,x1=-1,y1=0,x2=1,y2=0,Stot=0,layer=1):
        LineSink.__init__(self,model,x1,y1,x2,y2,0.0,layer)
	self.Nunknowns = self.Nparam
        self.Stot = Stot
	self.name = 'MscreenLineSink'
    def initialize(self):
        LineSink.initialize(self)
        self.Qtot = self.Stot
    def check(self,full_output=False):
        h = self.model.phi(self.xc,self.yc) / self.aq.T[:,np.newaxis]
        maxerror = np.amax( np.abs( h[self.pylayer[:-1],:] - h[self.pylayer[1:],:] ) )
        maxerror = np.amax( maxerror, np.abs( np.sum(self.parameters) - self.Stot ) )
        print 'Error in S ',np.amax( np.abs( np.sum(self.parameters,0) - self.Stot ) )
	if full_output:
	    print self.name+' with control point at '+str((self.xc,self.yc))+' max error ',maxerror
	return maxerror
    
class HconnLineSink(LineSink,HconnEquation):
    def __init__(self,model,x1=-1,y1=0,x2=1,y2=0,res=0,layer=1):
        LineSink.__init__(self,model,x1,y1,x2,y2,0.0,layer)
	self.Nunknowns = self.Nparam
        self.res = np.atleast_1d(res)
	self.name = 'HconnLineSink'
    def initialize(self):
        LineSink.initialize(self)
        if len(self.res) == 1: self.res = self.res[0] * np.ones(self.Nparam-1)
        assert len(self.res) == self.Nparam-1, 'Length of res needs to be ' + str(len(Nparam)-1)
    def check(self,full_output=False):
        h = self.model.phi(self.xc,self.yc) / self.aq.T[:,np.newaxis]
        delh = h[:-1,:] - h[1:,:]
        maxerror = 0.0
        Q = np.zeros(self.model.Np,'D')
        for i in range(self.Nparam-1):
            Q += self.parameters[i,:]
            maxerror = np.amax( np.amax( np.abs( delh[i,:] - self.res[i] * Q ) ), maxerror )
        maxerror = np.amax( maxerror, np.amax( np.abs( np.sum(self.parameters,0) ) ) )
	if full_output:
	    print self.name+' with control point at '+str((self.xc,self.yc))+' max error ',maxerror
	return maxerror
    
class HeadWell(Well,HeadEquation):
    def __init__(self,model,xw=0,yw=0,rw=0.1,h=0,layer=1):
        Well.__init__(self,model,xw,yw,rw,0.0,layer)
	self.Nunknowns = self.Nparam
        self.xc = self.xw + self.rw; self.yc = self.yw # To make sure the point is always the same for all elements
        self.hc  = np.atleast_1d(h).astype('d')
	self.name = 'HeadWell'
    def initialize(self):
        Well.initialize(self)
	self.pc = self.aq.headToPotential(self.hc,self.pylayer)
    def check(self,full_output=False):
        maxerror = np.amax( np.abs( self.pc[:,np.newaxis] / self.model.p - self.model.phi(self.xc,self.yc)[self.pylayer,:] ) )
	if full_output:
	    print self.name+' with control point at '+str((self.xc,self.yc))+' max error ',maxerror
	return maxerror
    
class HeadLineSink(LineSink,HeadEquation):
    def __init__(self,model,x1=-1,y1=0,x2=1,y2=0,h=0,layer=1):
        LineSink.__init__(self,model,x1,y1,x2,y2,0.0,layer)
	self.Nunknowns = self.Nparam
        self.hc  = np.atleast_1d(h).astype('d')
	self.name = 'HeadLineSink'
    def initialize(self):
        LineSink.initialize(self)
	self.pc = self.aq.headToPotential(self.hc,self.pylayer)
    def check(self,full_output=False):
        maxerror = np.amax( np.abs( self.pc[:,np.newaxis] / self.model.p - self.model.phi(self.xc,self.yc)[self.pylayer,:] ) )
	if full_output:
	    print self.name+' with control point at '+str((self.xc,self.yc))+' max error ',maxerror
	return maxerror

    
def xsection(ml,x1=0,x2=1,y1=0,y2=0,N=100,t=1,layer=1,color=None,newfig=True,sendback=False):
    if newfig: plt.figure()
    x = np.linspace(x1,x2,N)
    y = np.linspace(y1,y2,N)
    s = np.sqrt( (x-x[0])**2 + (y-y[0])**2 )
    t = np.atleast_1d(t)
    pylayer = np.atleast_1d(layer) - 1
    h = np.zeros((len(x),len(pylayer),len(t)))
    for i in range(len(x)):
        h[i,:,:] = ml.head(x[i],y[i],t)[pylayer,:]
    for i in range(len(t)):
        for j in range(len(pylayer)):
            if color is None:
                plt.plot(s,h[:,j,i])
            else:
                plt.plot(s,h[:,j,i],color)
    if sendback:
        return s,h
    
def xsectionmovie(ml,x1,x2,y1=0,y2=0,N=100,t=1,layer=1,color=None,hmin=None,hmax=None,fname='timmovie',xt=40,yt=0.9):
    plt.figure()
    x = np.linspace(x1,x2,N)
    y = np.linspace(y1,y2,N)
    s = np.sqrt( (x-x[0])**2 + (y-y[0])**2 )
    t = np.atleast_1d(t)
    pylayer = np.atleast_1d(layer) - 1
    h = np.zeros((len(x),len(pylayer),len(t)))
    for i in range(len(x)):
        h[i,:,:] = ml.head(x[i],y[i],t)[pylayer,:]
    for i in range(len(t)):
        for j in range(len(pylayer)):
            plt.plot(s,h[:,j,i],color[j])
            plt.ylim(hmin,hmax)
            plt.text(xt,yt,'t=%.2f'%t[i])
            plt.savefig('movie/'+fname+str(1000+i)+'.png')
        plt.close()
        
def pycontour( ml, xmin, xmax, nx, ymin, ymax, ny, levels = 10, t=0.0, layer = 1,\
	       color = 'k', width = 0.5, style = 'solid',layout = True, newfig = True, \
	       labels = False, labelfmt = '%1.2f', fill=False, sendback = False):
    '''Contours head with pylab'''
    plt.rcParams['contour.negative_linestyle']='solid'
    # Compute grid
    xg,yg = np.meshgrid( np.linspace(xmin,xmax,nx), np.linspace(ymin,ymax,ny) ) 
    print 'grid of '+str((nx,ny))+'. gridding in progress. hit ctrl-c to abort'
    pot = np.zeros( ( ml.aq.Naq, ny, nx ), 'd' )
    t = np.atleast_1d(t)
    for irow in range(ny):
        for jcol in range(nx):
            pot[:,irow,jcol] = ml.head(xg[irow,jcol], yg[irow,jcol], t)[:,0]
    # Contour
    if type(levels) is list:
        levels = np.arange( levels[0],levels[1],levels[2] )
    elif levels == 'ask':
        print ' min,max: ',pot.min(),', ',pot.max(),'. Enter: hmin hmax step '
        input = raw_input().split()
        levels = np.arange(float(input[0]),float(input[1])+1e-8,float(input[2]))
    print 'Levels are ',levels
    # Colors
    if color is not None:
        color = [color]   
    if newfig:
        fig = plt.figure( figsize=(8,8) )
	ax = fig.add_subplot(111)
    else:
	fig = plt.gcf()
	ax = plt.gca()
    ax.set_aspect('equal','box')
    ax.set_xlim(xmin,xmax); ax.set_ylim(ymin,ymax)
    ax.set_autoscale_on(False)
    if layout: pylayout(ml,ax)
    if fill:
        a = ax.contourf( xg, yg, pot[layer-1], levels )
    else:
        if color is None:
            a = ax.contour( xg, yg, pot[layer-1], levels, linewidths = width, linestyles = style )
        else:
            a = ax.contour( xg, yg, pot[layer-1], levels, colors = color[0], linewidths = width, linestyles = style )
    if labels and not fill:
        ax.clabel(a,fmt=labelfmt)
    fig.canvas.draw()
    if sendback == 1: return a
    if sendback == 2: return xg,yg,pot
    
def pycontourmovie( ml, xmin, xmax, nx, ymin, ymax, ny, levels = 10, t=0.0, layer = 1,\
	       color = None, width = 0.5, style = 'solid',layout = True, newfig = True, \
	       labels = False, labelfmt = '%1.3f', fill=False, fname='timmovie',xt=0.0,yt=0.0):
    '''Contours head with pylab'''
    plt.rcParams['contour.negative_linestyle']='solid'
    # Compute grid
    xg,yg = np.meshgrid( np.linspace(xmin,xmax,nx), np.linspace(ymin,ymax,ny) ) 
    print 'grid of '+str((nx,ny))+'. gridding in progress. hit ctrl-c to abort'
    t = np.atleast_1d(t)
    hphi = np.zeros( ( ml.Np, ny, nx ), 'd' )
    pylayer = layer - 1
    for irow in range(ny):
        for jcol in range(nx):
            hphi[:,irow,jcol] = ml.phihead(xg[irow,jcol], yg[irow,jcol])[pylayer]
    # Contour
    # Colors
    h = np.zeros( (len(t),ny,nx) )
    for irow in range(ny):
        for jcol in range(nx):
            h[:,irow,jcol] = ml.inverseLapTran(hphi[:,irow,jcol],t)
    if color is not None:
        color = [color]   
    for i in range(len(t)):
        fig = plt.figure( figsize=(8,8) )
	ax = fig.add_subplot(111)
        ax.set_aspect('equal','box')
        ax.set_xlim(xmin,xmax); ax.set_ylim(ymin,ymax)
        ax.set_autoscale_on(False)
        if layout: pylayout(ml,ax,color='b',width=2)
        if fill:
            a = ax.contourf( xg, yg, h[i], levels )
        else:
            if color is None:
                a = ax.contour( xg, yg, h[i], levels, linewidths = width, linestyles = style )
            else:
                a = ax.contour( xg, yg, h[i], levels, colors = color[0], linewidths = width, linestyles = style )
        if labels and not fill:
            ax.clabel(a,fmt=labelfmt)
        plt.text(xt,yt,'t=%.2f'%t[i])
        fig.savefig('movie/'+fname+str(1000+i)+'.png')

    
def pylayout( ml, ax, color = 'k', overlay = 1, width = 0.5, style = '-' ):
    for e in ml.elementList:
        t,x,y = e.layout()
	if t == 'point':
	    ax.plot( [x], [y], color+'o', markersize=3 ) 
        if t == 'line':
	    ax.plot( x, y, color=color, ls = style, lw = width )
	if t == 'area':
            col = 0.7 + 0.2*np.random.rand()
            ax.fill( x, y, facecolor = [col,col,col], edgecolor = [col,col,col] )
            
from scipy.special import exp1,k0
def theis(r,t,Q,T,S):
    u = S * r**2 / ( 4 * T * t )
    rv = -Q/(4*np.pi*T) * exp1( u )
    return rv

def hantushkees(r,t,Q,T,S,c):
    lab = np.sqrt(T*c)
    rho = r / lab
    tau = np.log(2.0*(lab/r)*t/(c*S))
    w = ( exp1(rho) - k0(rho) ) / ( exp1(rho) - exp1(rho/2.0) )
    rv = np.zeros(len(t))
    neg = tau<=0
    rv[neg] = w * exp1( rho/2 * np.exp(-tau[neg]) ) - (w-1) * exp1( rho * np.cosh(tau[neg]) )
    pos = tau>=0
    rv[pos] = 2*k0(rho) - w * exp1( rho/2 * np.exp(tau[pos]) ) + (w-1) * exp1( rho * np.cosh(tau[pos]) )
    return -Q/(4*np.pi*T) * rv

#ml = ModelMaq(kaq=[1,2],z=[5,4,3,2,0],c=[100,200],Saq=[0.003,0.004],Sll=[1e-20,1e-20],topboundary='semi',phreatictop=False,tmin=0.01,tmax=10,M=20)
##ml = ModelMaq(kaq=[1,5.0],z=[14,9,5,2,0],c=[100,100],Saq=[0.003,0.003],Sll=[1e-20,1e-20],topboundary='semi',phreatictop=False,tmin=1,tmax=1000,M=20)
##ml = Model3D(kaq=[1,2.0,5.0],z=[9,5,2,0],kzoverkh=[.1,.1,.1],Saq=[0.3,0.003],Sll=[0.001],topboundary='imp',phreatictop=True,tmin=1,tmax=10,M=20)
##ml = TimModel(T=[1.0,5.0],c=[100],Saq=[0.3,0.003],Sll=[0.001],topboundary='imp',tmin=1,tmax=10.0,M=20)
#w = Well(ml,0.0,0,.1,1,[1])
#ml.solve()

##NEUMAN
#kaq = 1.0 * np.ones(11)
#z = np.arange(11.0,-1,-1); z[0] = 10.01
#S = 0.00001 * np.ones(11); S[0] = 0.1
##
#T = 10.0; Saq = S[-1] * 10
#sig = Saq / S[0]
#ts = 10**np.linspace(-1,5,40)
#t = ts * Saq * 10**2 / T
##
##ml = Model(k=k,z=z,S=S,kzoverkh=1.0,threed=True,tmin=t[0],tmax=t[-1],M=20)
#ml = Model3D(kaq,z,S,kzoverkh=1,phreatictop=True,tmin=t[0],tmax=t[-1],M=20)
#w = Well(ml,0,0,.1,np.ones(10),range(2,12))
##for i in range(2,12,1):
##    w = Well(ml,0,0,.1,1,i)
### Replace with constant head well
##w = MscreenWell(ml,0,0,.1,10,range(2,12))
#ml.solve()
#h = ml.potential(10,0,t)
#h = -h*4*np.pi*T/10.0
##plot(log10(ts),log10(h[-1]),'r')

