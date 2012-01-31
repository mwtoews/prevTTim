'''
Copyright (C), 2010-2012, Mark Bakker.
TTim is distributed under the MIT license
'''

import numpy as np
import matplotlib.pyplot as plt
from bessel import *
from invlap import *
from scipy.special import kv # Needed for K1 in Well class
from cmath import tanh as cmath_tanh

def ModelMaq(kaq=[1],z=[1,0],c=[],Saq=[],Sll=[],topboundary='imp',phreatictop=False,tmin=1,tmax=10,M=20):
    kaq = np.atleast_1d(kaq).astype('d')
    Naq = len(kaq)
    z = np.atleast_1d(z).astype('d')
    c = np.atleast_1d(c).astype('d')
    Saq = np.atleast_1d(Saq).astype('d')
    Sll = np.atleast_1d(Sll).astype('d')
    H = z[:-1] - z[1:]
    assert np.all(H >= 0), 'Error: Not all layer thicknesses are non-negative' + str(H) 
    if topboundary[:3] == 'imp':
        assert len(z) == 2*Naq, 'Error: Length of z needs to be ' + str(2*Naq)
        assert len(c) == Naq-1, 'Error: Length of c needs to be ' + str(Naq-1)
        assert len(Saq) == Naq, 'Error: Length of Saq needs to be ' + str(Naq)
        assert len(Sll) == Naq-1, 'Error: Length of Sll needs to be ' + str(Naq-1)
        Haq = H[::2]
        Saq = Saq * Haq
        if phreatictop: Saq[0] = Saq[0] / H[0]
        Sll = Sll * H[1::2]
        c = np.hstack((np.nan,c))
        Sll = np.hstack((np.nan,Sll))
    else: # leaky layer on top
        assert len(z) == 2*Naq+1, 'Error: Length of z needs to be ' + str(2*Naq+1)
        assert len(c) == Naq, 'Error: Length of c needs to be ' + str(Naq)
        assert len(Saq) == Naq, 'Error: Length of Saq needs to be ' + str(Naq)
        assert len(Sll) == Naq, 'Error: Length of Sll needs to be ' + str(Naq)
        Haq = H[1::2]
        Saq = Saq * Haq
        Sll = Sll * H[::2]
        if phreatictop and (topboundary[:3]=='lea'): Sll[0] = Sll[0] / H[0]
    return TimModel(kaq,Haq,c,Saq,Sll,topboundary,tmin,tmax,M)
    
class TimModel:
    def __init__(self,kaq=[1,1],Haq=[1,1],c=[np.nan,100],Saq=[0.3,0.003],Sll=[1e-3],topboundary='imp',tmin=1,tmax=10,M=20):
        self.elementList = []
        self.vbcList = []  # List with variable boundary condition 'v' elements
        self.zbcList = []  # List with zero and constant boundary condition 'z' elements
        self.gbcList = []  # List with given boundary condition 'g' elements; given bc elements don't have any unknowns
        self.tmin = float(tmin)
        self.tmax = float(tmax)
        self.M = M
        self.aq = Aquifer(self,kaq,Haq,c,Saq,Sll,topboundary)
        self.compute_laplace_parameters()
        self.name = 'TimModel'
        bessel.initialize()
    def __repr__(self):
        return 'Model'
    def initialize(self):
        self.gvbcList = self.gbcList + self.vbcList
        self.elementList = self.gbcList + self.vbcList + self.zbcList  # Given elements are first in list
        self.Ngbc = len(self.gbcList)
        self.Nvbc = len(self.vbcList)
        self.Nzbc = len(self.zbcList)
        self.Ngvbc = self.Ngbc + self.Nvbc
        self.aq.initialize()
        for e in self.elementList:
            e.initialize()
    def addElement(self,e):
        if e.type == 'g':
            self.gbcList.append(e)
        elif e.type == 'v':
            self.vbcList.append(e)
        elif e.type == 'z':
            self.zbcList.append(e)
    def compute_laplace_parameters(self):
        '''
        Nin: Number of time intervals
        Npin: Number of p values per interval
        Np: Total number of p values (Nin*Np)
        p[Np]: Array with p values
        '''
        itmin = int(np.floor(np.log10(self.tmin)))
        itmax = int(np.ceil(np.log10(self.tmax)))
        self.tintervals = np.arange(itmin,itmax+1)
        self.tintervals = 10.0**self.tintervals
        #alpha = 1.0
        alpha = 0.0  # I don't see why it shouldn't be 0.0
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
    def potential(self,x,y,t,pylayer=None,aq=None,derivative=0):
        '''Returns pot[Naq,Ntimes] if layer=None, otherwise pot[len(pylayer,Ntimes)]
        t must be ordered '''
        #
        if derivative > 0:
            print 'derivative > 0 not yet implemented in potentialnew'
            return
        #
        if aq is None: aq = self.aq.findAquiferData(x,y)
        if pylayer is None: pylayer = range(aq.Naq)
        Nlayer = len(pylayer)
        time = np.atleast_1d(t).copy()
        pot = np.zeros((self.Ngvbc, aq.Naq, self.Np),'D')
        for i in range(self.Ngbc):
            pot[i,:] += self.gbcList[i].unitpotential(x,y,aq)
        for e in self.vbcList:
            pot += e.potential(x,y,aq)
        for e in self.zbcList:
            pot += e.potential(x,y,aq)
        if pylayer is None:
            pot = np.sum( pot[:,np.newaxis,:,:] * aq.eigvec, 2 )
        else:
            pot = np.sum( pot[:,np.newaxis,:,:] * aq.eigvec[pylayer,:], 2 )
        rv = np.zeros((Nlayer,len(time)))
        if (time[0] < self.tmin) or (time[-1] > self.tmax): print 'Warning, some of the times are smaller than tmin or larger than tmax; zeros are substituted'
        #
        for k in range(self.Ngvbc):
            e = self.gvbcList[k]
            for itime in range(e.Ntstart):
                t = time - e.tstart[itime]
                it = 0
                if t[-1] >= self.tmin:  # Otherwise all zero
                    if (t[0] < self.tmin):
                        it = np.argmax( t >= self.tmin )  # clever call that should be replaced with find_first function when included in numpy
                    for n in range(self.Nin):
                        if n == self.Nin-1:
                            tp = t[ (t >= self.tintervals[n]) & (t <= self.tintervals[n+1]) ]
                        else:
                            tp = t[ (t >= self.tintervals[n]) & (t < self.tintervals[n+1]) ]
                        Nt = len(tp)
                        if Nt > 0:  # if all values zero, don't do the inverse transform
                            for i in range(Nlayer):
                                if np.abs( pot[k,i,n*self.Npin] ) > 1e-20:  # First value very small
                                    if not np.any( pot[k,i,n*self.Npin:(n+1)*self.Npin] == 0.0) : # If there is a zero item, zero should be returned; funky enough this can be done with a straight equal comparison
                                        rv[i,it:it+Nt] += e.bc[itime] * invlaptrans.invlap( tp, self.tintervals[n], self.tintervals[n+1], pot[k,i,n*self.Npin:(n+1)*self.Npin], self.gamma[n], self.M, Nt )
                            it = it + Nt
        return rv
    ### DOESN'T WORK RIGHT NOW
    #def phi(self,x,y,aq=None):
    #    '''Returns the potential in Laplace space pot[Naq,Np] Used in check functions'''
    #    if aq is None: aq = self.aq.findAquiferData(x,y)
    #    pot = np.zeros((self.Nvbc,aq.Naq,self.Np),'D')
    #    for e in self.elementList:
    #        pot += e.potential(x,y,aq)
    #    pot = np.sum( pot[:,np.newaxis,:,:] * aq.eigvec, 2 )
    #    return pot
    def head(self,x,y,t,layer=None,aq=None,derivative=0):
        if aq is None: aq = self.aq.findAquiferData(x,y)
        if layer is None:
            pylayer = range(aq.Naq)
        else:
            pylayer = np.atleast_1d(layer) - 1
        pot = self.potential(x,y,t,pylayer,aq,derivative)
        return aq.potentialToHead(pot,pylayer)
    #def headgrid(self,x1,x2,nx,y1,y2,ny,t):
    #    '''Returns h[Naq,Ntimes,Ny,Nx]'''
    #    xg,yg = np.meshgrid( np.linspace(x1,x2,nx), np.linspace(y1,y2,ny) ) 
    #    t = np.atleast_1d(t)
    #    hphi = np.zeros( ( self.aq.Naq, self.Np, ny, nx ), 'D' )
    #    for irow in range(ny):
    #        for jcol in range(nx):
    #            hphi[:,:,irow,jcol] = self.phihead(xg[irow,jcol], yg[irow,jcol])
    #    # Contour
    #    # Colors
    #    h = np.zeros( (self.aq.Naq,len(t),ny,nx) )
    #    for irow in range(ny):
    #        for jcol in range(nx):
    #            for k in range(self.aq.Naq):
    #                h[k,:,irow,jcol] = self.inverseLapTran(hphi[k,:,irow,jcol],t)
    #    return h
    def headalongline(self,x,y,t,layer=None):
        '''Returns head[Nlayer,len(t),len(x)]
        Assumes same number of layers for each x and y
        layers may be None or list of layers for which head is computed'''
        xg,yg = np.atleast_1d(x),np.atleast_1d(y)
        if layer is None:
            Nlayer = self.aq.findAquiferData(xg[0],yg[0]).Naq
        else:
            Nlayer = len(np.atleast_1d(layer))
        nx = len(xg)
        if len(yg) == 1:
            yg = yg * np.ones(nx)
        t = np.atleast_1d(t)
        h = np.zeros( (Nlayer,len(t),nx) )
        for i in range(nx):
            h[:,:,i] = self.head(xg[i],yg[i],t,layer)
        return h
    def inverseLapTran(self,pot,t):
        '''returns array of potentials of len(t)
        t must be ordered and tmin <= t <= tmax'''
        t = np.atleast_1d(t)
        rv = np.zeros(len(t))
        it = 0
        if t[-1] >= self.tmin:  # Otherwise all zero
            if (t[0] < self.tmin): it = np.argmax( t >= self.tmin )  # clever call that should be replaced with find_first function when included in numpy
            for n in range(self.Nin):
                if n == self.Nin-1:
                    tp = t[ (t >= self.tintervals[n]) & (t <= self.tintervals[n+1]) ]
                else:
                    tp = t[ (t >= self.tintervals[n]) & (t < self.tintervals[n+1]) ]
                Nt = len(tp)
                if Nt > 0:  # if all values zero, don't do the inverse transform
                    if np.abs( pot[n*self.Npin] ) > 1e-20:
                        if not np.any( pot[n*self.Npin:(n+1)*self.Npin] == 0.0) : # If there is a zero item, zero should be returned; funky enough this can be done with a straight equal comparison
                            rv[it:it+Nt] = invlaptrans.invlap( tp, self.tintervals[n], self.tintervals[n+1], pot[n*self.Npin:(n+1)*self.Npin], self.gamma[n], self.M, Nt )
                    it = it + Nt
        return rv
    def solve(self,printmat = 0,sendback=0):
        '''Compute solution'''
        # Initialize elements
        self.initialize()
        # Compute number of equations
        self.Neq = np.sum( [e.Nunknowns for e in self.elementList] )
        print 'self.Neq ',self.Neq
        if self.Neq == 0:
            print 'No unknowns. Solution complete'
            return
        mat = np.empty( (self.Neq,self.Neq,self.Np), 'D' )
        rhs = np.empty( (self.Neq,self.Ngvbc,self.Np), 'D' )
        ieq = 0
        for e in self.elementList:
            if e.Nunknowns > 0:
                mat[ ieq:ieq+e.Nunknowns, :, : ], rhs[ ieq:ieq+e.Nunknowns, :, : ] = e.equation()
                ieq += e.Nunknowns
        if printmat:
            print 'mat ',mat
            print 'rhs ',rhs
        for i in range( self.Np ):
            sol = np.linalg.solve( mat[:,:,i], rhs[:,:,i] )
            icount = 0
            for e in self.elementList:
                for j in range(e.Nunknowns):
                    e.parameters[:,j,i] = sol[icount,:]
                    icount += 1
        print 'solution complete'
        if sendback:
            return sol
        return
    #def solve(self,printmat = 0,sendback=0):
    #    '''Compute solution'''
    #    # Initialize elements
    #    self.initialize()
    #    # Compute number of equations
    #    self.Neq = np.sum( [e.Nunknowns for e in self.elementList] )
    #    print 'self.Neq ',self.Neq
    #    if self.Neq == 0:
    #        print 'No unknowns. Solution complete'
    #        return
    #    mat = np.empty( (self.Neq,self.Neq,self.Np), 'D' )
    #    rhs = np.empty( (self.Ngvbc,self.Neq,self.Np), 'D' )
    #    ieq = 0
    #    for e in self.elementList:
    #        if e.Nunknowns > 0:
    #            mat[ ieq:ieq+e.Nunknowns, :, : ], rhs[ :, ieq:ieq+e.Nunknowns, : ] = e.equation()
    #            ieq += e.Nunknowns
    #    if printmat:
    #        print 'mat ',mat
    #        print 'rhs ',rhs
    #    for i in range( self.Np ):
    #        # This is ugly. I really would like rhs[Nvbc,Neq,Np], but how to do that with potentiallayer?
    #        sol = np.linalg.solve( mat[:,:,i], np.transpose(rhs[:,:,i]) )
    #        icount = 0
    #        for e in self.elementList:
    #            for j in range(e.Nunknowns):
    #                e.parameters[:,j,i] = sol[icount,:]
    #                icount += 1
    #    print 'solution complete'
    #    if sendback:
    #        return sol
    #    return
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
    def __init__(self,model,kaq,Haq,c,Saq,Sll,topboundary):
        self.model = model
        self.kaq = np.atleast_1d(kaq).astype('d')
        self.Naq = len(kaq)
        self.Haq = np.atleast_1d(Haq).astype('d')
        self.T = self.kaq * self.Haq
        self.Tcol = self.T.reshape(self.Naq,1)
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
        '''
        eigval[Naq,Np]: Array with eigenvalues
        lab[Naq,Np]: Array with lambda values
        lab2[Naq,Nin,Npin]: Array with lambda values reorganized per interval
        eigvec[Naq,Naq,Np]: Array with eigenvector matrices
        coef[Naq,Naq,Np]: Array with coefficients;
        coef[ipylayer,:,np] are the coefficients if the element is in ipylayer belonging to Laplace parameter number np
        '''
        self.eigval = np.zeros((self.Naq,self.model.Np),'D')
        self.lab = np.zeros((self.Naq,self.model.Np),'D')
        self.eigvec = np.zeros((self.Naq,self.Naq,self.model.Np),'D')
        self.coef = np.zeros((self.Naq,self.Naq,self.model.Np),'D')
        b = np.diag(np.ones(self.Naq))
        for i in range(self.model.Np):
            w,v = self.compute_lab_eigvec(self.model.p[i]) # Eigenvectors are columns of v
            index = np.argsort( abs(w) )[::-1]
            w = w[index]; v = v[:,index]
            self.eigval[:,i] = w; self.eigvec[:,:,i] = v
            self.coef[:,:,i] = np.linalg.solve( v, b ).T
        self.lab = 1.0 / np.sqrt(self.eigval)
        self.lab2 = self.lab.copy(); self.lab2.shape = (self.Naq,self.model.Nin,self.model.Npin)
    def findAquiferData(self,x,y):
        return self
    def headToPotential(self,h,pylayer):
        return h * self.Tcol[pylayer]
    def potentialToHead(self,p,pylayer):
        return p / self.Tcol[pylayer]
    def compute_lab_eigvec(self,p):
        sqrtpSc = np.sqrt( p * self.Sll * self.c )
        a, b = np.zeros_like(sqrtpSc), np.zeros_like(sqrtpSc)
        small = np.abs(sqrtpSc) < 200
        a[small] = sqrtpSc[small] / np.tanh(sqrtpSc[small])
        b[small] = sqrtpSc[small] / np.sinh(sqrtpSc[small])
        a[~small] = sqrtpSc[~small] / ( (1.0 - np.exp(-2.0*sqrtpSc[~small])) / (1.0 + np.exp(-2.0*sqrtpSc[~small])) )
        b[~small] = sqrtpSc[~small] * 2.0 * np.exp(-sqrtpSc[~small]) / (1.0 - np.exp(-2.0*sqrtpSc[~small]))
        if (self.topboundary == 'sem') or (self.topboundary == 'lea'):
            if abs(sqrtpSc[0]) < 200:
                dzero = sqrtpSc[0] * np.tanh( sqrtpSc[0] )
            else:
                dzero = sqrtpSc[0] * cmath_tanh( sqrtpSc[0] )  # Bug in complex tanh in numpy
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
    def __init__(self, model, Nparam=1, Nunknowns=0, layer=1, tstart=0, bcin=0, type='z', name='', Rzero=20.0):
        '''Types of elements
        'v': boundary condition is variable through time
        'z': boundary condition is zero through time
        '''
        self.model = model
        self.aq = None # Set in the initialization function
        self.Nparam = Nparam  # Number of parameters
        self.Nunknowns = Nunknowns
        self.layer = np.atleast_1d(layer)
        self.pylayer = self.layer - 1
        self.Nlayer = len(self.layer)
        #
        self.tstart = np.array(tstart,dtype=float)
        self.bcin = np.atleast_1d(bcin).astype('d') # atleast_1d has no keyword dtype; self.bc is 1D. Only one bc per element.
        if self.tstart[0] > 0:
            self.tstart = np.hstack((np.zeros(1),self.tstart))
            self.bcin = np.hstack((np.zeros(1),self.bcin))
        #
        self.type = type  # 'z' boundary condition through time or 'v' boundary condition through time
        self.name = name
        self.Rzero = Rzero
    def setbc(self):
        if len(self.tstart) > 1:
            self.bc = np.zeros_like(self.bcin)
            self.bc[0] = self.bcin[0]
            self.bc[1:] = self.bcin[1:] - self.bcin[:-1]
        else:
            self.bc = self.bcin.copy()
        self.Ntstart = len(self.tstart)
    def initialize(self):
        '''Initialization of terms that cannot be initialized before other elements or the aquifer is defined.
        As we don't want to require a certain order of entering elements, these terms are initialized when Model.solve is called 
        The initialization class needs to be overloaded by all derived classes'''
        pass
    def potinf(self,x,y,aq=None):
        '''Returns complex array of size (Nparam,Naq,Np)'''
        raise 'Must overload Element.potinf()'
    def potential(self,x,y,aq=None):
        '''Returns complex array of size (Ngvbc,Naq,Np)
        Can be more efficient for given elements'''
        if aq is None: aq = self.model.aq.findAquiferData(x,y)
        return np.sum( self.parameters[:,:,np.newaxis,:] * self.potinf(x,y,aq), 1 )
    def unitpotential(self,x,y,aq=None):
        '''Returns complex array of size (Naq,Np)
        Can be more efficient for given elements'''
        if aq is None: aq = self.model.aq.findAquiferData(x,y)
        return np.sum( self.potinf(x,y,aq), 0 )
    # Functions used to build equations
    def potinflayer(self,x,y,pylayer=0,aq=None):
        '''pylayer can be scalar, list, or array. returns array of size (len(pylayer),Nparam,Np)
        only used in building equations'''
        if aq is None: aq = self.model.aq.findAquiferData( x, y )
        pot = self.potinf(x,y,aq)
        rv = np.sum( pot[:,np.newaxis,:,:] * aq.eigvec, 2 )
        rv = rv.swapaxes(0,1) # As the first axes needs to be the number of layers
        return rv[pylayer,:]
    def potentiallayer(self,x,y,pylayer=0,aq=None):
        '''Returns complex array of size (Ngvbc,len(pylayer),Np)
        only used in building equations'''
        if aq is None: aq = self.model.aq.findAquiferData(x,y)
        pot = self.potential(x,y,aq)
        phi = np.sum( pot[:,np.newaxis,:,:] * aq.eigvec, 2 )
        return phi[:,pylayer,:]
    def unitpotentiallayer(self,x,y,pylayer=0,aq=None):
        '''Returns complex array of size (len(pylayer),Np)
        only used in building equations'''
        if aq is None: aq = self.model.aq.findAquiferData(x,y)
        pot = self.unitpotential(x,y,aq)
        phi = np.sum( pot[np.newaxis,:,:] * aq.eigvec, 1 )
        return phi[pylayer,:]
    def headinside(self,t):
        print "This function not implemented for this element"
        return
    def layout(self):
        return '','',''
    def check(self,full_output=False):
        '''Prints data to verify solution'''
        if full_output:
            print 'Given element has no unknown parameters'
        return 0.0
    
class VbcHeadEquation:
    def equation(self):
        '''Mix-in class that returns matrix rows for head-specified conditions.
        Works for Nunknowns = 1
        Returns matrix part Nunknowns,Neq,Np, complex
        Returns rhs part Nunknowns,Nvbc,Np, complex'''
        mat = np.empty( (self.Nunknowns,self.model.Neq,self.model.Np), 'D' )
        rhs = np.zeros( (self.Nunknowns,self.model.Ngvbc,self.model.Np), 'D' )  # Needs to be initialized to zero
        ieq = 0
        for e in self.model.elementList:
            if e.Nunknowns > 0:
                mat[:,ieq:ieq+e.Nunknowns,:] = e.potinflayer(self.xc,self.yc,self.pylayer)
                ieq += e.Nunknowns
        for i in range(self.model.Ngbc):
            rhs[:,i,:] -= self.model.gbcList[i].unitpotentiallayer(self.xc,self.yc,self.pylayer)  # Pretty cool that this works, really
        if self.type == 'v':
            iself = self.model.vbcList.index(self)
            for i in range(self.Nunknowns):
                rhs[i,self.model.Ngbc+iself,:] = self.pc[i] / self.model.p
        return mat, rhs
    
#class VbcHeadEquation:
#    def equation(self):
#        '''Mix-in class that returns matrix rows for head-specified conditions.
#        Works for Nunknowns = 1
#        Returns matrix part Nunknowns,Neq,Np, complex
#        Returns rhs part Nunknowns,Nvbc,Np, complex'''
#        mat = np.empty( (self.Nunknowns,self.model.Neq,self.model.Np), 'D' )
#        rhs = np.zeros( (self.model.Ngvbc,self.Nunknowns,self.model.Np), 'D' )  # Needs to be initialized to zero
#        ieq = 0
#        for e in self.model.elementList:
#            if e.Nunknowns > 0:
#                mat[:,ieq:ieq+e.Nunknowns,:] = e.potinflayer(self.xc,self.yc,self.pylayer)
#                ieq += e.Nunknowns
#        for i in range(self.model.Ngbc):
#            rhs[i,:] -= self.model.gbcList[i].unitpotentiallayer(self.xc,self.yc,self.pylayer)
#        if self.type == 'v':
#            iself = self.model.vbcList.index(self)
#            for i in range(self.Nunknowns):
#                rhs[self.model.Ngbc+iself,i,:] = self.pc[i] / self.model.p
#        return mat, rhs
    
class WellBase(Element):
    '''Variable boundary condition well. May be screened in multiple layers, but each has the same bc'''
    def __init__(self,model,xw=0,yw=0,rw=0.1,tstart=[0],bcin=[1.0],layer=1,type='',name='WellBase',Rzero=20.0):
        Element.__init__(self, model, Nparam=1, Nunknowns=0, layer=layer, tstart=tstart, bcin=bcin, type=type, name=name, Rzero=Rzero)
        self.Nparam = len(self.pylayer)
        self.xw = float(xw); self.yw = float(yw); self.rw = float(rw)
        self.model.addElement(self)
    def __repr__(self):
        return self.name + ' at ' + str((self.xw,self.yw))
    def initialize(self):
        self.aq = self.model.aq.findAquiferData(self.xw,self.yw)
        self.setbc()
        coef = self.aq.coef[self.pylayer,:]
        laboverrwk1 = self.aq.lab / (self.rw * kv(1,self.rw/self.aq.lab))
        self.setflowcoef()
        self.term = -1.0 / (2*np.pi) * laboverrwk1 * self.flowcoef * coef
        self.term2 = self.term.reshape(self.Nparam,self.aq.Naq,self.model.Nin,self.model.Npin)
    def setflowcoef(self):
        '''Separate function so that this can be overloaded for other types'''
        self.flowcoef = 1.0 / self.model.p  # Step function
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
                    if r / abs(self.aq.lab2[i,j,0]) < self.Rzero:
                        bessel.k0besselv( r / self.aq.lab2[i,j,:], pot )
                        rv[:,i,j,:] = self.term2[:,i,j,:] * pot
        rv.shape = (self.Nparam,aq.Naq,self.model.Np)
        return rv
    def headinside(self,t):
        return self.model.head(self.xc,self.yc,t)[self.pylayer]
    def layout(self):
        return 'point',self.xw,self.yw
    
class Well(WellBase):
    '''Well with non-zero and potentially variable discharge through time'''
    def __init__(self,model,xw=0,yw=0,rw=0.1,tstart=[0],Q=[1.0],layer=1,Rzero=20.0):
        WellBase.__init__(self,model,xw,yw,rw,tstart=tstart,bcin=Q,layer=layer,type='g',name='Well',Rzero=Rzero)
    
class ZeroHeadWell(WellBase,VbcHeadEquation):
    '''HeadWell that remains zero and constant through time'''
    def __init__(self,model,xw=0,yw=0,rw=0.1,layer=1,Rzero=20.0):
        WellBase.__init__(self,model,xw,yw,rw,tstart=[0.0],bcin=[0.0],layer=layer,type='z',name='ZeroHeadWell',Rzero=Rzero)
        self.Nunknowns = self.Nparam
        self.xc = self.xw + self.rw; self.yc = self.yw # To make sure the point is always the same for all elements
    def initialize(self):
        WellBase.initialize(self)
        self.parameters = np.zeros( (self.model.Ngvbc, self.Nparam, self.model.Np), 'D' )
    def check(self,full_output=False):
        maxerror = np.amax( np.abs( self.pc[:,np.newaxis] / self.model.p - self.model.phi(self.xc,self.yc)[self.pylayer,:] ) )
        if full_output:
            print self.name+' with control point at '+str((self.xc,self.yc))+' max error ',maxerror
        return maxerror
    
class HeadWell(WellBase,VbcHeadEquation):
    '''HeadWell that varies through time. May be screened in multiple layers but all with the same head'''
    def __init__(self,model,xw=0,yw=0,rw=0.1,tstart=[0.0],h=[1.0],layer=1,Rzero=20.0):
        WellBase.__init__(self,model,xw,yw,rw,tstart=tstart,bcin=h,layer=layer,type='v',name='HeadWell',Rzero=Rzero)
        self.Nparam = len(self.pylayer)
        self.Nunknowns = self.Nparam
        self.xc = self.xw + self.rw; self.yc = self.yw # To make sure the point is always the same for all elements
        self.name = 'HeadWell'
    def initialize(self):
        WellBase.initialize(self)
        self.parameters = np.zeros( (self.model.Ngvbc, self.Nparam, self.model.Np), 'D' )
        self.pc = self.aq.T[self.pylayer] # Needed in solving; We solve for a unit head
    def check(self,full_output=False):
        maxerror = np.amax( np.abs( self.pc[:,np.newaxis] / self.model.p - self.model.phi(self.xc,self.yc)[self.pylayer,:] ) )
        if full_output:
            print self.name+' with control point at '+str((self.xc,self.yc))+' max error ',maxerror
        return maxerror
    
def xsection(ml,x1=0,x2=1,y1=0,y2=0,N=100,t=1,layer=1,color=None,lw=1,newfig=True):
    if newfig: plt.figure()
    x = np.linspace(x1,x2,N)
    y = np.linspace(y1,y2,N)
    s = np.sqrt( (x-x[0])**2 + (y-y[0])**2 )
    h = ml.headalongline(x,y,t,layer)
    Nlayer,Ntime,Nx = h.shape
    for i in range(Nlayer):
        for j in range(Ntime):
            if color is None:
                plt.plot(s,h[i,j,:],lw=lw)
            else:
                plt.plot(s,h[i,j,:],color,lw=lw)

##########################################
    
ml = ModelMaq(kaq=[10,5],z=[4,2,1,0],c=[100],Saq=[1e-3,1e-4],Sll=[1e-6],tmin=0.1,tmax=10,M=15)
w1 = Well(ml,0,0,.1,tstart=[0,1],Q=[5,2],layer=[1,2])
w2 = Well(ml,100,0,.1,tstart=[0,2],Q=[3,7],layer=[1])
w3 = HeadWell(ml,0,100,.1,tstart=[0,3],h=[2,1],layer=[1,2])
#w3 = ZeroHeadWell(ml,0,100,.1,layer=[1,2])
ml.solve()
#print ml.potential(2,3,[.5,5])
print ml.potential(50,50,[0.5,5])
#print ml.potential(w3.xc,w3.yc,[.5,5])