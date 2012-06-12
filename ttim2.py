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
import inspect # Used for storing the input
import os

class TimModel:
    def __init__(self,kaq=[1,1],Haq=[1,1],c=[np.nan,100],Saq=[0.3,0.003],Sll=[1e-3],topboundary='imp',tmin=1,tmax=10,M=20):
        self.elementList = []
        self.elementDict = {}
        self.vbcList = []  # List with variable boundary condition 'v' elements
        self.zbcList = []  # List with zero and constant boundary condition 'z' elements
        self.gbcList = []  # List with given boundary condition 'g' elements; given bc elements don't have any unknowns
        self.tmin = float(tmin)
        self.tmax = float(tmax)
        self.M = M
        self.aq = Aquifer(self,kaq,Haq,c,Saq,Sll,topboundary)
        self.compute_laplace_parameters()
        self.name = 'TimModel'
        self.modelname = 'ml' # Used for writing out input
        bessel.initialize()
    def __repr__(self):
        return 'Model'
    def initialize(self):
        self.gvbcList = self.gbcList + self.vbcList
        self.vzbcList = self.vbcList + self.zbcList
        self.elementList = self.gbcList + self.vbcList + self.zbcList  # Given elements are first in list
        self.Ngbc = len(self.gbcList)
        self.Nvbc = len(self.vbcList)
        self.Nzbc = len(self.zbcList)
        self.Ngvbc = self.Ngbc + self.Nvbc
        self.aq.initialize()
        for e in self.elementList:
            e.initialize()
    def addElement(self,e):
        if e.label is not None: self.elementDict[e.label] = e
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
        itmin = np.floor(np.log10(self.tmin))
        itmax = np.ceil(np.log10(self.tmax))
        self.tintervals = 10.0**np.arange(itmin,itmax+1)
        # lower and upper limit are adjusted to prevent any problems from t exactly at the beginning and end of the interval
        # also, you cannot count on t >= 10**log10(t) for all possible t
        self.tintervals[0] = self.tintervals[0] * ( 1 - np.finfo(float).epsneg )
        self.tintervals[-1] = self.tintervals[-1] * ( 1 + np.finfo(float).eps )
        #alpha = 1.0
        alpha = 0.0  # I don't see why it shouldn't be 0.0
        tol = 1e-9
        self.Nin = len(self.tintervals)-1
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
    def potential(self,x,y,t,pylayers=None,aq=None,derivative=0):
        '''Returns pot[Naq,Ntimes] if layers=None, otherwise pot[len(pylayers,Ntimes)]
        t must be ordered '''
        #
        if derivative > 0:
            print 'derivative > 0 not yet implemented in potential'
            return
        #
        if aq is None: aq = self.aq.findAquiferData(x,y)
        if pylayers is None: pylayers = range(aq.Naq)
        Nlayers = len(pylayers)
        time = np.atleast_1d(t).copy()
        pot = np.zeros((self.Ngvbc, aq.Naq, self.Np),'D')
        for i in range(self.Ngbc):
            pot[i,:] += self.gbcList[i].unitpotential(x,y,aq)
        for e in self.vzbcList:
            pot += e.potential(x,y,aq)
        if pylayers is None:
            pot = np.sum( pot[:,np.newaxis,:,:] * aq.eigvec, 2 )
        else:
            pot = np.sum( pot[:,np.newaxis,:,:] * aq.eigvec[pylayers,:], 2 )
        rv = np.zeros((Nlayers,len(time)))
        if (time[0] < self.tmin) or (time[-1] > self.tmax): print 'Warning, some of the times are smaller than tmin or larger than tmax; zeros are substituted'
        #
        for k in range(self.Ngvbc):
            e = self.gvbcList[k]
            for itime in range(e.Ntstart):
                t = time - e.tstart[itime]
                it = 0
                if t[-1] >= self.tmin:  # Otherwise all zero
                    if (t[0] < self.tmin): it = np.argmax( t >= self.tmin )  # clever call that should be replaced with find_first function when included in numpy
                    for n in range(self.Nin):
                        tp = t[ (t >= self.tintervals[n]) & (t < self.tintervals[n+1]) ]
                        ## I think these lines are not needed anymore as I modified tintervals[0] and tintervals[-1] by eps
                        #if n == self.Nin-1:
                        #    tp = t[ (t >= self.tintervals[n]) & (t <= self.tintervals[n+1]) ]
                        #else:
                        #    tp = t[ (t >= self.tintervals[n]) & (t < self.tintervals[n+1]) ]
                        Nt = len(tp)
                        if Nt > 0:  # if all values zero, don't do the inverse transform
                            for i in range(Nlayers):
                                # I used to check the first value only, but it seems that checking that nothing is zero is needed and should be sufficient
                                #if np.abs( pot[k,i,n*self.Npin] ) > 1e-20:  # First value very small
                                if not np.any( pot[k,i,n*self.Npin:(n+1)*self.Npin] == 0.0) : # If there is a zero item, zero should be returned; funky enough this can be done with a straight equal comparison
                                    rv[i,it:it+Nt] += e.bc[itime] * invlaptrans.invlap( tp, self.tintervals[n], self.tintervals[n+1], pot[k,i,n*self.Npin:(n+1)*self.Npin], self.gamma[n], self.M, Nt )
                            it = it + Nt
        return rv
    def head(self,x,y,t,layers=None,aq=None,derivative=0):
        if aq is None: aq = self.aq.findAquiferData(x,y)
        if layers is None:
            pylayers = range(aq.Naq)
        else:
            pylayers = np.atleast_1d(layers) - 1
        pot = self.potential(x,y,t,pylayers,aq,derivative)
        return aq.potentialToHead(pot,pylayers)
    def headinside(self,elabel,t):
        return self.elementDict[elabel].headinside(t)
    def strength(self,elabel,t):
        return self.elementDict[elabel].strength(t)
    def headalongline(self,x,y,t,layers=None):
        '''Returns head[Nlayers,len(t),len(x)]
        Assumes same number of layers for each x and y
        layers may be None or list of layers for which head is computed'''
        xg,yg = np.atleast_1d(x),np.atleast_1d(y)
        if layers is None:
            Nlayers = self.aq.findAquiferData(xg[0],yg[0]).Naq
        else:
            Nlayers = len(np.atleast_1d(layers))
        nx = len(xg)
        if len(yg) == 1:
            yg = yg * np.ones(nx)
        t = np.atleast_1d(t)
        h = np.zeros( (Nlayers,len(t),nx) )
        for i in range(nx):
            h[:,:,i] = self.head(xg[i],yg[i],t,layers)
        return h
    def headgrid(self,x1,x2,nx,y1,y2,ny,t,layers=None):
        '''Returns h[Nlayers,Ntimes,Ny,Nx]. If layers is None, all layers are returned'''
        xg,yg = np.linspace(x1,x2,nx), np.linspace(y1,y2,ny)
        if layers is None:
            Nlayers = self.aq.findAquiferData(xg[0],yg[0]).Naq
        else:
            Nlayers = len(np.atleast_1d(layers))
        t = np.atleast_1d(t)
        h = np.empty( (Nlayers,len(t),ny,nx) )
        for j in range(ny):
            for i in range(nx):
                h[:,:,j,i] = self.head(xg[i],yg[j],t,layers)
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
                    # Not needed anymore: if np.abs( pot[n*self.Npin] ) > 1e-20:
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
            return mat,rhs
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
    def storeinput(self,frame):
        self.inputargs, _, _, self.inputvalues = inspect.getargvalues(frame)
    def write(self):
        rv = self.modelname + ' = '+self.name+'(\n'
        for key in self.inputargs[1:]:  # The first argument (self) is ignored
            if isinstance(self.inputvalues[key],np.ndarray):
                rv += key + ' = ' + np.array2string(self.inputvalues[key],separator=',') + ',\n'
            elif isinstance(self.inputvalues[key],str):                
                rv += key + " = '" + self.inputvalues[key] + "',\n"
            else:
                rv += key + ' = ' + str(self.inputvalues[key]) + ',\n'
        rv += ')\n'
        return rv
    def writemodel(self,fname):
        f = open(fname,'w')
        f.write('from ttim2 import *\n')
        f.write( self.write() )
        for e in self.elementList:
            f.write( e.write() )
        f.close()
        
class ModelMaq(TimModel):
    def __init__(self,kaq=[1],z=[1,0],c=[],Saq=[0.001],Sll=[],topboundary='imp',phreatictop=False,tmin=1,tmax=10,M=20):
        self.storeinput(inspect.currentframe())
        kaq = np.atleast_1d(kaq).astype('d')
        Naq = len(kaq)
        z = np.atleast_1d(z).astype('d')
        c = np.atleast_1d(c).astype('d')
        Saq = np.atleast_1d(Saq).astype('d')
        Sll = np.atleast_1d(Sll).astype('d')
        H = z[:-1] - z[1:]
        assert np.all(H >= 0), 'Error: Not all layers thicknesses are non-negative' + str(H) 
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
        else: # leaky layers on top
            assert len(z) == 2*Naq+1, 'Error: Length of z needs to be ' + str(2*Naq+1)
            assert len(c) == Naq, 'Error: Length of c needs to be ' + str(Naq)
            assert len(Saq) == Naq, 'Error: Length of Saq needs to be ' + str(Naq)
            assert len(Sll) == Naq, 'Error: Length of Sll needs to be ' + str(Naq)
            Haq = H[1::2]
            Saq = Saq * Haq
            Sll = Sll * H[::2]
            if phreatictop and (topboundary[:3]=='lea'): Sll[0] = Sll[0] / H[0]
        TimModel.__init__(self,kaq,Haq,c,Saq,Sll,topboundary,tmin,tmax,M)
        self.name = 'ModelMaq'
        
class Model3D(TimModel):
    def __init__(self,kaq=[1,1,1],z=[4,3,2,1],Saq=[0.3,0.001,0.001],kzoverkh=[.1,.1,.1],phreatictop=True,tmin=1,tmax=10,M=20):
        '''z must have the length of the number of layers + 1'''
        self.storeinput(inspect.currentframe())
        kaq = np.atleast_1d(kaq).astype('d')
        z = np.atleast_1d(z).astype('d')
        Naq = len(z) - 1
        if len(kaq) == 1: kaq = kaq * np.ones(Naq)
        Saq = np.atleast_1d(Saq).astype('d')
        if len(Saq) == 1: Saq = Saq * np.ones(Naq)
        kzoverkh = np.atleast_1d(kzoverkh).astype('d')
        if len(kzoverkh) == 1: kzoverkh = kzoverkh * np.ones(Naq)
        H = z[:-1] - z[1:]
        c = 0.5 * H[:-1] / ( kzoverkh[:-1] * kaq[:-1] ) + 0.5 * H[1:] / ( kzoverkh[1:] * kaq[1:] )
        Saq = Saq * H
        if phreatictop: Saq[0] = Saq[0] / H[0]
        c = np.hstack((np.nan,c))
        Sll = 1e-20 * np.ones(len(c))
        TimModel.__init__(self,kaq,H,c,Saq,Sll,'imp',tmin,tmax,M)
        self.name = 'Model3D'
        
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
        coef[ipylayers,:,np] are the coefficients if the element is in ipylayers belonging to Laplace parameter number np
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
    def headToPotential(self,h,pylayers):
        return h * self.Tcol[pylayers]
    def potentialToHead(self,p,pylayers):
        return p / self.Tcol[pylayers]
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
    def __init__(self, model, Nparam=1, Nunknowns=0, layers=1, tsandbc=[(0.0,0.0)], type='z', name='', label=None):
        '''Types of elements
        'g': strength is given through time
        'v': boundary condition is variable through time
        'z': boundary condition is zero through time
        Definition of Nlayers, Ncp, Npar, Nunknowns:
        Nlayers: Number of layers that the element is screened in, set in Element
        Ncp: Number of control points along the element
        Nparam: Number of parameters, commonly Nlayers * Ncp
        Nunknowns: Number of unknown parameters, commonly zero or Npar
        '''
        self.model = model
        self.aq = None # Set in the initialization function
        self.Nparam = Nparam  # Number of parameters
        self.Nunknowns = Nunknowns
        self.layers = np.atleast_1d(layers)
        self.pylayers = self.layers - 1
        self.Nlayers = len(self.layers)
        #
        tsandbc = np.atleast_2d(tsandbc).astype('d')
        assert tsandbc.shape[1] == 2, "TTim input error: tsandQ or tsandh need to be 2D lists or arrays like [(0,1),(2,5),(8,0)] "
        self.tstart,self.bcin = tsandbc[:,0],tsandbc[:,1]
        if self.tstart[0] > 0:
            self.tstart = np.hstack((np.zeros(1),self.tstart))
            self.bcin = np.hstack((np.zeros(1),self.bcin))
        #
        self.type = type  # 'z' boundary condition through time or 'v' boundary condition through time
        self.name = name
        self.label = label
        if self.label is not None: assert self.label not in self.model.elementDict.keys(), "TTim error: label "+self.label+" already exists"
        self.Rzero = 30.0
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
        '''Returns complex array of size (Ngvbc,Naq,Np)'''
        if aq is None: aq = self.model.aq.findAquiferData(x,y)
        return np.sum( self.parameters[:,:,np.newaxis,:] * self.potinf(x,y,aq), 1 )
    def unitpotential(self,x,y,aq=None):
        '''Returns complex array of size (Naq,Np)
        Can be more efficient for given elements'''
        if aq is None: aq = self.model.aq.findAquiferData(x,y)
        return np.sum( self.potinf(x,y,aq), 0 )
    # Functions used to build equations
    def potinflayers(self,x,y,pylayers=0,aq=None):
        '''pylayers can be scalar, list, or array. returns array of size (len(pylayers),Nparam,Np)
        only used in building equations'''
        if aq is None: aq = self.model.aq.findAquiferData( x, y )
        pot = self.potinf(x,y,aq)
        rv = np.sum( pot[:,np.newaxis,:,:] * aq.eigvec, 2 )
        rv = rv.swapaxes(0,1) # As the first axes needs to be the number of layers
        return rv[pylayers,:]
    def potentiallayers(self,x,y,pylayers=0,aq=None):
        '''Returns complex array of size (Ngvbc,len(pylayers),Np)
        only used in building equations'''
        if aq is None: aq = self.model.aq.findAquiferData(x,y)
        pot = self.potential(x,y,aq)
        phi = np.sum( pot[:,np.newaxis,:,:] * aq.eigvec, 2 )
        return phi[:,pylayers,:]
    def unitpotentiallayers(self,x,y,pylayers=0,aq=None):
        '''Returns complex array of size (len(pylayers),Np)
        only used in building equations'''
        if aq is None: aq = self.model.aq.findAquiferData(x,y)
        pot = self.unitpotential(x,y,aq)
        phi = np.sum( pot[np.newaxis,:,:] * aq.eigvec, 1 )
        return phi[pylayers,:]
    def strength(self,t,derivative=0):
        '''returns array of strengths (Nlayers,len(t)) t must be ordered and tmin <= t <= tmax'''
        # Could potentially be more efficient if s is pre-computed for all elements, but I don't know if that is worthwhile to store as it is quick now
        time = np.atleast_1d(t).copy()
        rv = np.zeros((self.Nlayers,len(time)))
        if self.type == 'g':
            s = self.strengthinflayers * self.model.p ** derivative
            for itime in range(self.Ntstart):
                t = time - self.tstart[itime]
                for i in range(self.Nlayers):
                    rv[i] += self.bc[itime] * self.model.inverseLapTran(s[i],t)
        else:
            s = np.sum( self.parameters[:,:,np.newaxis,:] * self.strengthinf, 1 )
            s = np.sum( s[:,np.newaxis,:,:] * self.aq.eigvec, 2 )
            s = s[:,self.pylayers,:]
            for k in range(self.model.Ngvbc):
                e = self.model.gvbcList[k]
                for itime in range(e.Ntstart):
                    t = time - e.tstart[itime]
                    for i in range(self.Nlayers):
                        rv[i] += e.bc[itime] * self.model.inverseLapTran(s[k,i],t)
        return rv
    def headinside(self,t):
        print "This function not implemented for this element"
        return
    def layout(self):
        return '','',''
    def storeinput(self,frame):
        self.inputargs, _, _, self.inputvalues = inspect.getargvalues(frame)
    def write(self):
        rv = self.name + '(' + self.model.modelname + ',\n'
        for key in self.inputargs[2:]:  # The first two are ignored
            if isinstance(self.inputvalues[key],np.ndarray):
                rv += key + ' = ' + np.array2string(self.inputvalues[key],separator=',') + ',\n'
            elif isinstance(self.inputvalues[key],str):                
                rv += key + " = '" + self.inputvalues[key] + "',\n"
            else:
                rv += key + ' = ' + str(self.inputvalues[key]) + ',\n'
        rv += ')\n'
        return rv
    
class HeadEquation:
    def equation(self):
        '''Mix-in class that returns matrix rows for head-specified conditions. (really written as constant potential element)
        Works for Nunknowns = 1
        Returns matrix part Nunknowns,Neq,Np, complex
        Returns rhs part Nunknowns,Nvbc,Np, complex
        Phi_out - c*T*q_s = Phi_in
        Well: q_s = Q / (2*pi*r_w*H)
        LineSink: q_s = sigma / H = Q / (L*H)
        '''
        mat = np.empty( (self.Nunknowns,self.model.Neq,self.model.Np), 'D' )
        rhs = np.zeros( (self.Nunknowns,self.model.Ngvbc,self.model.Np), 'D' )  # Needs to be initialized to zero
        for icp in range(self.Ncp):
            istart = icp*self.Nlayers
            ieq = 0  
            for e in self.model.elementList:
                if e.Nunknowns > 0:
                    mat[istart:istart+self.Nlayers,ieq:ieq+e.Nunknowns,:] = e.potinflayers(self.xc[icp],self.yc[icp],self.pylayers)
                    if e == self:
                        for i in range(self.Nlayers): mat[istart+i,ieq+istart+i,:] -= self.resfacp[istart+i] * e.strengthinflayers[istart+i]
                    ieq += e.Nunknowns
            for i in range(self.model.Ngbc):
                rhs[istart:istart+self.Nlayers,i,:] -= self.model.gbcList[i].unitpotentiallayers(self.xc[icp],self.yc[icp],self.pylayers)  # Pretty cool that this works, really
            if self.type == 'v':
                iself = self.model.vbcList.index(self)
                for i in range(self.Nlayers):
                    rhs[istart+i,self.model.Ngbc+iself,:] = self.pc[istart+i] / self.model.p
        return mat, rhs

class MscreenEquation:
    def equation(self):
        '''Mix-in class that returns matrix rows for multi-scren conditions where total discharge is specified.
        Works for Nunknowns = 1
        Returns matrix part Nunknowns,Neq,Np, complex
        Returns rhs part Nunknowns,Nvbc,Np, complex
        head_out - c*q_s = h_in
        Set h_i - h_(i+1) = 0 and Sum Q_i = Q'''
        mat = np.zeros( (self.Nunknowns,self.model.Neq,self.model.Np), 'D' )  # Needs to be zero for last equation, but I think setting the whole array is quicker
        rhs = np.zeros( (self.Nunknowns,self.model.Ngvbc,self.model.Np), 'D' )  # Needs to be initialized to zero
        ieq = 0
        for icp in range(self.Ncp):
            istart = icp*self.Nlayers
            ieq = 0 
            for e in self.model.elementList:
                if e.Nunknowns > 0:
                    head = e.potinflayers(self.xc[icp],self.yc[icp],self.pylayers) / self.aq.T[self.pylayers][:,np.newaxis,np.newaxis]  # T[self.pylayers,np.newaxis,np.newaxis] is not allowed
                    mat[istart:istart+self.Nlayers-1,ieq:ieq+e.Nunknowns,:] = head[:-1,:] - head[1:,:]
                    if e == self:
                        for i in range(self.Nlayers-1):
                            mat[istart+i,ieq+istart+i,:] -= self.resfach[istart+i] * e.strengthinflayers[istart+i]
                            mat[istart+i,ieq+istart+i+1,:] += self.resfach[istart+i+1] * e.strengthinflayers[istart+i+1]
                            mat[istart+i,ieq+istart:ieq+istart+i+1,:] -= self.vresfac[istart+i] * e.strengthinflayers[istart+i]
                            #mat[i,ieq:ieq+i+1,:] -= self.res[i] * disinf[i]
                        mat[istart+self.Nlayers-1,ieq+istart:ieq+istart+self.Nlayers,:] = 1.0
                    ieq += e.Nunknowns
            for i in range(self.model.Ngbc):
                head = self.model.gbcList[i].unitpotentiallayers(self.xc[icp],self.yc[icp],self.pylayers) / self.aq.T[self.pylayers][:,np.newaxis]
                rhs[istart:istart+self.Nlayers-1,i,:] -= head[:-1,:] - head[1:,:]
            if self.type == 'v':
                iself = self.model.vbcList.index(self)
                rhs[istart+self.Nlayers-1,self.model.Ngbc+iself,:] = 1.0  # If self.type == 'z', it should sum to zero, which is the default value of rhs
        return mat, rhs
    
class MscreenDitchEquation:
    def equation(self):
        '''Mix-in class that returns matrix rows for multi-scren conditions where total discharge is specified.
        Works for Nunknowns = 1
        Returns matrix part Nunknowns,Neq,Np, complex
        Returns rhs part Nunknowns,Nvbc,Np, complex
        head_out - c*q_s = h_in
        Set h_i - h_(i+1) = 0 and Sum Q_i = Q'''
        mat = np.zeros( (self.Nunknowns,self.model.Neq,self.model.Np), 'D' )  # Needs to be zero for last equation, but I think setting the whole array is quicker
        rhs = np.zeros( (self.Nunknowns,self.model.Ngvbc,self.model.Np), 'D' )  # Needs to be initialized to zero
        ieq = 0
        for icp in range(self.Ncp):
            istart = icp*self.Nlayers
            ieq = 0 
            for e in self.model.elementList:
                if e.Nunknowns > 0:
                    head = e.potinflayers(self.xc[icp],self.yc[icp],self.pylayers) / self.aq.T[self.pylayers][:,np.newaxis,np.newaxis]  # T[self.pylayers,np.newaxis,np.newaxis] is not allowed
                    mat[istart:istart+self.Nlayers-1,ieq:ieq+e.Nunknowns,:] = head[:-1,:] - head[1:,:]
                    mat[istart+self.Nlayers-1,ieq:ieq+e.Nunknowns,:] = head[0,:] # Store head in top layer in last equation of this control point
                    if e == self:
                        if icp == 0:
                            istartself = ieq  # Needed to build last equation
                        for i in range(self.Nlayers-1):
                            mat[istart+i,ieq+istart+i,:] -= self.resfach[istart+i] * e.strengthinflayers[istart+i]
                            mat[istart+i,ieq+istart+i+1,:] += self.resfach[istart+i+1] * e.strengthinflayers[istart+i+1]
                            mat[istart+i,ieq+istart:ieq+istart+i+1,:] -= self.vresfac[istart+i] * e.strengthinflayers[istart+i]
                    ieq += e.Nunknowns
            for i in range(self.model.Ngbc):
                head = self.model.gbcList[i].unitpotentiallayers(self.xc[icp],self.yc[icp],self.pylayers) / self.aq.T[self.pylayers][:,np.newaxis]
                rhs[istart:istart+self.Nlayers-1,i,:] -= head[:-1,:] - head[1:,:]
                rhs[istart+self.Nlayers-1,i,:] -= head[0,:] # Store minus the head in top layer in last equation for this control point
        # Modify last equations
        for icp in range(self.Ncp-1):
            ieq = (icp+1) * self.Nlayers - 1
            mat[ieq,:,:] -= mat[ieq+self.Nlayers,:,:]  # Head first layer control point icp - Head first layer control point icp + 1
            rhs[ieq,:,:] -= rhs[ieq+self.Nlayers,:,:]
        # Last equation setting the total discharge of the ditch
        print 'istartself ',istartself
        mat[-1,:,:] = 0.0  
        mat[-1,istartself:istartself+self.Nparam,:] = 1.0
        rhs[-1,:,:] = 0.0
        if self.type == 'v':
            iself = self.model.vbcList.index(self)
            rhs[-1,self.model.Ngbc+iself,:] = 1.0  # If self.type == 'z', it should sum to zero, which is the default value of rhs
        return mat, rhs
    
class WellBase(Element):
    '''Well Base Class. All Well elements are derived from this class'''
    def __init__(self,model,xw=0,yw=0,rw=0.1,tsandbc=[(0.0,1.0)],res=0.0,layers=1,type='',name='WellBase',label=None):
        Element.__init__(self, model, Nparam=1, Nunknowns=0, layers=layers, tsandbc=tsandbc, type=type, name=name, label=label)
        self.Nparam = len(self.pylayers)  # Defined here and not in Element as other elements can have multiple parameters per layers
        self.xw = float(xw); self.yw = float(yw); self.rw = float(rw); self.res = res
        self.model.addElement(self)
    def __repr__(self):
        return self.name + ' at ' + str((self.xw,self.yw))
    def initialize(self):
        self.xc = np.array([self.xw + self.rw]); self.yc = np.array([self.yw]) # Control point to make sure the point is always the same for all elements
        self.Ncp = 1
        self.aq = self.model.aq.findAquiferData(self.xw,self.yw)
        self.setbc()
        coef = self.aq.coef[self.pylayers,:]
        laboverrwk1 = self.aq.lab / (self.rw * kv(1,self.rw/self.aq.lab))
        self.setflowcoef()
        self.term = -1.0 / (2*np.pi) * laboverrwk1 * self.flowcoef * coef  # shape (self.Nparam,self.aq.Naq,self.model.Np)
        self.term2 = self.term.reshape(self.Nparam,self.aq.Naq,self.model.Nin,self.model.Npin)
        self.strengthinf = self.flowcoef * coef
        self.strengthinflayers = np.sum(self.strengthinf * self.aq.eigvec[self.pylayers,:,:], 1) 
        self.resfach = self.res / ( 2*np.pi*self.rw*self.aq.Haq[self.pylayers] )  # Q = (h - hw) / resfach
        self.resfacp = self.resfach * self.aq.T[self.pylayers]  # Q = (Phi - Phiw) / resfacp
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
        '''Returns head inside the well for the layers that the well is screened in'''
        return self.model.head(self.xc,self.yc,t)[self.pylayers] - self.resfach[:,np.newaxis] * self.strength(t)
    def layout(self):
        return 'point',self.xw,self.yw

class LineSinkBase(Element):
    '''LineSink Base Class. All LineSink elements are derived from this class'''
    def __init__(self,model,x1=-1,y1=0,x2=1,y2=0,tsandbc=[(0.0,1.0)],res=0.0,wh='H',layers=1,type='',name='LineSinkBase',label=None,addtomodel=True):
        Element.__init__(self, model, Nparam=1, Nunknowns=0, layers=layers, tsandbc=tsandbc, type=type, name=name, label=label)
        self.Nparam = len(self.pylayers)
        self.x1 = float(x1); self.y1 = float(y1); self.x2 = float(x2); self.y2 = float(y2); self.res = res; self.wh = wh
        if addtomodel: self.model.addElement(self)
        self.xa,self.ya,self.xb,self.yb,self.np = np.zeros(1),np.zeros(1),np.zeros(1),np.zeros(1),np.zeros(1,'i')  # needed to call bessel.circle_line_intersection
    def __repr__(self):
        return self.name + ' from ' + str((self.x1,self.y1)) +' to '+str((self.x2,self.y2))
    def initialize(self):
        self.xc = np.array([0.5*(self.x1+self.x2)]); self.yc = np.array([0.5*(self.y1+self.y2)])
        self.Ncp = 1
        self.z1 = self.x1 + 1j*self.y1; self.z2 = self.x2 + 1j*self.y2
        self.L = np.abs(self.z1-self.z2)
        self.aq = self.model.aq.findAquiferData(self.xc,self.yc)
        self.setbc()
        coef = self.aq.coef[self.pylayers,:]
        self.setflowcoef()
        self.term = self.flowcoef * coef  # shape (self.Nparam,self.aq.Naq,self.model.Np)
        self.term2 = self.term.reshape(self.Nparam,self.aq.Naq,self.model.Nin,self.model.Npin)
        self.strengthinf = self.flowcoef * coef
        self.strengthinflayers = np.sum(self.strengthinf * self.aq.eigvec[self.pylayers,:,:], 1)
        if self.wh == 'H':
            self.wh = self.aq.Haq[self.pylayers]
        elif self.wh == '2H':
            self.wh = 2.0 * self.aq.Haq[self.pylayers]
        else:
            self.wh = np.atleast_1d(self.wh) * np.ones(self.Nlayers)
        self.resfach = self.res / (self.wh * self.L)  # Q = (h - hls) / resfach
        self.resfacp = self.resfach * self.aq.T[self.pylayers]  # Q = (Phi - Phils) / resfacp
    def setflowcoef(self):
        '''Separate function so that this can be overloaded for other types'''
        self.flowcoef = 1.0 / self.model.p  # Step function
    def potinf(self,x,y,aq=None):
        '''Can be called with only one x,y value'''
        if aq is None: aq = self.model.aq.findAquiferData( x, y )
        rv = np.zeros((self.Nparam,aq.Naq,self.model.Nin,self.model.Npin),'D')
        if aq == self.aq:
            pot = np.zeros(self.model.Npin,'D')
            for i in range(self.aq.Naq):
                for j in range(self.model.Nin):
                    bessel.circle_line_intersection(self.z1,self.z2,x+y*1j,self.Rzero*abs(self.model.aq.lab2[i,j,0]),self.xa,self.ya,self.xb,self.yb,self.np)
                    if self.np > 0:
                        za = complex(self.xa,self.ya); zb = complex(self.xb,self.yb) # f2py has problem returning complex arrays -> fixed in new numpy
                        bessel.bessellsv(x,y,za,zb,self.aq.lab2[i,j,:],pot)
                        rv[:,i,j,:] = self.term2[:,i,j,:] * pot / self.L  # Divide by L as the parameter is now total discharge
        rv.shape = (self.Nparam,aq.Naq,self.model.Np)
        return rv
    def headinside(self,t):
        return self.model.head(self.xc,self.yc,t)[self.pylayers] - self.resfach[:,np.newaxis] * self.strength(t)
    def layout(self):
        return 'line', [self.x1,self.x2], [self.y1,self.y2]
    
class Well(WellBase):
    '''Well with non-zero and potentially variable discharge through time'''
    def __init__(self,model,xw=0,yw=0,rw=0.1,tsandQ=[(0.0,1.0)],res=0.0,layers=1,label=None):
        self.storeinput(inspect.currentframe())
        WellBase.__init__(self,model,xw,yw,rw,tsandbc=tsandQ,res=res,layers=layers,type='g',name='Well',label=label)
        
class LineSink(LineSinkBase):
    '''LineSink with non-zero and potentially variable discharge through time'''
    def __init__(self,model,x1=-1,y1=0,x2=1,y2=0,tsandQ=[(0.0,1.0)],res=0.0,wh='H',layers=1,label=None,addtomodel=True):
        self.storeinput(inspect.currentframe())
        LineSinkBase.__init__(self,model,x1=x1,y1=y1,x2=x2,y2=y2,tsandbc=tsandQ,res=res,wh=wh,layers=layers,type='g',name='LineSink',label=label,addtomodel=addtomodel)

class ZeroMscreenWell(WellBase,MscreenEquation):
    '''MscreenWell with zero discharge. Needs to be screened in multiple layers; Head is same in all screened layers'''
    def __init__(self,model,xw=0,yw=0,rw=0.1,res=0.0,layers=[1,2],vres=0.0,label=None):
        assert len(layers) > 1, "TTim input error: number of layers for ZeroMscreenWell must be at least 2"
        self.storeinput(inspect.currentframe())
        WellBase.__init__(self,model,xw,yw,rw,tsandbc=[(0.0,0.0)],res=res,layers=layers,type='z',name='ZeroMscreenWell',label=label)
        self.Nunknowns = self.Nparam
        self.vres = np.atleast_1d(vres)  # Vertical resistance inside well
        if len(self.vres) == 1: self.vres = self.vres[0] * np.ones(self.Nlayers-1)
        self.vresfac = self.vres / (np.pi * self.rw**2)  # Qv = (hn - hn-1) / vresfac[n-1]
    def initialize(self):
        WellBase.initialize(self)
        self.parameters = np.zeros( (self.model.Ngvbc, self.Nparam, self.model.Np), 'D' )
        
class ZeroMscreenLineSink(LineSinkBase,MscreenEquation):
    '''MscreenLineSink with zero discharge. Needs to be screened in multiple layers; Head is same in all screened layers'''
    def __init__(self,model,x1=-1,y1=0,x2=1,y2=0,res=0.0,wh='H',layers=[1,2],vres=0.0,wv=1.0,label=None,addtomodel=True):
        assert len(layers) > 1, "TTim input error: number of layers for ZeroMscreenLineSink must be at least 2"
        self.storeinput(inspect.currentframe())
        LineSinkBase.__init__(self,model,x1=x1,y1=y1,x2=x2,y2=y2,tsandbc=[(0.0,0.0)],res=res,wh=wh,layers=layers,type='z',name='ZeroMscreenLineSink',label=label,addtomodel=addtomodel)
        self.Nunknowns = self.Nparam
        self.vres = np.atleast_1d(vres)  # Vertical resistance inside line-sink
        self.wv = wv
        if len(self.vres) == 1: self.vres = self.vres[0] * np.ones(self.Nlayers-1)
    def initialize(self):
        LineSinkBase.initialize(self)
        self.parameters = np.zeros( (self.model.Ngvbc, self.Nparam, self.model.Np), 'D' )
        self.vresfac = self.vres / (self.wv * self.L)  # Qv = (hn - hn-1) / vresfac[n-1]
        
class MscreenWell(WellBase,MscreenEquation):
    '''MscreenWell that varies through time. May be screened in multiple layers but heads are same in all screened layers'''
    def __init__(self,model,xw=0,yw=0,rw=0.1,tsandQ=[(0.0,1.0)],res=0.0,layers=[1,2],label=None):
        assert len(layers) > 1, "TTim input error: number of layers for MscreenWell must be at least 2"
        self.storeinput(inspect.currentframe())
        WellBase.__init__(self,model,xw,yw,rw,tsandbc=tsandQ,res=res,layers=layers,type='v',name='MscreenWell',label=label)
        self.Nunknowns = self.Nparam
        self.vresfac = np.zeros(self.Nlayers-1)  # Vertical resistance inside well, defined but not used; only used for ZeroMscreenWell
    def initialize(self):
        WellBase.initialize(self)
        self.parameters = np.zeros( (self.model.Ngvbc, self.Nparam, self.model.Np), 'D' )
        
class MscreenLineSink(LineSinkBase,MscreenEquation):
    '''MscreenLineSink that varies through time. Must be screened in multiple layers but heads are same in all screened layers'''
    def __init__(self,model,x1=-1,y1=0,x2=1,y2=0,tsandQ=[(0.0,1.0)],res=0.0,wh='H',layers=[1,2],label=None,addtomodel=True):
        assert len(layers) > 1, "TTim input error: number of layers for MscreenLineSink must be at least 2"
        self.storeinput(inspect.currentframe())
        LineSinkBase.__init__(self,model,x1=x1,y1=y1,x2=x2,y2=y2,tsandbc=tsandQ,res=res,wh=wh,layers=layers,type='v',name='MscreenLineSink',label=label,addtomodel=addtomodel)
        self.Nunknowns = self.Nparam
    def initialize(self):
        LineSinkBase.initialize(self)
        self.parameters = np.zeros( (self.model.Ngvbc, self.Nparam, self.model.Np), 'D' )
        
class ZeroHeadWell(WellBase,HeadEquation):
    '''HeadWell that remains zero and constant through time'''
    def __init__(self,model,xw=0,yw=0,rw=0.1,res=0.0,layers=1,label=None):
        self.storeinput(inspect.currentframe())
        WellBase.__init__(self,model,xw,yw,rw,tsandbc=[(0.0,0.0)],res=res,layers=layers,type='z',name='ZeroHeadWell',label=label)
        self.Nunknowns = self.Nparam
    def initialize(self):
        WellBase.initialize(self)
        self.parameters = np.zeros( (self.model.Ngvbc, self.Nparam, self.model.Np), 'D' )
        
class ZeroHeadLineSink(LineSinkBase,HeadEquation):
    '''HeadLineSink that remains zero and constant through time'''
    def __init__(self,model,x1=-1,y1=0,x2=1,y2=0,res=0.0,wh='H',layers=1,label=None,addtomodel=True):
        self.storeinput(inspect.currentframe())
        LineSinkBase.__init__(self,model,x1=x1,y1=y1,x2=x2,y2=y2,tsandbc=[(0.0,0.0)],res=res,wh=wh,layers=layers,type='z',name='ZeroHeadLineSink',label=label,addtomodel=addtomodel)
        self.Nunknowns = self.Nparam
    def initialize(self):
        LineSinkBase.initialize(self)
        self.parameters = np.zeros( (self.model.Ngvbc, self.Nparam, self.model.Np), 'D' )
    
class HeadWell(WellBase,HeadEquation):
    '''HeadWell of which the head varies through time. May be screened in multiple layers but all with the same head'''
    def __init__(self,model,xw=0,yw=0,rw=0.1,tsandh=[(0.0,1.0)],res=0.0,layers=1,label=None):
        self.storeinput(inspect.currentframe())
        WellBase.__init__(self,model,xw,yw,rw,tsandbc=tsandh,res=res,layers=layers,type='v',name='HeadWell',label=label)
        self.Nunknowns = self.Nparam
    def initialize(self):
        WellBase.initialize(self)
        self.parameters = np.zeros( (self.model.Ngvbc, self.Nparam, self.model.Np), 'D' )
        self.pc = self.aq.T[self.pylayers] # Needed in solving; We solve for a unit head
        
class HeadLineSink(LineSinkBase,HeadEquation):
    '''HeadLineSink of which the head varies through time. May be screened in multiple layers but all with the same head'''
    def __init__(self,model,x1=-1,y1=0,x2=1,y2=0,tsandh=[(0.0,1.0)],res=0.0,wh='H',layers=1,label=None,addtomodel=True):
        self.storeinput(inspect.currentframe())
        LineSinkBase.__init__(self,model,x1=x1,y1=y1,x2=x2,y2=y2,tsandbc=tsandh,res=res,wh=wh,layers=layers,type='v',name='HeadLineSink',label=label,addtomodel=addtomodel)
        self.Nunknowns = self.Nparam
    def initialize(self):
        LineSinkBase.initialize(self)
        self.parameters = np.zeros( (self.model.Ngvbc, self.Nparam, self.model.Np), 'D' )
        self.pc = self.aq.T[self.pylayers] # Needed in solving; We solve for a unit head

class LineSinkStringBase(Element):
    def __init__(self,model,xy=[(-1,0),(1,0)],tsandbc=[(0.0,1.0)],layers=1,type='',name='LineSinkStringBase',label=None):
        Element.__init__(self, model, Nparam=1, Nunknowns=0, layers=layers, tsandbc=tsandbc, type=type, name=name, label=label)
        xy = np.atleast_2d(xy).astype('d')
        self.x,self.y = xy[:,0], xy[:,1]
        self.Nls = len(self.x) - 1
        self.Ncp = self.Nls
        self.Nparam = self.Nlayers * self.Nls
        self.Nunknowns = self.Nparam
        self.lsList = []
    def __repr__(self):
        return self.name + ' with nodes ' + str(zip(self.x,self.y))
    def initialize(self):
        for ls in self.lsList:
            ls.initialize()
        self.aq = self.model.aq.findAquiferData(self.lsList[0].xc,self.lsList[0].yc)
        self.parameters = np.zeros( (self.model.Ngvbc, self.Nparam, self.model.Np), 'D' )
        self.setbc()
        # As parameters are only stored for the element not the list, we need to combine the following
        self.resfach = []; self.resfacp = []
        for ls in self.lsList:
            ls.initialize()
            self.resfach.extend( ls.resfach.tolist() )  # Needed in solving
            self.resfacp.extend( ls.resfacp.tolist() )  # Needed in solving
        self.resfach = np.array(self.resfach); self.resfacp = np.array(self.resfacp)
        self.strengthinf = np.zeros((self.Nparam,self.aq.Naq,self.model.Np),'D')
        self.strengthinflayers = np.zeros((self.Nparam,self.model.Np),'D')
        self.xc, self.yc = np.zeros(self.Nls), np.zeros(self.Nls)
        for i in range(self.Nls):
            self.strengthinf[i*self.Nlayers:(i+1)*self.Nlayers,:] = self.lsList[i].strengthinf[:]
            self.strengthinflayers[i*self.Nlayers:(i+1)*self.Nlayers,:] = self.lsList[i].strengthinflayers
            self.xc[i], self.yc[i] = self.lsList[i].xc, self.lsList[i].yc
    def potinf(self,x,y,aq=None):
        '''Returns array (Nunknowns,Nperiods)'''
        if aq is None: aq = self.model.aq.findAquiferData( x, y )
        rv = np.zeros((self.Nparam,aq.Naq,self.model.Np),'D')
        for i in range(self.Nls):
            rv[i*self.Nlayers:(i+1)*self.Nlayers,:] = self.lsList[i].potinf(x,y,aq)
        return rv
    
#class LineSinkString(LineSinkStringBase):
#    def __init__(self,model,xy=[(-1,0),(1,0)],tsandQ=[(0.0,1.0)],res=0.0,layers=1,label=None):
#        LineSinkStringBase.__init__(self,model,xy=xy,tsandbc=tsandQ,res=res,layers=layers,type='g',name='LineSinkString',label=label)
#        for i in range(self.Nls):
#            self.lsList.append( LineSink(model,x1=self.x[i],y1=self.y[i],x2=self.x[i+1],y2=self.y[i+1],tsandQ=tsandQ,res=res,layers=layers,label=None,addtomodel=False) )
#        self.model.addElement(self)
    
class ZeroMscreenLineSinkString(LineSinkStringBase,MscreenEquation):
    def __init__(self,model,xy=[(-1,0),(1,0)],res=0.0,wh='H',layers=[1,2],vres=0.0,wv=1.0,label=None):
        LineSinkStringBase.__init__(self,model,xy=xy,tsandbc=[(0.0,0.0)],layers=layers,type='z',name='ZeroMscreenLineSinkString',label=label)
        for i in range(self.Nls):
            self.lsList.append( ZeroMscreenLineSink(model,x1=self.x[i],y1=self.y[i],x2=self.x[i+1],y2=self.y[i+1],res=res,wh=wh,layers=layers,vres=vres,wv=wv,label=None,addtomodel=False) )
        self.model.addElement(self)
    def initialize(self):
        LineSinkStringBase.initialize(self)
        self.vresfac = np.zeros_like( self.resfach )
        for i in range(self.Nls):
            self.vresfac[i*self.Nlayers:(i+1)*self.Nlayers-1] = self.lsList[i].vresfac[:]
    
class MscreenLineSinkString(LineSinkStringBase,MscreenEquation):
    def __init__(self,model,xy=[(-1,0),(1,0)],tsandQ=[(0.0,1.0)],res=0.0,wh='H',layers=[1,2],label=None):
        LineSinkStringBase.__init__(self,model,xy=xy,tsandbc=tsandQ,layers=layers,type='v',name='MscreenLineSinkString',label=label)
        for i in range(self.Nls):
            self.lsList.append( MscreenLineSink(model,x1=self.x[i],y1=self.y[i],x2=self.x[i+1],y2=self.y[i+1],tsandQ=tsandQ,res=res,wh=wh,layers=layers,label=None,addtomodel=False) )
        self.model.addElement(self)
        
class MscreenLineSinkDitchString(LineSinkStringBase,MscreenDitchEquation):
    def __init__(self,model,xy=[(-1,0),(1,0)],tsandQ=[(0.0,1.0)],res=0.0,wh='H',layers=[1,2],label=None):
        LineSinkStringBase.__init__(self,model,xy=xy,tsandbc=tsandQ,layers=layers,type='v',name='MscreenLineSinkStringDitch',label=label)
        for i in range(self.Nls):
            self.lsList.append( MscreenLineSink(model,x1=self.x[i],y1=self.y[i],x2=self.x[i+1],y2=self.y[i+1],tsandQ=tsandQ,res=res,wh=wh,layers=layers,label=None,addtomodel=False) )
        self.model.addElement(self)
    def initialize(self):
        LineSinkStringBase.initialize(self)
        self.vresfac = np.zeros_like( self.resfach )  # set to zero, as I don't quite know what it would mean if it is not zero
        
class ZeroHeadLineSinkString(LineSinkStringBase,HeadEquation):
    def __init__(self,model,xy=[(-1,0),(1,0)],res=0.0,wh='H',layers=1,label=None):
        LineSinkStringBase.__init__(self,model,xy=xy,tsandbc=[(0.0,0.0)],layers=layers,type='z',name='ZeroHeadLineSinkString',label=label)
        for i in range(self.Nls):
            self.lsList.append( ZeroHeadLineSink(model,x1=self.x[i],y1=self.y[i],x2=self.x[i+1],y2=self.y[i+1],res=res,wh=wh,layers=layers,label=None,addtomodel=False) )
        self.model.addElement(self)

class HeadLineSinkString(LineSinkStringBase,HeadEquation):
    def __init__(self,model,xy=[(-1,0),(1,0)],tsandh=[(0.0,1.0)],res=0.0,wh='H',layers=1,label=None):
        LineSinkStringBase.__init__(self,model,xy=xy,tsandbc=tsandh,layers=layers,type='v',name='HeadLineSinkString',label=label)
        for i in range(self.Nls):
            self.lsList.append( HeadLineSink(model,x1=self.x[i],y1=self.y[i],x2=self.x[i+1],y2=self.y[i+1],tsandh=tsandh,res=res,wh=wh,layers=layers,label=None,addtomodel=False) )
        self.model.addElement(self)
    def initialize(self):
        LineSinkStringBase.initialize(self)
        self.pc = np.zeros(self.Nls*self.Nlayers)
        for i in range(self.Nls): self.pc[i*self.Nlayers:(i+1)*self.Nlayers] = self.lsList[i].pc
    
def xsection(ml,x1=0,x2=1,y1=0,y2=0,N=100,t=1,layers=1,color=None,lw=1,newfig=True):
    if newfig: plt.figure()
    x = np.linspace(x1,x2,N)
    y = np.linspace(y1,y2,N)
    s = np.sqrt( (x-x[0])**2 + (y-y[0])**2 )
    h = ml.headalongline(x,y,t,layers)
    Nlayers,Ntime,Nx = h.shape
    for i in range(Nlayers):
        for j in range(Ntime):
            if color is None:
                plt.plot(s,h[i,j,:],lw=lw)
            else:
                plt.plot(s,h[i,j,:],color,lw=lw)
    plt.show()
                
def timcontour( ml, xmin, xmax, nx, ymin, ymax, ny, levels = 10, t=0.0, layers = 1,\
               color = 'k', lw = 0.5, style = 'solid',layout = True, newfig = True, \
               labels = False, labelfmt = '%1.2f'):
    '''Contour heads with pylab'''
    print 'grid of '+str((nx,ny))+'. gridding in progress. hit ctrl-c to abort'
    h = ml.headgrid(xmin,xmax,nx,ymin,ymax,ny,t,layers)  # h[Nlayers,Ntimes,Ny,Nx]
    xg, yg = np.linspace(xmin,xmax,nx), np.linspace(ymin,ymax,ny)
    Nlayers, Ntimes = h.shape[0:2]
    # Contour
    if type(levels) is list: levels = np.arange( levels[0],levels[1],levels[2] )
    # Colors
    if color is not None: color = [color]   
    if newfig:
        fig = plt.figure( figsize=(8,8) )
        ax = fig.add_subplot(111)
    else:
        fig = plt.gcf()
        ax = plt.gca()
    ax.set_aspect('equal','box')
    ax.set_xlim(xmin,xmax); ax.set_ylim(ymin,ymax)
    ax.set_autoscale_on(False)
    if layout: timlayout(ml,ax)
    # Contour
    plt.rcParams['contour.negative_linestyle'] = 'solid'
    if color is None:
        a = ax.contour( xg, yg, h[0,0], levels, linewidths = lw, linestyles = style )
    else:
        a = ax.contour( xg, yg, h[0,0], levels, colors = color[0], linewidths = lw, linestyles = style )
    if labels and not fill:
        ax.clabel(a,fmt=labelfmt)
    plt.show()
    
def pyvertcontour( ml, xmin, xmax, ymin, ymax, nx, zg, levels = 10, t=0.0,\
               color = 'k', width = 0.5, style = 'solid',layout = True, newfig = True, \
               labels = False, labelfmt = '%1.2f', fill=False, sendback = False):
    '''Contours head with pylab'''
    plt.rcParams['contour.negative_linestyle']='solid'
    # Compute grid
    xg = np.linspace(xmin,xmax,nx)
    yg = np.linspace(ymin,ymax,nx)
    sg = np.sqrt((xg-xg[0])**2 + (yg-yg[0])**2)
    print 'gridding in progress. hit ctrl-c to abort'
    pot = np.zeros( ( ml.aq.Naq, nx ), 'd' )
    t = np.atleast_1d(t)
    for ip in range(nx):
        pot[:,ip] = ml.head(xg[ip], yg[ip], t)[:,0]
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
    ax.set_xlim(sg.min(),sg.max()); ax.set_ylim(zg.min(),zg.max())
    ax.set_autoscale_on(False)
    if fill:
        a = ax.contourf( sg, zg, pot, levels )
    else:
        if color is None:
            a = ax.contour( sg, zg, pot, levels, linewidths = width, linestyles = style )
        else:
            a = ax.contour( sg, zg, pot, levels, colors = color[0], linewidths = width, linestyles = style )
    if labels and not fill:
        ax.clabel(a,fmt=labelfmt)
    fig.canvas.draw()
    if sendback == 1: return a
    if sendback == 2: return sg,zg,pot
                
def timlayout( ml, ax, color = 'k', lw = 0.5, style = '-' ):
    for e in ml.elementList:
        t,x,y = e.layout()
        if t == 'point':
            ax.plot( [x], [y], color+'o', markersize=3 ) 
        if t == 'line':
            ax.plot( x, y, color=color, ls = style, lw = lw )
        if t == 'area':
            col = 0.7 + 0.2*np.random.rand()
            ax.fill( x, y, facecolor = [col,col,col], edgecolor = [col,col,col] )

##########################################

ml = ModelMaq(kaq=[10,5],z=[4,2,1,0],c=[100],Saq=[1e-3,1e-4],Sll=[1e-6],tmin=.1,tmax=10)
w1 = Well(ml,0,2,.1,tsandQ=[(0,10)],layers=[1])
ls2 = ZeroHeadLineSinkString(ml,xy=[(-10,-2),(0,-4),(4,0)],layers=[1])
ls1 = MscreenLineSinkDitchString(ml,xy=[(-10,0),(0,0),(10,10)],tsandQ=[(0.0,7.0)],res=0.0,wh='H',layers=[1,2],label=None)
ml.solve()

#ml = ModelMaq([1,20,2],[25,20,18,10,8,0],c=[1000,2000],Saq=[0.1,1e-4,1e-4],Sll=[0,0],phreatictop=True,tmin=0.1,tmax=10000,M=30)
#w1 = Well(ml,0,0,.1,tsandQ=[(0,1000)],layers=[2])
#ls1 = ZeroMscreenLineSink(ml,10,-5,10,5,layers=[1,2,3],res=0.5,wh=1,vres=3,wv=1)
##w2 = ZeroMscreenWell(ml,10,0,res=1.0,layers=[1,2,3],vres=1.0)
##w3 = Well(ml,0,-10,.1,tsandQ=[(0,700)],layers=[2])
#ml.solve()
##ml1 = ModelMaq([1,20,2],[25,20,18,10,8,0],c=[1000,2000],Saq=[1e-4,1e-4,1e-4],Sll=[0,0],tmin=0.1,tmax=10000,M=30)
##w1 = Well(ml1,0,0,.1,tsandQ=[(0,1000)],layers=[2],res=0.1)
##ml1.solve()
#t = np.logspace(-1,3,100)
#h0 = ml.head(50,0,t)
##h1 = ml1.head(50,0,t)
##w = MscreenWell(ml,0,0,.1,tsandQ=[(0,1000),(100,0),(365,1000),(465,0)],layers=[2,3])
##w2 = HeadWell(ml,50,0,.2,tsandh=[(0,1)],layers=[2])
##y = [-500,-300,-200,-100,-50,0,50,100,200,300,500]
##x = 50 * np.ones(len(y))
##ls = ZeroHeadLineSinkString(ml,xy=zip(x,y),layers=[1])
##w = Well(ml,0,0,.1,tsandQ=[(0,1000),(100,0)],layers=[2])
##ml.solve()


#ml = Model3D( kaq=[2,1,5,10,4], z=[10,8,6,4,2,0], Saq=[.1,.0001,.0002,.0002,.0001], phreatictop=True, kzoverkh=0.1, tmin=1e-3, tmax=1e3 )
#w = MscreenWell(ml,0,-25,rw=.3,tsandQ=[(0,100),(100,50)],layers=[2,3])
#ml.solve()
    
##ml = Model3D(kaq=2.0,z=[10,5,0],Saq=[.002,.001],kzoverkh=0.2,phreatictop=False,tmin=.1,tmax=10,M=15)
#ml = ModelMaq(kaq=[10,5],z=[4,2,1,0],c=[100],Saq=[1e-3,1e-4],Sll=[1e-6],tmin=.1,tmax=10,M=15)
##L1 = np.sqrt(10**2+5**2)
##ls1 = LineSink(ml,-10,-10,0,-5,tsandQ=[(0,.05*L1),(1,.02*L1)],res=1.0,layers=[1,2],label='mark1')
#w = MscreenWell(ml,-5,-5,.1,[0,5],layers=[1,2])
#L2 = np.sqrt(10**2+15**2)
#ls2 = LineSink(ml,0,-5,10,10,tsandQ=[(0,.03*L2),(2,.07*L2)],layers=[1],label='mark2')
##ls3a = ZeroHeadLineSink(ml,-10,5,-5,5,res=1.0,layers=[1,2])
##ls3b = ZeroHeadLineSink(ml,-5,5,0,5,res=1.0,layers=[1,2])
##ls3c = ZeroHeadLineSink(ml,0,5,5,5,res=1.0,layers=[1,2])
##lss = HeadLineSinkString(ml,[(-10,5),(-5,5),(0,5)],tsandh=[(0,0.02),(3,0.01)],res=1.0,layers=[1,2])
#lss = ZeroHeadLineSinkString(ml,[(-10,5),(-5,5),(0,5),(5,5)],res=1.0,layers=[1,2])
##lss = MscreenLineSinkString(ml,[(-10,5),(-5,5),(0,5)],tsandQ=[(0,0.2),(3,0.1)],res=1.0,layers=[1,2])
##lss = ZeroMscreenLineSinkString(ml,[(-10,5),(-5,5),(0,5)],res=1.0,layers=[1,2])
##ml.initialize()
#ml.solve()
#print ml.potential(50,50,[0.5,5])

#ml2 = ModelMaq(kaq=[10,5],z=[4,2,1,0],c=[100],Saq=[1e-3,1e-4],Sll=[1e-6],tmin=.1,tmax=10,M=15)
#L1 = np.sqrt(10**2+5**2)
#ls1b = LineSink(ml2,-10,-10,0,-5,tsandQ=[(0,.05*L1),(1,.02*L1)],res=1.0,layers=[1,2],label='mark1')
#L2 = np.sqrt(10**2+15**2)
#ls2b = LineSink(ml2,0,-5,10,10,tsandQ=[(0,.03*L2),(2,.07*L2)],layers=[1],label='mark2')
##ls3a = HeadLineSink(ml2,-10,5,-5,5,tsandh=[(0,0.02),(3,0.01)],res=1.0,layers=[1,2])
##ls3b = HeadLineSink(ml2,-5,5,0,5,tsandh=[(0,0.02),(3,0.01)],res=1.0,layers=[1,2])
##ls3a = ZeroHeadLineSink(ml2,-10,5,-5,5,res=1.0,layers=[1,2])
##ls3b = ZeroHeadLineSink(ml2,-5,5,0,5,res=1.0,layers=[1,2])
##ls3a = MscreenLineSink(ml2,-10,5,-5,5,tsandQ=[(0,0.2),(3,0.1)],res=1.0,layers=[1,2])
##ls3b = MscreenLineSink(ml2,-5,5,0,5,tsandQ=[(0,0.2),(3,0.1)],res=1.0,layers=[1,2])
#ls3a = ZeroMscreenLineSink(ml2,-10,5,-5,5,res=1.0,layers=[1,2])
#ls3b = ZeroMscreenLineSink(ml2,-5,5,0,5,res=1.0,layers=[1,2])
##lssb = HeadLineSinkStringOld(ml2,[(-10,5),(-5,5),(0,5)],tsandh=[(0,0.02),(3,0.01)],res=0.0,layers=[1,2])
#ml2.solve()
#print ml2.potential(50,50,[0.5,5])

#lss = HeadLineSinkString(ml,[(-10,5),(-5,5),(0,5)],tsandh=[(0,0.02),(3,0.01)],res=1.0,layers=[1,2])
#lss = MscreenLineSinkString(ml,[(-10,5),(-5,5),(0,5)],tsandQ=[(0,.03*5),(2,.07*5)],res=0.5,layers=[1,2])
#ls3a = MscreenLineSink(ml,-10,5,-5,5,tsandQ=[(0,.03*5),(2,.07*5)],res=0.5,layers=[1,2])
#ls3b = MscreenLineSink(ml,-5,5,0,5,tsandQ=[(0,.03*5),(2,.07*5)],res=0.5,layers=[1,2])
#
#ml2 = ModelMaq(kaq=[10,5],z=[4,2,1,0],c=[100],Saq=[1e-3,1e-4],Sll=[1e-6],tmin=.1,tmax=10,M=15)
#L1 = np.sqrt(10**2+5**2)
#ls1a = LineSink(ml2,-10,-10,0,-5,tsandQ=[(0,.05*L1),(1,.02*L1)],res=1.0,layers=[1,2],label='mark1')
#L2 = np.sqrt(10**2+15**2)
#ls2a = LineSink(ml2,0,-5,10,10,tsandQ=[(0,.03*L2),(2,.07*L2)],layers=[1],label='mark2')
#ls3a = HeadLineSink(ml2,-10,5,-5,5,tsandh=[(0,0.02),(3,0.01)],res=1.0,layers=[1,2])
#ls3b = HeadLineSink(ml2,-5,5,0,5,tsandh=[(0,0.02),(3,0.01)],res=1.0,layers=[1,2])


##lss = HeadLineSinkString(ml,[(-10,5),(-5,5),(0,5)],tsandh=[(0,0.02),(3,0.01)],res=0.0,layers=[1,2])
##ls3 = ZeroMscreenLineSink(ml,-10,5,0,5,res=1.0,layers=[1,2])
#ml = ModelMaq(kaq=[10,5],z=[4,2,1,0],c=[100],Saq=[1e-3,1e-4],Sll=[1e-6],tmin=.1,tmax=10,M=15)
#w1 = Well(ml,0,0,.1,tsandQ=[(0,5),(1,2)],res=1.0,layers=[1,2])
#w2 = Well(ml,100,0,.1,tsandQ=[(0,3),(2,7)],layers=[1])
##w3 = MscreenWell(ml,0,100,.1,tsandQ=[(0,2),(3,1)],res=2.0,layers=[1,2])
#w3 = ZeroMscreenWell(ml,0,100,.1,res=2.0,layers=[1,2])
##w3 = ZeroHeadWell(ml,0,100,.1,res=1.0,layers=[1,2])
##w3 = HeadWell(ml,0,100,.1,tsandh=[(0,2),(3,1)],res=1.0,layers=[1,2])
#ml.solve()
###print ml.potential(2,3,[.5,5])
#print ml.potential(50,50,[0.5,5])
#ml2.solve()
#print ml2.potential(50,50,[.5,5])
#print lss.strength([.5,5])
#
#ml2 = ModelMaq(kaq=[10,5],z=[4,2,1,0],c=[100],Saq=[1e-3,1e-4],Sll=[1e-6],tmin=0.1,tmax=10,M=15)
#ls1a = LineSink(ml2,-10,-10,0,-5,tsandsig=[(0,.05),(1,.02)],res=1.0,layers=[1,2],label='mark1')
#ls2a = LineSink(ml2,0,-5,10,10,tsandsig=[(0,.03),(2,.07)],layers=[1],label='mark2')
#ls3a = HeadLineSinkStringOld(ml2,[(-10,5),(-5,5),(0,5)],tsandh=[(0,0.02),(3,0.01)],res=0.0,layers=[1,2])
#ml2.solve()
#print ml2.potential(50,50,[0.5,5])

#print 'Q from strength:  ',w3.strength(.5)
#print 'Q from head diff: ',(ml.head(w3.xc,w3.yc,.5)-w3.headinside(.5))/w3.res*2*np.pi*w3.rw*ml.aq.Haq[:,np.newaxis]
#print 'Q from head diff: ',(ml.head(w3.xc,w3.yc,.5)-2.0)/w3.res*2*np.pi*w3.rw*ml.aq.Haq[:,np.newaxis]
#print w3.strength([.5,5])
#print ls3.strength([.5,5])
#print sum(ls3.strength([.5,5]),0)
#Q = w3.strength([.5,5])
#print sum(Q,0)
#print ml.potential(w3.xc,w3.yc,[.5,5])