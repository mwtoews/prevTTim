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

class TimModel:
    def __init__(self,kaq=[1,1],Haq=[1,1],c=[np.nan,100],Saq=[0.3,0.003],Sll=[1e-3],topboundary='imp',tmin=1,tmax=10,M=15):
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
    def potential(self,x,y,t,pylayer=None,aq=None,derivative=0):
        '''Returns pot[Naq,Ntimes] if layer=None, otherwise pot[len(pylayer,Ntimes)]
        t must be ordered '''
        #
        if derivative > 0:
            print 'derivative > 0 not yet implemented in potential'
            return
        #
        if aq is None: aq = self.aq.findAquiferData(x,y)
        if pylayer is None: pylayer = range(aq.Naq)
        Nlayer = len(pylayer)
        time = np.atleast_1d(t).copy()
        pot = np.zeros((self.Ngvbc, aq.Naq, self.Np),'D')
        for i in range(self.Ngbc):
            pot[i,:] += self.gbcList[i].unitpotential(x,y,aq)
        for e in self.vzbcList:
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
                            for i in range(Nlayer):
                                # I used to check the first value only, but it seems that checking that nothing is zero is needed and should be sufficient
                                #if np.abs( pot[k,i,n*self.Npin] ) > 1e-20:  # First value very small
                                if not np.any( pot[k,i,n*self.Npin:(n+1)*self.Npin] == 0.0) : # If there is a zero item, zero should be returned; funky enough this can be done with a straight equal comparison
                                    rv[i,it:it+Nt] += e.bc[itime] * invlaptrans.invlap( tp, self.tintervals[n], self.tintervals[n+1], pot[k,i,n*self.Npin:(n+1)*self.Npin], self.gamma[n], self.M, Nt )
                            it = it + Nt
        return rv
    def head(self,x,y,t,layer=None,aq=None,derivative=0):
        if aq is None: aq = self.aq.findAquiferData(x,y)
        if layer is None:
            pylayer = range(aq.Naq)
        else:
            pylayer = np.atleast_1d(layer) - 1
        pot = self.potential(x,y,t,pylayer,aq,derivative)
        return aq.potentialToHead(pot,pylayer)
    def headinside(self,elabel,t):
        return self.elementDict[elabel].headinside(t)
    def strength(self,elabel,t):
        return self.elementDict[elabel].strength(t)
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
    def headgrid(self,x1,x2,nx,y1,y2,ny,t,layer=None):
        '''Returns h[Nlayers,Ntimes,Ny,Nx]. If layers is None, all layers are returned'''
        xg,yg = np.linspace(x1,x2,nx), np.linspace(y1,y2,ny)
        if layer is None:
            Nlayer = self.aq.findAquiferData(xg[0],yg[0]).Naq
        else:
            Nlayer = len(np.atleast_1d(layer))
        t = np.atleast_1d(t)
        h = np.empty( (Nlayer,len(t),ny,nx) )
        for j in range(ny):
            for i in range(nx):
                h[:,:,j,i] = self.head(xg[i],yg[j],t,layer)
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
    def __init__(self,kaq=[1],z=[1,0],c=[],Saq=[],Sll=[],topboundary='imp',phreatictop=False,tmin=1,tmax=10,M=15):
        self.storeinput(inspect.currentframe())
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
        TimModel.__init__(self,kaq,Haq,c,Saq,Sll,topboundary,tmin,tmax,M)
        self.name = 'ModelMaq'
        
class Model3D(TimModel):
    def __init__(self,kaq=[1,1,1],z=[4,3,2,1],Saq=[0.3,0.001,0.001],kzoverkh=[.1,.1,.1],phreatictop=True,tmin=1,tmax=10,M=15):
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
    def __init__(self, model, Nparam=1, Nunknowns=0, layer=1, tsandbc=[(0.0,0.0)], type='z', name='', label=None):
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
        self.Rzero = 20.0
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
    def strength(self,t,derivative=0):
        '''returns array of strengths (Nlayer,len(t)) t must be ordered and tmin <= t <= tmax'''
        # Could potentially be more efficient if s is pre-computed for all elements, but I don't know if that is worthwhile to store as it is quick now
        time = np.atleast_1d(t).copy()
        rv = np.zeros((self.Nlayer,len(time)))
        if self.type == 'g':
            #s = np.sum(self.strengthinf * self.aq.eigvec, 1) * self.model.p ** derivative
            s = self.strengthinflayer * self.model.p ** derivative
            for itime in range(self.Ntstart):
                t = time - self.tstart[itime]
                for i in self.pylayer:
                    rv[i] += self.bc[itime] * self.model.inverseLapTran(s[i],t)
        else:
            s = np.sum( self.parameters[:,:,np.newaxis,:] * self.strengthinf, 1 )
            s = np.sum( s[:,np.newaxis,:,:] * self.model.aq.eigvec, 2 ) * self.model.p ** derivative
            for k in range(self.model.Ngvbc):
                e = self.model.gvbcList[k]
                for itime in range(e.Ntstart):
                    t = time - e.tstart[itime]
                    for i in self.pylayer:
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
        LineSink: q_s = sigma / H
        '''
        mat = np.empty( (self.Nunknowns,self.model.Neq,self.model.Np), 'D' )
        rhs = np.zeros( (self.Nunknowns,self.model.Ngvbc,self.model.Np), 'D' )  # Needs to be initialized to zero
        ieq = 0
        for e in self.model.elementList:
            if e.Nunknowns > 0:
                mat[:,ieq:ieq+e.Nunknowns,:] = e.potinflayer(self.xc,self.yc,self.pylayer)
                if e == self:
                    for i in range(self.Nunknowns): mat[i,ieq+i,:] -= self.resfac[i] * e.strengthinflayer[i]
                ieq += e.Nunknowns
        for i in range(self.model.Ngbc):
            rhs[:,i,:] -= self.model.gbcList[i].unitpotentiallayer(self.xc,self.yc,self.pylayer)  # Pretty cool that this works, really
        if self.type == 'v':
            iself = self.model.vbcList.index(self)
            for i in range(self.Nunknowns):
                rhs[i,self.model.Ngbc+iself,:] = self.pc[i] / self.model.p
        return mat, rhs
    
class MscreenEquation:
    def equation(self):
        '''Mix-in class that returns matrix rows for multi-scren conditions where total discharge is specified.
        Works for Nunknowns = 1
        Returns matrix part Nunknowns,Neq,Np, complex
        Returns rhs part Nunknowns,Nvbc,Np, complex
        head_out - c*q_s = h_in
        Require h_i - h_(i+1) = 0
        Sum Q_i = Q'''
        mat = np.zeros( (self.Nunknowns,self.model.Neq,self.model.Np), 'D' )  # Needs to be zero for last equation, but I think setting the whole array is quicker
        rhs = np.zeros( (self.Nunknowns,self.model.Ngvbc,self.model.Np), 'D' )  # Needs to be initialized to zero
        ieq = 0
        for e in self.model.elementList:
            if e.Nunknowns > 0:
                head = e.potinflayer(self.xc,self.yc,self.pylayer) / self.aq.T[self.pylayer][:,np.newaxis,np.newaxis]  # T[self.pylayer,np.newaxis,np.newaxis] is not allowed
                mat[:-1,ieq:ieq+e.Nunknowns,:] = head[:-1,:] - head[1:,:]
                if e == self:
                    for i in range(self.Nunknowns-1):
                        mat[i,ieq+i,:] -= self.resfac[i] * e.strengthinflayer[i] / self.aq.T[self.pylayer[i]]  # As equation is in form of heads
                        mat[i,ieq+i+1,:] += self.resfac[i+1] * e.strengthinflayer[i+1] / self.aq.T[self.pylayer[i+1]]
                    mat[-1,ieq:ieq+e.Nunknowns,:] = 1.0
                ieq += e.Nunknowns
        for i in range(self.model.Ngbc):
            head = self.model.gbcList[i].unitpotentiallayer(self.xc,self.yc,self.pylayer) / self.aq.T[self.pylayer][:,np.newaxis]
            rhs[:-1,i,:] -= head[:-1,:] - head[1:,:]
        if self.type == 'v':
            iself = self.model.vbcList.index(self)
            rhs[-1,self.model.Ngbc+iself,:] = 1.0  # If self.type == 'z', it should sum to zero, which is the default value of rhs
        return mat, rhs
    
class WellBase(Element):
    '''Well Base Class. All Well elements are derived from this class'''
    def __init__(self,model,xw=0,yw=0,rw=0.1,tsandbc=[(0.0,1.0)],res=0.0,layer=1,type='',name='WellBase',label=None):
        Element.__init__(self, model, Nparam=1, Nunknowns=0, layer=layer, tsandbc=tsandbc, type=type, name=name, label=label)
        self.Nparam = len(self.pylayer)  # Defined here and not in Element as other elements can have multiple parameters per layer
        self.xw = float(xw); self.yw = float(yw); self.rw = float(rw); self.res = res
        self.model.addElement(self)
    def __repr__(self):
        return self.name + ' at ' + str((self.xw,self.yw))
    def initialize(self):
        self.xc = self.xw + self.rw; self.yc = self.yw # Control point to make sure the point is always the same for all elements
        self.aq = self.model.aq.findAquiferData(self.xw,self.yw)
        self.setbc()
        coef = self.aq.coef[self.pylayer,:]
        laboverrwk1 = self.aq.lab / (self.rw * kv(1,self.rw/self.aq.lab))
        self.setflowcoef()
        self.term = -1.0 / (2*np.pi) * laboverrwk1 * self.flowcoef * coef  # shape (self.Nparam,self.aq.Naq,self.model.Np)
#        self.term = -1.0 / (2*np.pi) * self.flowcoef * coef  # shape (self.Nparam,self.aq.Naq,self.model.Np)
        self.term2 = self.term.reshape(self.Nparam,self.aq.Naq,self.model.Nin,self.model.Npin)
        self.strengthinf = self.flowcoef * coef
        self.strengthinflayer = np.sum(self.strengthinf * self.aq.eigvec[self.pylayer,:,:], 1) 
        self.resfac = self.res * self.aq.T[self.pylayer] / ( 2*np.pi*self.rw*self.aq.Haq[self.pylayer] )
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
        return self.model.head(self.xc,self.yc,t)[self.pylayer] - self.resfac[:,np.newaxis] / self.aq.Tcol[self.pylayer] * self.strength(t)
    def layout(self):
        return 'point',self.xw,self.yw

class LineSinkBase(Element):
    '''LineSink Base Class. All LineSink elements are derived from this class'''
    def __init__(self,model,x1=-1,y1=0,x2=1,y2=0,tsandbc=[(0.0,1.0)],res=0.0,layer=1,type='',name='LineSinkBase',label=None,addtomodel=True):
        Element.__init__(self, model, Nparam=1, Nunknowns=0, layer=layer, tsandbc=tsandbc, type=type, name=name, label=label)
        self.Nparam = len(self.pylayer)
        self.x1 = float(x1); self.y1 = float(y1); self.x2 = float(x2); self.y2 = float(y2); self.res = res
        if addtomodel: self.model.addElement(self)
        self.xa,self.ya,self.xb,self.yb,self.np = np.zeros(1),np.zeros(1),np.zeros(1),np.zeros(1),np.zeros(1,'i')  # needed to call bessel.circle_line_intersection
    def __repr__(self):
        return self.name + ' from ' + str((self.x1,self.y1)) +' to '+str((self.x2,self.y2))
    def initialize(self):
        self.xc = 0.5*(self.x1+self.x2); self.yc = 0.5*(self.y1+self.y2)
        self.z1 = self.x1 + 1j*self.y1; self.z2 = self.x2 + 1j*self.y2
        self.aq = self.model.aq.findAquiferData(self.xc,self.yc)
        self.setbc()
        coef = self.aq.coef[self.pylayer,:]
        self.setflowcoef()
        self.term = self.flowcoef * coef  # shape (self.Nparam,self.aq.Naq,self.model.Np)
        self.term2 = self.term.reshape(self.Nparam,self.aq.Naq,self.model.Nin,self.model.Npin)
        self.strengthinf = self.flowcoef * coef
        self.strengthinflayer = np.sum(self.strengthinf * self.aq.eigvec[self.pylayer,:,:], 1)
        self.resfac = self.res * self.aq.T[self.pylayer] / self.aq.Haq[self.pylayer]
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
                    bessel.circle_line_intersection(self.z1,self.z2,x+y*1j,20.0*abs(self.model.aq.lab2[i,j,0]),self.xa,self.ya,self.xb,self.yb,self.np)
                    if self.np > 0:
                        za = complex(self.xa,self.ya); zb = complex(self.xb,self.yb) # f2py has problem returning complex arrays -> fixed in new numpy
                        bessel.bessellsv(x,y,za,zb,self.aq.lab2[i,j,:],pot)
                        rv[:,i,j,:] = self.term2[:,i,j,:] * pot
        rv.shape = (self.Nparam,aq.Naq,self.model.Np)
        return rv
    def headinside(self,t):
        return self.model.head(self.xc,self.yc,t)[self.pylayer] - self.resfac[:,np.newaxis] / self.aq.Tcol[self.pylayer] * self.strength(t)
    def layout(self):
        return 'line', [self.x1,self.x2], [self.y1,self.y2]
    
class Well(WellBase):
    '''Well with non-zero and potentially variable discharge through time'''
    def __init__(self,model,xw=0,yw=0,rw=0.1,tsandQ=[(0.0,1.0)],res=0.0,layer=1,label=None):
        self.storeinput(inspect.currentframe())
        WellBase.__init__(self,model,xw,yw,rw,tsandbc=tsandQ,res=res,layer=layer,type='g',name='Well',label=label)
        
class LineSink(LineSinkBase):
    '''LineSink with non-zero and potentially variable discharge through time'''
    def __init__(self,model,x1=-1,y1=0,x2=1,y2=0,tsandsig=[(0.0,1.0)],res=0.0,layer=1,label=None):
        self.storeinput(inspect.currentframe())
        LineSinkBase.__init__(self,model,x1=x1,y1=y1,x2=x2,y2=y2,tsandbc=tsandsig,res=res,layer=layer,type='g',name='LineSink',label=label,addtomodel=True)

def LineSinkString(model,xy=[(-1,0),(1,0)],tsandsig=[(0.0,1.0)],res=0.0,layer=1):
    # Helper function to create string of line-sinks
    lslist = []
    xy = np.atleast_2d(xy)
    for i in range(len(xy)-1):
        ls = LineSink(model,xy[i,0],xy[i,1],xy[i+1,0],xy[i+1,1],tsandsig,res,layer)
        lslist.append(ls)
    return lslist

class ZeroMscreenWell(WellBase,MscreenEquation):
    '''MscreenWell with zero discharge. Needs to be screened in multiple layers; Head is same in all screened layers'''
    def __init__(self,model,xw=0,yw=0,rw=0.1,res=0.0,layer=[1,2],label=None):
        assert len(layer) > 1, "TTim input error: number of layers for ZeroMscreenWell must be at least 2"
        self.storeinput(inspect.currentframe())
        WellBase.__init__(self,model,xw,yw,rw,tsandbc=[(0.0,0.0)],res=res,layer=layer,type='z',name='ZeroMscreenWell',label=label)
        self.Nunknowns = self.Nparam
    def initialize(self):
        WellBase.initialize(self)
        self.parameters = np.zeros( (self.model.Ngvbc, self.Nparam, self.model.Np), 'D' )
        
class ZeroMscreenLineSink(LineSinkBase,MscreenEquation):
    '''MscreenLineSink with zero discharge. Needs to be screened in multiple layers; Head is same in all screened layers'''
    def __init__(self,model,x1=-1,y1=0,x2=1,y2=0,res=0.0,layer=[1,2],label=None):
        assert len(layer) > 1, "TTim input error: number of layers for ZeroMscreenLineSink must be at least 2"
        self.storeinput(inspect.currentframe())
        LineSinkBase.__init__(self,model,x1=x1,y1=y1,x2=x2,y2=y2,tsandbc=[(0.0,0.0)],res=res,layer=layer,type='z',name='ZeroMscreenLineSink',label=label,addtomodel=True)
        self.Nunknowns = self.Nparam
    def initialize(self):
        LineSinkBase.initialize(self)
        self.parameters = np.zeros( (self.model.Ngvbc, self.Nparam, self.model.Np), 'D' )
        
def ZeroMscreenLineSinkString(model,xy=[(-1,0),(1,0)],res=0.0,layer=1):
    # Helper function to create string of line-sinks
    lslist = []
    xy = np.atleast_2d(xy)
    for i in range(len(xy)-1):
        ls = ZeroMscreenLineSink(model,xy[i,0],xy[i,1],xy[i+1,0],xy[i+1,1],res,layer)
        lslist.append(ls)
    return lslist
        
class MscreenWell(WellBase,MscreenEquation):
    '''MscreenWell that varies through time. May be screened in multiple layers but heads are same in all screened layers'''
    def __init__(self,model,xw=0,yw=0,rw=0.1,tsandQ=[(0.0,1.0)],res=0.0,layer=[1,2],label=None):
        assert len(layer) > 1, "TTim input error: number of layers for MscreenWell must be at least 2"
        self.storeinput(inspect.currentframe())
        WellBase.__init__(self,model,xw,yw,rw,tsandbc=tsandQ,res=res,layer=layer,type='v',name='MscreenWell',label=label)
        self.Nunknowns = self.Nparam
        self.xc = self.xw + self.rw; self.yc = self.yw # To make sure the point is always the same for all elements
    def initialize(self):
        WellBase.initialize(self)
        self.parameters = np.zeros( (self.model.Ngvbc, self.Nparam, self.model.Np), 'D' )
        
class MscreenLineSink(LineSinkBase,MscreenEquation):
    '''MscreenLineSink that varies through time. Must be screened in multiple layers but heads are same in all screened layers'''
    def __init__(self,model,x1=-1,y1=0,x2=1,y2=0,tsandsig=[(0.0,1.0)],res=0.0,layer=[1,2],label=None):
        assert len(layer) > 1, "TTim input error: number of layers for MscreenLineSink must be at least 2"
        self.storeinput(inspect.currentframe())
        LineSinkBase.__init__(self,model,x1=x1,y1=y1,x2=x2,y2=y2,tsandbc=tsandsig,res=res,layer=layer,type='v',name='MscreenLineSink',label=label,addtomodel=True)
        self.Nunknowns = self.Nparam
    def initialize(self):
        LineSinkBase.initialize(self)
        self.parameters = np.zeros( (self.model.Ngvbc, self.Nparam, self.model.Np), 'D' )
        
def MscreenLineSinkString(model,xy=[(-1,0),(1,0)],tsandsig=[(0.0,1.0)],res=0.0,layer=1):
    # Helper function to create string of line-sinks
    lslist = []
    xy = np.atleast_2d(xy)
    for i in range(len(xy)-1):
        ls = MscreenLineSink(model,xy[i,0],xy[i,1],xy[i+1,0],xy[i+1,1],tsandsig,res,layer)
        lslist.append(ls)
    return lslist
        
class ZeroHeadWell(WellBase,HeadEquation):
    '''HeadWell that remains zero and constant through time'''
    def __init__(self,model,xw=0,yw=0,rw=0.1,res=0.0,layer=1,label=None):
        self.storeinput(inspect.currentframe())
        WellBase.__init__(self,model,xw,yw,rw,tsandbc=[(0.0,0.0)],res=res,layer=layer,type='z',name='ZeroHeadWell',label=label)
        self.Nunknowns = self.Nparam
        self.xc = self.xw + self.rw; self.yc = self.yw # To make sure the point is always the same for all elements
    def initialize(self):
        WellBase.initialize(self)
        self.parameters = np.zeros( (self.model.Ngvbc, self.Nparam, self.model.Np), 'D' )
        
class ZeroHeadLineSink(LineSinkBase,HeadEquation):
    '''HeadLineSink that remains zero and constant through time'''
    def __init__(self,model,x1=-1,y1=0,x2=1,y2=0,res=0.0,layer=1,label=None):
        self.storeinput(inspect.currentframe())
        LineSinkBase.__init__(self,model,x1=x1,y1=y1,x2=x2,y2=y2,tsandbc=[(0.0,0.0)],res=res,layer=layer,type='z',name='ZeroHeadLineSink',label=label,addtomodel=True)
        self.Nunknowns = self.Nparam
    def initialize(self):
        LineSinkBase.initialize(self)
        self.parameters = np.zeros( (self.model.Ngvbc, self.Nparam, self.model.Np), 'D' )
        
def ZeroHeadLineSinkString(model,xy=[(-1,0),(1,0)],res=0.0,layer=1):
    # Helper function to create string of line-sinks
    lslist = []
    xy = np.atleast_2d(xy)
    for i in range(len(xy)-1):
        ls = ZeroHeadLineSink(model,xy[i,0],xy[i,1],xy[i+1,0],xy[i+1,1],res,layer)
        lslist.append(ls)
    return lslist
    
class HeadWell(WellBase,HeadEquation):
    '''HeadWell of which the head varies through time. May be screened in multiple layers but all with the same head'''
    def __init__(self,model,xw=0,yw=0,rw=0.1,tsandh=[(0.0,1.0)],res=0.0,layer=1,label=None):
        self.storeinput(inspect.currentframe())
        WellBase.__init__(self,model,xw,yw,rw,tsandbc=tsandh,res=res,layer=layer,type='v',name='HeadWell',label=label)
        self.Nunknowns = self.Nparam
        self.xc = self.xw + self.rw; self.yc = self.yw # To make sure the point is always the same for all elements
    def initialize(self):
        WellBase.initialize(self)
        self.parameters = np.zeros( (self.model.Ngvbc, self.Nparam, self.model.Np), 'D' )
        self.pc = self.aq.T[self.pylayer] # Needed in solving; We solve for a unit head
        
class HeadLineSink(LineSinkBase,HeadEquation):
    '''HeadLineSink of which the head varies through time. May be screened in multiple layers but all with the same head'''
    def __init__(self,model,x1=-1,y1=0,x2=1,y2=0,tsandh=[(0.0,1.0)],res=0.0,layer=1,label=None):
        self.storeinput(inspect.currentframe())
        LineSinkBase.__init__(self,model,x1=x1,y1=y1,x2=x2,y2=y2,tsandbc=tsandh,res=res,layer=layer,type='v',name='HeadLineSink',label=label,addtomodel=True)
        self.Nunknowns = self.Nparam
    def initialize(self):
        LineSinkBase.initialize(self)
        self.parameters = np.zeros( (self.model.Ngvbc, self.Nparam, self.model.Np), 'D' )
        self.pc = self.aq.T[self.pylayer] # Needed in solving; We solve for a unit head
        
def HeadLineSinkString(model,xy=[(-1,0),(1,0)],tsandh=[(0.0,1.0)],res=0.0,layer=1):
    # Helper function to create string of line-sinks
    lslist = []
    xy = np.atleast_2d(xy)
    for i in range(len(xy)-1):
        ls = HeadLineSink(model,xy[i,0],xy[i,1],xy[i+1,0],xy[i+1,1],tsandh,res,layer)
        lslist.append(ls)
    return lslist
    
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
    plt.show()
                
def timcontour( ml, xmin, xmax, nx, ymin, ymax, ny, levels = 10, t=0.0, layer = 1,\
               color = 'k', lw = 0.5, style = 'solid',layout = True, newfig = True, \
               labels = False, labelfmt = '%1.2f'):
    '''Contour heads with pylab'''
    print 'grid of '+str((nx,ny))+'. gridding in progress. hit ctrl-c to abort'
    h = ml.headgrid(xmin,xmax,nx,ymin,ymax,ny,t,layer)  # h[Nlayers,Ntimes,Ny,Nx]
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
    if layout: pylayout(ml,ax)
    # Contour
    plt.rcParams['contour.negative_linestyle'] = 'solid'
    if color is None:
        a = ax.contour( xg, yg, h[0,0], levels, linewidths = lw, linestyles = style )
    else:
        a = ax.contour( xg, yg, h[0,0], levels, colors = color[0], linewidths = lw, linestyles = style )
    if labels and not fill:
        ax.clabel(a,fmt=labelfmt)
    plt.show()
                
def pylayout( ml, ax, color = 'k', lw = 0.5, style = '-' ):
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
    
##ml = Model3D(kaq=2.0,z=[10,5,0],Saq=[.002,.001],kzoverkh=0.2,phreatictop=False,tmin=.1,tmax=10,M=15)
#ml = ModelMaq(kaq=[10,5],z=[4,2,1,0],c=[100],Saq=[1e-3,1e-4],Sll=[1e-6],tmin=0.1,tmax=10,M=15)
#ls1 = LineSink(ml,-10,-10,0,-5,tsandsig=[(0,.05),(1,.02)],res=1.0,layer=[1,2],label='mark1')
#ls2 = LineSink(ml,0,-5,10,10,tsandsig=[(0,.03),(2,.07)],layer=[1],label='mark2')
##ls3 = HeadLineSink(ml,-10,5,0,5,tsandh=[(0,0.02),(3,0.01)],res=0.0,layer=[1,2])
#lss = HeadLineSinkString(ml,[(-10,5),(-5,5),(0,5)],tsandh=[(0,0.02),(3,0.01)],res=0.0,layer=[1,2])
##ls3 = ZeroMscreenLineSink(ml,-10,5,0,5,res=1.0,layer=[1,2])
##tQ = np.array([
##    (0,5),
##    (1,2)])
##w1 = Well(ml,0,0,.1,tsandQ=tQ,res=1.0,layer=[1,2])
##w2 = Well(ml,100,0,.1,tsandQ=[(0,3),(2,7)],layer=[1])
##w3 = MscreenWell(ml,0,100,.1,tsandQ=[(0,2),(3,1)],res=2.0,layer=[1,2])
##w3 = ZeroMscreenWell(ml,0,100,.1,res=2.0,layer=[1,2])
##w3 = ZeroHeadWell(ml,0,100,.1,res=1.0,layer=[1,2])
##w3 = HeadWell(ml,0,100,.1,tsandh=[(0,2),(3,1)],res=1.0,layer=[1,2])
#ml.solve()
##print ml.potential(2,3,[.5,5])
#print ml.potential(50,50,[0.5,5])

#print 'Q from strength:  ',w3.strength(.5)
#print 'Q from head diff: ',(ml.head(w3.xc,w3.yc,.5)-w3.headinside(.5))/w3.res*2*np.pi*w3.rw*ml.aq.Haq[:,np.newaxis]
#print 'Q from head diff: ',(ml.head(w3.xc,w3.yc,.5)-2.0)/w3.res*2*np.pi*w3.rw*ml.aq.Haq[:,np.newaxis]
#print w3.strength([.5,5])
#print ls3.strength([.5,5])
#print sum(ls3.strength([.5,5]),0)
#Q = w3.strength([.5,5])
#print sum(Q,0)
#print ml.potential(w3.xc,w3.yc,[.5,5])