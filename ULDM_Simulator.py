import os
import scipy
import timeit
import pyfftw
import numpy as np 
import matplotlib.pyplot as plt

from scipy.fft import fftfreq, fftn, ifftn
from matplotlib import animation
from scipy.special import k0, k1
from tqdm.notebook import tqdm
from scipy.interpolate import RegularGridInterpolator, interpn

class ULDM_Simulator():
    '''
    Schrodinger-Poisson Solver
    '''
    def __init__(self, dist='Iso_Gaussian', L=5, N=64, kJ=1e-3, compute_acc=False, TU=1):
        '''
        dist(str)       : Distribution type
        L   (scalar)    : Length of the box
        N   (scalar)    : Number of grid points in each dimension
        kJ  (scalar)    : Jeans wavelength = 2 (pi G rho)^(1/4) * m^1/2
        TU  (scalar)    : Total simulation time unit [L^2/ pi]
        '''
        self.kJ = kJ
        self.TU = TU

        self.set_grid(L, N)
        self.set_steps()
    
        self.f = self.set_distribution(dist)    
        self.set_initial_wavefunction(compute_acc)
    
    def set_grid(self, L: float, N: float):
        '''
        L (scalar): Length of the box
        N (scalar): Number of grid points in each dimension
        '''

        self.N = N 
        self.L = L
        self.dx = L / N

        # Set up the spatial grid
        self.coordinate = (
            np.linspace(-L/2, L/2, N),
            np.linspace(-L/2, L/2, N),
            np.linspace(-L/2, L/2, N)
        )

        self.X, self.Y, self.Z = np.meshgrid(
            np.linspace(-L/2, L/2, N),
            np.linspace(-L/2, L/2, N),
            np.linspace(-L/2, L/2, N),
            indexing='ij'
        )

        # Set up the Fourier grid
        self.KX, self.KY, self.KZ = np.meshgrid(
            fftfreq(self.N, self.dx) * 2 * np.pi, 
            fftfreq(self.N, self.dx) * 2 * np.pi, 
            fftfreq(self.N, self.dx) * 2 * np.pi,
            indexing='ij'
        )

        # 1 / k^2
        self.K2 = self.KX**2 + self.KY**2 + self.KZ**2
        self.invK2 = np.divide(1, self.K2, out=np.zeros((self.N, self.N, self.N)), where=(self.K2 != 0))
    
    def set_steps(self):
        self.T = self.L**2 / np.pi * self.TU
        self.nt = self.N**2
        self.dt = self.T / self.nt

        self.time = np.arange(0, self.T, self.dt)

    def set_distribution(self, dist='Iso_Gaussian'):
        '''
        dist (str): Distribution type
        '''
        if dist == 'Iso_Gaussian':
            return lambda k: (2 * np.pi)**1.5 * np.exp(-k**2 / 2)
        
    def set_initial_wavefunction(self, compute_acc):
        self.farr = self.f(np.sqrt(self.KX**2 + self.KY**2 + self.KZ**2))

        PSI = np.random.rayleigh(size=self.farr.shape).astype('complex128')
        PSI *= 1 / self.L**1.5
        PSI *= np.exp(2j * np.pi * np.random.rand(self.N, self.N, self.N))
        PSI *= np.sqrt(self.farr / 2)
        
        self.psi = ifftn(PSI, norm='forward')                       
        self.rhob = np.mean(np.abs(self.psi)**2)    # average density

        self.Phi_fourier = -(self.kJ**4 / 4) * fftn((np.abs(self.psi)**2 - self.rhob)) * self.invK2
        self.Phi = np.real(ifftn(self.Phi_fourier))
        
        if compute_acc:
            self.uldm_acc = [
                np.real(ifftn(-1j * self.KX * self.Phi_fourier)),
                np.real(ifftn(-1j * self.KY * self.Phi_fourier)),
                np.real(ifftn(-1j * self.KZ * self.Phi_fourier))
            ]

    def evolve(self):
        '''
        Evolve field according to kick-drift-kick scheme
        '''
        self.psi *= np.exp(-0.5j * self.Phi * self.dt)  # kick
        self.psi = fftn(self.psi)
        self.psi *= np.exp(-0.5j * self.K2 * self.dt)   # drift
        self.psi = ifftn(self.psi)

        self.rhob = self.N**(-3) * np.sum(np.abs(self.psi)**2)

        self.Phi_fourier = fftn(-(self.kJ**4 / 4) * (np.abs(self.psi)**2 - self.rhob) ) * self.invK2
        self.Phi = np.real(ifftn(self.Phi_fourier))

        self.psi *= np.exp(-0.5j * self.Phi * self.dt)  # kick
    
    def solve(self, save_rho=True):
        if save_rho == True:
            self.rho = np.zeros(len(self.time))
            for i, _ in enumerate(tqdm(self.time)):
                self.rho[i] = (np.abs(self.psi)**2)[0,0,0]
                self.evolve()



class ULDM_FreeParticle(ULDM_Simulator):
    '''
    Schrodinger-Poisson Solver
    + Free Particle Evolution
    '''
    def __init__(self, dist='Iso_Gaussian', L=5, N=64, kJ=1e-3):
        super().__init__(dist=dist, L=L, N=N, kJ=kJ)
        self.set_initial_kinematics()

    def set_initial_kinematics(self):
        self.grid = np.linspace(-self.L/2, self.L/2, self.N)

        self.ax = np.real(ifftn(-1j * self.KX * self.Phi_fourier))
        self.ay = np.real(ifftn(-1j * self.KY * self.Phi_fourier))
        self.az = np.real(ifftn(-1j * self.KZ * self.Phi_fourier))

        self.pos = np.array([0, 0, 0])
        self.vel = np.array([0, 0, 0])
        self.acc = np.array([interpn(self.coordinate, self.ax, self.pos)[0],
                             interpn(self.coordinate, self.ay, self.pos)[0],
                             interpn(self.coordinate, self.az, self.pos)[0]])
        
    # THIS OVERRIDES evolve METHOD IN PARENT CLASS
    def evolve(self):
        '''
        Evolve field according to kick-drift-kick scheme
        Evolve particle according to drift-kick-drift (leapfrog)
        '''
        
        # Initial kick - drift sequence
        self.psi *= np.exp(-0.5j * self.Phi * self.dt)
        self.psi = fftn(self.psi)
        self.psi *= np.exp(-0.5j * self.K2 * self.dt)   
        self.psi = ifftn(self.psi)

        # Update Phi and acceleration
        self.rhob = self.N**(-3) * np.sum(np.abs(self.psi)**2)
        
        self.Phi_fourier = fftn(-(self.kJ**4 / 4) * (np.abs(self.psi)**2 - self.rhob)) * self.invK2
        self.Phi = np.real(ifftn(self.Phi_fourier))
        
        self.ax = np.real(ifftn(-1j * self.KX * self.Phi_fourier))
        self.ay = np.real(ifftn(-1j * self.KY * self.Phi_fourier))
        self.az = np.real(ifftn(-1j * self.KZ * self.Phi_fourier))
        
        self.psi *= np.exp(-0.5j * self.Phi * self.dt)

        # Free particle evolution
        self.pos = self.pos + 0.5 * self.vel * self.dt
        self.acc = np.array([interpn(self.coordinate, self.ax, self.pos)[0],
                             interpn(self.coordinate, self.ay, self.pos)[0],
                             interpn(self.coordinate, self.az, self.pos)[0]])
        self.vel = self.vel + self.acc * self.dt
        self.pos = self.pos + 0.5 * self.vel * self.dt

    # THIS OVERRIDES solve METHOD IN PARENT CLASS
    def solve(self, save=True, progress=True):
        if progress:
            ite = tqdm(self.time)
        else:
            ite = self.time

        if save == True:
            self.rho = np.zeros(len(self.time))
            self.pos_arr = np.zeros((len(self.time), 3))
            self.vel_arr = np.zeros((len(self.time), 3))
            self.acc_arr = np.zeros((len(self.time), 3))

            for i, _ in enumerate(ite):
                self.rho[i] = (np.abs(self.psi)**2)[0,0,0]
                self.pos_arr[i] = self.pos
                self.vel_arr[i] = self.vel
                self.acc_arr[i] = self.acc
                self.evolve()
                
class ULDM_HarmonicOscillator(ULDM_Simulator):
    '''
    Schrodinger-Poisson Solver
    + Harmonic force along 1D / Free in another 2D
    '''
    def __init__(self, f_osc, dist='Iso_Gaussian', L=5, N=64, kJ=1e-3):
        super().__init__(dist=dist, L=L, N=N, kJ=kJ)
        self.f_osc = f_osc
        self.set_initial_kinematics()
        self.time = np.arange(0, self.T, self.dt)

    def set_initial_kinematics(self):
        self.grid = np.linspace(-self.L/2, self.L/2, self.N)

        self.ax = np.real(ifftn(-1j * self.KX * self.Phi_fourier))
        self.ay = np.real(ifftn(-1j * self.KY * self.Phi_fourier))
        self.az = np.real(ifftn(-1j * self.KZ * self.Phi_fourier))

        self.pos = np.array([1, 0, 0])
        self.vel = np.array([0, 0, 0])

        self.acc = np.array([interpn(self.coordinate, self.ax, self.pos)[0] - (2 * np.pi * self.f_osc)**2 * self.pos[0],
                             interpn(self.coordinate, self.ay, self.pos)[0],
                             interpn(self.coordinate, self.az, self.pos)[0]])

    # THIS OVERRIDES evolve METHOD IN PARENT CLASS
    def evolve(self):
        '''
        Evolve field according to kick-drift-kick scheme
        Evolve particle according to drift-kick-drift (leapfrog)
        '''
        
        # Initial kick - drift sequence
        self.psi *= np.exp(-0.5j * self.Phi * self.dt)
        self.psi = fftn(self.psi)
        self.psi *= np.exp(-0.5j * self.K2 * self.dt)   
        self.psi = ifftn(self.psi)

        # Update Phi and acceleration
        self.rhob = self.N**(-3) * np.sum(np.abs(self.psi)**2)
        
        self.Phi_fourier = fftn(-(self.kJ**4 / 4) * (np.abs(self.psi)**2 - self.rhob)) * self.invK2
        self.Phi = np.real(ifftn(self.Phi_fourier))
        
        self.ax = np.real(ifftn(-1j * self.KX * self.Phi_fourier))
        self.ay = np.real(ifftn(-1j * self.KY * self.Phi_fourier))
        self.az = np.real(ifftn(-1j * self.KZ * self.Phi_fourier))
        
        self.psi *= np.exp(-0.5j * self.Phi * self.dt)
        
        def force(pos):
            return np.array([interpn(self.coordinate, self.ax, pos)[0],
                             interpn(self.coordinate, self.ay, pos)[0],
                             interpn(self.coordinate, self.az, pos)[0]]) + self.harmonic_force(pos)

        # For free particle evolution
        self.pos, self.vel, self.acc = self.yoshida(self.pos, self.vel, force)
        
    def leapfrog(self, pos, vel, force):
        '''
        Leapfrog integrator
        '''
        dt = self.dt
        
        x = pos + 0.5 * vel * dt
        v = vel + force(x) * dt
        x = x + 0.5 * v * dt

        return x, v, force(x)

    def yoshida(self, pos, vel, force):
        '''
        4th order Yoshida integrator
        '''
        dt = self.dt

        w0 = - 2**(1 / 3) / (2 - 2**(1 / 3))
        w1 = 1 / (2 - 2**(1/3))
        c1 = c4 = w1 / 2
        c2 = c3 = (w0 + w1) / 2
        d1 = d3 = w1
        d2 = w0

        x1 = pos + c1 * vel * dt
        v1 = vel + d1 * force(x1) * dt
        
        x2 = x1 + c2 * v1 * dt
        v2 = v1 + d2 * force(x2) * dt
        
        x3 = x2 + c3 * v2 * dt
        v3 = v2 + d3 * force(x3) * dt

        x4 = x3 + c4 * v3 * dt
        v4 = v3

        return x4, v4, force(x4)

    def harmonic_force(self, pos):
        return np.array([-(2 * np.pi * self.f_osc)**2 * pos[0], 0, 0])

    # THIS OVERRIDES solve METHOD IN PARENT CLASS
    def solve(self, save=True):
        if save == True:
            self.rho = np.zeros(len(self.time))
            self.pos_arr = np.zeros((len(self.time), 3))
            self.vel_arr = np.zeros((len(self.time), 3))
            self.acc_arr = np.zeros((len(self.time), 3))

            for i, _ in enumerate(tqdm(self.time)):
                self.rho[i] = (np.abs(self.psi)**2)[0,0,0]
                self.pos_arr[i] = self.pos
                self.vel_arr[i] = self.vel
                self.acc_arr[i] = self.acc
                self.evolve()

    def subtract(self, truth=True):
        '''
        Subtracting motion due to haronic force
        Input
            truth (bool)    if true, subtraction done by prior knowledge of injected force
        '''
        if truth:
            self.x_sub = self.pos_arr[:,0] - np.cos(2 * np.pi * self.f_osc * self.time)
            self.a_sub = self.acc_arr[:,0] + (2 * np.pi * self.f_osc)**2 * self.pos_arr[:,0]