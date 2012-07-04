import numpy as np
DFT = np.fft.fft2
IDFT = np.fft.ifft2
shift = np.fft.fftshift

class random_walk():
    """ Methods for generating random-walk simulations (more accurately, wiener-processes)"""

    def __init__(self,n,radius=0):
        
        assert isinstance(n,int), "number of balls must be int"
        assert isinstance(radius, (int,float)), "ball radius must be float or int"
        
        if radius == 0:
            self.do_convolve == False
        if radius > 0:
            self.dft_object = DFT(shift(speckle.shape.circle((cp.N,cp.N),radius)))
            self.do_convolve = True
        
        self.coordinates = (cp.N*np.random.rand(2,n)).astype(float)

    def displace(self,stepsize):
        
        assert isinstance(stepsize,(int,float)), "step size must be float or int"
        
        deltas = stepsize*np.random.randn(2,len(self.coordinates))
        self.coordinates += deltas
        self.coordinates = np.mod(self.coordinates,cp.N)
        
    def place_on_grid(self):
        
        grid = np.zeros((cp.N,cp.N),float)
    
        # i don't know how to do this without a loop, barf
        sites = len(self.coordinates)
        for i in range(sites):
            grid[self.coordinates] = 1
            
        if self.do_convolve:
            grid = np.clip(abs(IDFT(DFT(grid)*self.dft_object)),0,1)
            
        return 1-grid