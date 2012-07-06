import numpy as np
DFT = np.fft.fft2
IDFT = np.fft.ifft2
shift = np.fft.fftshift

class random_walk():
    """ Methods for generating random-walk simulations (more accurately, wiener-processes)"""

    def __init__(self,N,n,radius=0):
        
        assert isinstance(n,int), "number of balls must be int"
        assert isinstance(radius, (int,float)), "ball radius must be float or int"
        
        self.N = N
        
        if radius == 0:
            self.do_convolve == False
        if radius > 0:
            from .. import shape
            obj = shape.circle((self.N,self.N),radius)
            self.dft_object = DFT(shift(obj))
            self.do_convolve = True
        
        self.coordinates = (self.N*np.random.rand(2,n)).astype(float)

    def displace(self,stepsize):
        
        assert isinstance(stepsize,(int,float)), "step size must be float or int"
        
        deltas = stepsize*np.random.randn(2,len(self.coordinates[0]))
        
        self.coordinates += deltas
        self.coordinates = np.mod(self.coordinates,self.N).astype('int')
        
    def place_on_grid(self):
        
        grid = np.zeros((self.N,self.N),float)
    
        # i don't know how to do this without a loop, barf
        for site in range(len(self.coordinates[0])):
            y,x = self.coordinates[:,site]
            grid[y,x] = 1.
            
        if self.do_convolve:
            grid = np.clip(abs(IDFT(DFT(grid)*self.dft_object)),0.,1.)
            
        return 1.-grid