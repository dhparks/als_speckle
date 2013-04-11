import numpy as np
DFT = np.fft.fft2
IDFT = np.fft.ifft2
shift = np.fft.fftshift

class random_walk():
    """ Methods for generating random-walk simulations (more accurately, wiener-processes)"""

    def __init__(self,N,n,radius=0):
        
        try:
            n = int(n)
            N = int(N)
            radius = float(radius)
        except ValueError:
            print "inputs to random_walk must be numbers"
            exit()
        
        self.N = N
        
        if radius == 0:
            self.do_convolve = False
        if radius > 0:
            from .. import shape
            obj = shape.circle((self.N,self.N),radius)
            self.dft_object = DFT(shift(obj))
            self.do_convolve = True
        
        self.coordinates = (self.N*np.random.rand(2,n)).astype(float)

    def displace(self,stepsize):
        """ Add some random offsets to the current coordinates. """
      
        try: stepsize = float(stepsize) # fails is if stepsize is not a number
        except ValueError: print "cant cast stepsize %s to float"%stepsize
        
        deltas = stepsize*np.random.randn(2,len(self.coordinates[0]))
        
        self.coordinates += deltas
        self.coordinates = np.mod(self.coordinates,self.N)
        
    def place_on_grid(self):
        """ Take self.coordinates and make those coordinates 1.
        If the size of the balls is greater than 1 pixel, convolve with the
        ball shape. """
        
        grid = np.zeros((self.N,self.N),float)
        grid[self.coordinates.tolist()] = 1.
        
        if self.do_convolve:
            grid = np.clip(abs(IDFT(DFT(grid)*self.dft_object)),0.,1.)
            
        return grid