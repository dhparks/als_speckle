import numpy

class permutations:
    
    """ Defines a class whose methods allow easy access to permutations of list
    elements. To use, instantiate via
    
    foo = permutations(list1,list2,list3...listn)
    
    where the lists hold elements to be iterated. Permutations of elements can
    then be accessed one at a time through foo.get(0), foo.get(1), etc, or all
    at once through foo.all().
    
    The goal of this class is to avoid code such as:
    
    for i in list1:
        for j in list2:
            for k in list3:
                do_something(i,j,k)
                
    which instead becomes something like:
    
    foo = permutations(list1,list2,list3).all()
    for bar in foo:
        i,j,k = bar
        do_something(i,j,k)    
    """
    
    def __init__(self,*args):
        self.items = []
        self.n_items = len(args)
        self.lengths = numpy.ones(len(args)+1)
        for n,arg in enumerate(args):
            try:
                self.lengths[n+1] = len(arg)
                self.items.append(arg)
            except TypeError:
                self.lengths[n+1] = 1
                self.items.append((arg,))
        modulos =  numpy.cumprod(self.lengths)
        self.num = int(modulos[-1])
        self.modulos1 = (modulos[:-1]).astype(int)
        self.modulos2 = (self.lengths[1:]).astype(int)
        
    def get(self,n):
        toreturn = []
        assert isinstance(n,int), "must be int"
        for m in range(self.n_items):
            modulo1 = self.modulos1[m]
            modulo2 = self.modulos2[m]
            index = (n/modulo1)%modulo2 # this is the magic right here
            toreturn.append(self.items[m][index])
        return toreturn
        
    def all(self):
        toreturn = []
        for m in range(self.num):
            toreturn.append(self.get(m))
        return toreturn
    
# example usage:
#permute = permutations([0,1],['a','b','c'],[9,10],4)
#print permute.get(0) # return just the first permutation
#print permute.all() # return all permutations as a list
    
        
    