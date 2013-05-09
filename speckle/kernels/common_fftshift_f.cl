__kernel void execute(__global float* in,
		      __global float* out)
        
        {
	    
	int i = get_global_id(0);
	int j = get_global_id(1);
	int N = get_global_size(0);
	int M = get_global_size(1);
	
	int io = i+N/2;
	if (i >= N/2) {io = io-N;}
	
	int jo = j+M/2;
	if (j >= M/2) {jo = jo-M;}
	
	int idx_o = jo*N+io;
	int idx_i = j*N+i;
	out[idx_o] = in[idx_i]; 
	
	}