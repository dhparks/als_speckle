__kernel void execute(
    __global float2* in,
    __global float2* out)

    {	
	int i = get_global_id(0);
	int j = get_global_id(1);
	int N = get_global_size(1);
	
	int ii = i*N+j;
	
	int i2 = N-i-1;
	int j2 = N-j-1;
	
	int io = i2*N+j2;
	
	out[io] = in[ii];
	
    }