__kernel void execute(
    __global float2* in,
    __global float2* out)

    {

	int i = get_global_id(0);
	int j = get_global_id(1);
	int N = get_global_size(0);
	
	int i2 = N-i-1;
	int j2 = N-j-1;
	
	int k1 = j+i*N;
	int k2 = j2+i2*N;
	
	out[k2] = in[k1];
	
    }