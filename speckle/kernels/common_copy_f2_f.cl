__kernel void execute(
    __global float2* in,
    __global float* out)

    {	
	int i = get_global_id(0);
	out[i] = in[i].x;
    }