__kernel void execute(
    __global float* in,
    __global float2* out)

    {	
	int i = get_global_id(0);
	out[i] = (float2)(in[i],0);
	
    }