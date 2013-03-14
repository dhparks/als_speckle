__kernel void execute(
    __global float2* in,
    __global float2* out)

    {	
	int i = get_global_id(0);
	
	// pull the components into registers
	out[i] = in[i];
    }