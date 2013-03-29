__kernel void execute(
    __global float* in,
    __global float* out)

    {	
	int i = get_global_id(0);
	
	// pull the components into registers

	out[i] = native_sqrt(in[i]);
    }