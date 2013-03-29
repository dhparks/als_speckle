__kernel void execute(
    __global float* in1,
    __global float* in2,
    __global float* out)

    {	
	int i = get_global_id(0);
	
	// pull the components into registers
	out[i] = in1[i]+in2[i];
    }