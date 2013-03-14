__kernel void execute(
    __global float* in1,
    __global float* in2
    __global float* out)

    {	
	int i = get_global_id(0);
	
	// pull the components into registers
	float a = in[i];
	float b = in[i];
	out[i] = native_sqrt(a*a+b*b);
    }