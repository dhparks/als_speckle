__kernel void execute(
    __global float2* in,
    __global float* out)

    {	
	int i = get_global_id(0);
	
	// pull the components into registers
	float a = in[i].x;
	float b = in[i].y;
	float h = hypot(a,b);

	out[i] = native_sqrt(h);
    }