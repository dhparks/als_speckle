__kernel void execute(
    __global float2* in,
    __global float2* out)

    {	
	int i = get_global_id(0);
	
	// pull the components into registers
	float a = in[i].x;
	float b = in[i].y;
	
	// do math
	out[i] = (float2)(a*a+b*b,0);
    }