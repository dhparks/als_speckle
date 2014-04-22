__kernel void execute(
    __global float2* in,
    __global float2* out)

    {	
	int i = get_global_id(0);
	
	// pull the components into registers
	float a = in[i].x;
	float b = in[i].y;
	out[i] = (float2)(native_sqrt(a*a+b*b),0);
    }