__kernel void execute(
    __global float2* in1,
    __global float2* out)

    {	
	int i = get_global_id(0);
	
	// pull the components into registers
	float a = in1[i].x;
	float b = in1[i].y;
	float h = hypot(a,b);
	
	float re = native_sqrt((h+a)/2);
	float im = sign(b)*native_sqrt((h-a)/2);

	out[i] = (float2)(re,im);
    }