__kernel void execute(
    __global float* in1,
    __global float2* in2,
    __global float2* out)

    {	
	int i = get_global_id(0);
	
	// pull the components into registers
	float a = in2[i].x;
	float b = in2[i].y;
	float c = in1[i];
	out[i] = (float2)(a*c,b*c);
    }