__kernel void execute(
    __global float2* in1,
    __global float* in2,
    __global float2* out)

    {	
	int i = get_global_id(0);
	
	// pull the components into registers
	float a = in1[i].x;
	float b = in1[i].y;
	float c = in2[i];

	out[i] = (float2)(a/c,b/c);
    }