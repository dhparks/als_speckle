__kernel void execute(
    __global float2* in1,
    __global float2* in2,
    __global float2* out)

    {	
	int i = get_global_id(0);
	float2 a = in1[i];
	float2 b = in2[i];
	out[i] = (float2)(a.x+b.x,a.y+b.y);
	
    }