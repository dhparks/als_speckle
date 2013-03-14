__kernel void execute(
    __global float2* in1,
    __global float2* in2,
    __global float2* out)

    {	
	int i = get_global_id(0);
	a = in1[i];
	b = in2[i];
	out[i] = (float2)(a.x+b.x,a.y+b.y);
	
    }