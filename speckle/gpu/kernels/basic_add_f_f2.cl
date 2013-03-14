__kernel void execute(
    __global float* in1,
    __global float2* in2,
    __global float2* out)

    {	
	int i = get_global_id(0);
	b = in2[i];
	out[i] = (float2)(in1[i]+b.x,b.y);
	
    }