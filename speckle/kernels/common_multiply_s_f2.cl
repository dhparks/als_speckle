__kernel void execute(
    float s,
    __global float2* in,
    __global float2* out)

    {	
	int i = get_global_id(0);
	float2 j = in[i];
	out[i] = (float2)(s*j.x,s*j.y);
    }