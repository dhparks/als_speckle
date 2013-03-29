__kernel void execute(
    __global float2* in)

    {	
	int i = get_global_id(0);
	in[i] = (float2)(0.f,0.f);
    }