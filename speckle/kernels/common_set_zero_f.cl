__kernel void execute(
    __global float* in)

    {	
	int i = get_global_id(0);
	in[i] = 0.f;
    }