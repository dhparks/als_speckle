__kernel void execute(
    __global float2* in)

    {	
	int i = get_global_id(0);
	
	// pull the components into registers
	
	in[i] = (float2)(in[i].x,0.0f);
    }