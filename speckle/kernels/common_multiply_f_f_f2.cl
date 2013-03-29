__kernel void execute(
    __global float* in1,
    __global float* in2,
    __global float2* out)

    {	
	int i = get_global_id(0);
	
	// pull the components into registers
	
	out[i] = (float2)(in1[i]*in2[i],0.0f);
    }