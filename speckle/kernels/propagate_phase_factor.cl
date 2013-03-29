__kernel void execute(
    __global float* r,
    float z,
    __global float2* out)

    {	
	int i = get_global_id(0);
	
	out[i] = (float2)(native_cos(z*r[i]),native_sin(-1*z*r[i]));
	
    }