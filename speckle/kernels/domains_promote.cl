__kernel void execute(
    __global float *domains,
    __global float *available,
    float x)

    {	
        int i = get_global_id(0);
	float a = available[i];
	domains[i] += a*x;
    }