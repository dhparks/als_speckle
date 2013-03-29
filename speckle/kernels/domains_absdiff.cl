__kernel void execute(
    __global float *a1,
    __global float *a2,
    __global float *a3)

    {	
	int i = get_global_id(0);

	a3[i] = fabs(a1[i]-a2[i]);
	
    }