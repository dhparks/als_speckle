__kernel void execute(
    __global float *domains,
    __global float *boundary,
    __global float *bv)

    {	
        int i = get_global_id(0);

	float d = domains[i];
	float b = boundary[i];
	
	domains[i] = d*(1-b)+b*bv[i];
	
    }