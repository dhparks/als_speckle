__kernel void execute(
    __global float *domains,
    __global uchar *available,
    float x)

    {	
        int i = get_global_id(0);
		uchar t = 1;
	
	if (available[i] = t){
	    float d = domains[i];
	    domains[i] = d+x;}
    }