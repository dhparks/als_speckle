__kernel void execute(
    __global float2* dft,
    __global float* goal,
    __global float2* blurred) 					  
    {	
	int i = get_global_id(0);
	
	// pull from global
	float g = goal[i];
	float b = blurred[i].x;
	float2 d = dft[i];
	
	// make the rescaler
	float r = native_sqrt(g/b);
	if (isnan(r)) {r = 0;}
	
	// put the rescaled dft back in place
	dft[i] = (float2)(d.x*r,d.y*r);
    }