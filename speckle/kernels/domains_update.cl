__kernel void execute(
    __global float* domains,
    __global float* incoming,
    __global bool* walls)

    {	
	// this function clamps the output from the rescaling
	// by definition, the magnetization must be real, so
	// the real component is bounded while the imaginary
	// component is set to zero
	
	
	int i = get_global_id(0);
	bool w = walls[i];
	bool t = 0;
	float o = 0;
	
	if (w = t) {domains[i] = incoming[i];}
    
    }