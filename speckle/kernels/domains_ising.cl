__kernel void execute(
    float a,
    __global float* domains)

    {	
	// this function clamps the output from the rescaling
	// by definition, the magnetization must be real, so
	// the real component is bounded while the imaginary
	// component is set to zero
	
	int i = get_global_id(0);
	
	float x = domains[i];
	float y = pown(x,3);
	domains[i] = (1+a)*x-a*y;
    
    }