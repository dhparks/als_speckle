__kernel void execute(
    __global float* in,
    __global float* out)

    {	
	// this function clamps the output from the rescaling
	// by definition, the magnetization must be real, so
	// the real component is bounded while the imaginary
	// component is set to zero
	
	int i = get_global_id(0);
	out[i] = clamp(in[i],-1.0f,1.0f);
    
    }