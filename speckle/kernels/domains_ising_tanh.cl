__kernel void execute(
    float a,
    __global float* data)

    {	
	// this function clamps the output from the rescaling
	// by definition, the magnetization must be real, so
	// the real component is bounded while the imaginary
	// component is set to zero
	
	int i = get_global_id(0);

	float x = a*data[i];
	float k = native_exp(-2*x);
	float j = native_exp(-2*a);
	float t1 = (1-k)*(1+j);
	float t2 = native_recip((1+k)*(1-j));
	float z = t1*t2;
	data[i] = clamp(z,-1.0f,1.0f);
    
    }