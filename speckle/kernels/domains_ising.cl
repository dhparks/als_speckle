__kernel void execute(
    float a,
    int mode,
    __global float* data)

    {	
	// this function clamps the output from the rescaling
	// by definition, the magnetization must be real, so
	// the real component is bounded while the imaginary
	// component is set to zero
	
	int i = get_global_id(0);
	
	float z = 0;
	
	if (mode == 0) {
	    float x = data[i];
	    //float y = pown(x,3);
	    z = (1+a)*x-a*x*x*x;
	};
	
	if (mode == 1) {
	    float x = a*data[i];
	    float k = native_exp(-2*x);
	    float j = native_exp(-2*a);
	    float t1 = (1-k)*(1+j);
	    float t2 = native_recip((1+k)*(1-j));
	    z = t1*t2;
	    //z = tanh(x)/tanh(a);
	}

	data[i] = clamp(z,-1.0f,1.0f);
    
    }