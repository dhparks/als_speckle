__kernel void execute(
    __global float* gaussian,
    float sx, float sy)  // these are the coherence lengths; should be in PIXELS

    {	
	int i = get_global_id(0);
	int j = get_global_id(1);
	int N = get_global_size(0);
	int index = i*N+j;
	
	// transform from index into coordinates
	if (i > N/2){i = i-N;}
	if (j > N/2){j = j-N;}
	
	// calculate the value of the gaussian
	float i2 = (float)i;
	float j2 = (float)j;

	//float pi = 3.14159265358979323846f;
	//float norm = native_recip(2*sx*sy*pi);

	float vy = native_exp(native_divide(-1*i2*i2,2*sx*sx));
	float vx = native_exp(native_divide(-1*j2*j2,2*sy*sy));
	
	gaussian[index] = vx*vy;
    }