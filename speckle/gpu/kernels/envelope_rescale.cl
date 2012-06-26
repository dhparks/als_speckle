__kernel void execute(
    __global float* dft_re,
    __global float* dft_im,
    __global float* goal,
    __global float* blurred,
    __global float* rescaler) 					  
    {	
	int i = get_global_id(0);
	float rescale = native_sqrt(goal[i]/blurred[i]); // sqrt because goal and blurred are intensity but dft_re and dft_im are field
	if (isnan(rescale)) {rescale = 0.0f;}

	dft_re[i] = dft_re[i]*rescale;
	dft_im[i] = dft_im[i]*rescale;
	rescaler[i] = rescale;
    }