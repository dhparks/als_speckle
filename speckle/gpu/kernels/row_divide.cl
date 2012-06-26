__kernel void execute(
    __global float* corr,
    __global float* norm,
    __global float* out)
    
    // divide rows in an angular correlation by a normalization factor
{   
	
	int angles = 512;
	int i = get_global_id(0); // number of columns (512)
	int j = get_global_id(1); // number of rows
	float r = native_recip(norm[j]);
	out[i+j*angles] = corr[i+j*angles]*r;
}