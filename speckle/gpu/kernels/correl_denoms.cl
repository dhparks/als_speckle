__kernel void execute(
    __global float* in,
    __global float* out,
    int rows) 
{   
	
	int angles = 512;
	
	int j = get_global_id(0);
	float current = 0.0f;
	
	for (int k = 0; k < angles; k++) {
		current = in[k+j*angles];
		out[j] += current*current/rows;
	}
	
}