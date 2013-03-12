__kernel void execute(
    __global float* in,
    __global float* out,
    int angles,
    int mode) 
{   

	int j = get_global_id(0);
	float current = 0.0f;
	
	if (mode == 0){ // this is the mode in the wochner paper
	    for (int k = 0; k < angles; k++) {
		current = in[k+j*angles];
		out[j] += current/angles;
		//out[j] += current*current/rows;
	    }
	    }
	    
	if (mode == 1){ // divide by the delta=0 value
	    out[j] = in[j*angles];
	    }
	
}