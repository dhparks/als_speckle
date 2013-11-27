__kernel void execute(
    __global float2* in,
    __local float* out,
    int mode) 
{   

    int j = get_global_id(0);
    float temp = 0;
    int angles = 512;
    float m = 0;
    int aj = j*angles;
    float2 current = 0;
    
    if (mode == 0){ // this is the mode in the wochner paper
	for (int k = 0; k < angles; k++) {
	    current = in[k+aj];
	    temp += native_sqrt(current.x*current.x+current.y*current.y);
	out[j] = temp/angles;
	}
    } // end if
	
    if (mode == 1){ // divide by the delta=0 value
	out[j] = in[j*angles].x;
    }
	
}