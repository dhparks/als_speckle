__kernel void execute(
    __global float2* in,
    __global float2* out)
{   

    // define some registers
    int j      = get_global_id(0);
    int angles = 512;
    float2 current = 0;
    float temp = 0;
    int offset = j*angles
    int idx    = 0;
    
    float sum = 0;
    for (int k = 0; k < angles; k++) {
	
	
	
	
    }

    float m = 0;
    int aj = j*angles;
    
    for (int k = 0; k < angles; k++) {
	current = in[k+aj];
	temp += native_sqrt(current.x*current.x+current.y*current.y);
    out[j] = temp/angles;
    }
	
}