__kernel void execute(
    float b,
    __global float* support,
    __global float2* psi_in,
    __global float2* psi_out,
    __global float2* output)

    {	
	int i = get_global_id(0);
	
	// pull components into registers
	float s   = support[i];
	float2 po = psi_out[i];
	float2 pi = psi_in[i];
	
	float re  = (1-s)*(pi.x-b*po.x)+s*po.x;
	float im  = (1-s)*(pi.y-b*po.y)+s*po.y;
	
	// save
	output[i] = (float2)(re,im);
    }