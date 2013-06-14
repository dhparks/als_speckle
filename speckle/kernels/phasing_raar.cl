__kernel void execute(
    float b,
    __global float* support,
    __global float2* psi_in,
    __global float2* psi_out,
    __global float2* output)

    {
	
	float b2 = 1-2*b;
	
	int i = get_global_id(0);
	
	// pull components into registers
	float s   = support[i];
	float2 po = psi_out[i];
	float2 pi = psi_in[i];
	
	float re  = (1-s)*(b*pi.x+b2*po.x)+s*po.x;
	float im  = (1-s)*(b*pi.y+b2*po.y)+s*po.y;
	
	// save
	output[i] = (float2)(re,im);
    }