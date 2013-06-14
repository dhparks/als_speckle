__kernel void execute(
    float g,
    __global float* support,
    __global float2* psi_in,
    __global float2* output)

    {

	int i = get_global_id(0);
	
	// pull components into registers
	float s   = support[i];
	float2 pi = psi_in[i];
	
	// save
	output[i] = -1*g*(1-s)*pi+s*pi;
    }