__kernel void execute(
    __global float beta,
    __global float* support,
    __global float2* psi_in,
    __global float2* psi_out,
    __global float2* output)

    {	
	int i = get_global_id(0);
	
	// pull components into registers
	float s   = support[i];
	float2 po = psi_out[i];
	float2 pi = psi_in[i]
	
	// pull the components into registers
	out[i] = 

        out[i] = (float2)((1-support[i])*(psi_in[i].x-beta*psi_out[i].x)+support[i]*psi_out[i].x,(1-support[i])*(psi_in[i].y-beta*psi_out[i].y)+support[i]*psi_out[i].y)",
        hio")
	
    }