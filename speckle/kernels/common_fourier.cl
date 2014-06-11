__kernel void execute(
    __global float2* psi,
    __global float2* div,
    __global float* mod,
    __global float2* out)

    {	
	int i = get_global_id(0);
	
	// pull components into registers. div is nominally float2, but
	// any imaginary component is numerical leak from ffts so discard it
	float2 p = psi[i];
	float  s = mod[i]/div[i].x;
	
	out[i] = (float2)(p.x*s,p.y*s);
    }
