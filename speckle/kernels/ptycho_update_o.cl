__kernel void execute(
	__global float2* o_top,
	__global float*  o_btm,
	__global float2* probe,
	__global float2* wave,
	int r0, int c0, int C)

    {

	int i = get_global_id(0);
	int j = get_global_id(1);
	int J = get_global_size(1);
	
	int idx_psi = i*J+j;
	int idx_obj = (i+r0)*C+j+c0;

	float2 p = probe[idx_psi];
	float2 w = wave[idx_psi];
	float2 o = o_top[idx_obj];

	o_top[idx_obj] = (float2)(o.x+p.x*w.x+p.y*w.y,o.y+p.x*w.y-p.y*w.x);

	float o2 = o_btm[idx_obj]; 
	o_btm[idx_obj] = o2+p.x*p.x+p.y*p.y;
    }
