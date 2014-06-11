__kernel void execute(
	__global float2* p_top,
	__global float*  p_btm,
	__global float2* object,
	__global float2* psi,
	int r0, int c0, int C)

    {

	int i = get_global_id(0);
	int j = get_global_id(1);
	int J = get_global_size(1);
	
	int idx_psi = i*J+j;
	int idx_obj = (i+r0)*C+j+c0;

	float2 o  = object[idx_obj];
	float2 a  = psi[idx_psi];
	float2 p  = p_top[idx_psi];

	p_top[idx_psi] = (float2)(p.x+o.x*a.x+o.y*a.y,p.x+o.x*a.y-o.y*a.x);
	p_btm[idx_psi] = o.x*o.x+o.y*o.y;
    }
