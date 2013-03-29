__kernel void execute(__global int* whenflipped, 
					  __global float* new_domains,
					  __global float* old_domains,
					  __global float* recency_val,
					  float target, int iteration) 					  
{	

	int i = get_global_id(0);
	float st = sign(target);
	float use = 0.0f;
	
	if (whenflipped[i] > 0.1f) {
		int age = abs(whenflipped[i]-iteration);
		float want = new_domains[i]-old_domains[i];
		float need = recency_val[iteration-whenflipped[i]];
		
		float sod = sign(old_domains[i]);
		float sw  = sign(want);
		
		if (isequal(sod,st)) {
			if (isequal(sod,sw)) {
				if (isgreater(fabs(want), need)) {
					use = 1.0f;
				}
			}	
		}					
					
		new_domains[i] = new_domains[i]*use+old_domains[i]*(1-use);
	}
	
}





