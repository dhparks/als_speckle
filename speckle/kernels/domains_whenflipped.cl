__kernel void execute(__global int* whenflipped, 
					  __global float* new_domains,
					  __global float* old_domains,
					  int iteration) 					  
{	

	int i = get_global_id(0);
	float n = as_float(iteration);
	
	float new = new_domains[i];
	float old = old_domains[i];
	if (sign(new) != sign(old)) {whenflipped[i] = iteration;}
	
}





