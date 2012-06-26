__kernel void execute(
    __global float* cor_vals, // correlation values
    __global float* cos_vals, // cosine values
    __global float* out_vals, // output buffer
    int cos_rows)             // number of rows in the cosine array

{   
	int i = get_global_id(0);
	int j = get_global_id(1); 
	
	for (int k = 0; k < 360; k++) {
		out_vals[i+cos_rows*j] += cor_vals[k+360*j]*cos_vals[k+360*i]/360;
	}
	
}