__kernel void execute(
	__global float2* object,  // the real space object, arbitrary shape
	__global float2* probe,   // the probe function, square 
	__global float2* psi,     // the 3d array of current psi values
	__global float2* out1,    // the 3d array of psi values to be fft-ed
	__global float2* out2,    // the difference psi-p*o
	__global int* r,          // list of row coordinates for each probe spot
	__global int* c,          // list of col coordinates for each probe spot
	int object_cols)          // number of columns in the object array
	
	// compute all the psi-in values for difference map
	// PsiIn = 2*Prb*Obj-Psi

{
	int i = get_global_id(0); // frame number in output
	int j = get_global_id(1); // row number in output
	int k = get_global_id(2); // col number in output
	
	int J = get_global_size(1); // number of rows in output
	int K = get_global_size(2); // number of columns in output
	
	int this_r = r[i];
	int this_c = c[i];
	
	// figure out the indices of the data for this work element
	int out_index = i*J*K+j*K+k;
	int prb_index = j*K+k;
	int obj_index = (j+this_r)*object_cols+this_c+k;
	
	// pull all the data into registers
	float2 this_prb = probe[prb_index];
	float2 this_psi = psi[out_index];
	float2 this_obj = object[obj_index];
	
	// equation: 2*Probe*Object-Psi
	float2 tmp  = (float2) (this_prb.x*this_obj.x-this_prb.y*this_obj.y,this_prb.y*this_obj.x+this_prb.x*this_obj.y);
	out1[out_index] = tmp+tmp-this_psi;
	out2[out_index] = this_psi-tmp;
}
