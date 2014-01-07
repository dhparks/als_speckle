__kernel void execute(
	__global float2* cor_vals, // fft of correlation values
	__global float2* out_vals, // output buffer
	__local  float2* localmem) // local memory to coalesce global writes

	{   
	// get global indices. these refer to the output coordinates
	int gi = get_global_id(0);   
	int gj = get_global_id(1); 
	int N  = get_global_size(0);
	int io = gj*N+gi;
	
	// get local indices. these refer to the local memory buffer
	int li = get_local_id(0);
	int lj = get_local_id(1);
	int M  = get_local_size(0); // 16
	int il = lj*M+li;
	
	// based on the global output indices, calculate the global
	// indices for the input values. we only slice out the even columns.
	int c  = (gi+1)*2;
	int ic = gj*512+c;
	
	// copy the correlation value from cor_vals into localmem
	localmem[il] = cor_vals[ic]/512.;
	
	//int c = (i+1)*2; // this is the component, and also the column in cor_vals
	//int io = j*N+i; // this is the index in out_vals
	//int ic = j*512+c; // this is the index in cor_vals (angles = 512)
	
	//float2 cv = cor_vals[ic];
	//float2 m = cv;//native_sqrt(cv.x*cv.x+cv.y*cv.y);
	
	// wait for all the copies to finish, then write to the output
	barrier(CLK_LOCAL_MEM_FENCE);
	
	out_vals[io] = localmem[il];
	}