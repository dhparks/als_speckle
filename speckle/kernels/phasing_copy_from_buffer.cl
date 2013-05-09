__kernel void execute(__global float2 *dst, // destination (rows,cols)
                      __global float2 *src, // source (N,N)
                      int c0, int r0,       // col and row of lower corner ie c0 r0
                      int n, int N)         // frame number, self.N
    {
	
	// copy frame n out of scr and put it in dst. dst has size (rows,cols)
	// source has size (NxN) -- this should be changed in the future!
	
        int i_dst = get_global_id(0); // x coordinate
        int j_dst = get_global_id(1); // y coordinate
		
	int cols = get_global_size(0);
	int rows = get_global_size(1);
		
	int i_src = i_dst+c0;
	int j_src = j_dst+r0;
		
	int offset = n*rows*cols; // frame offset
	int k_dst  = i_dst+j_dst*cols;
	int k_src  = i_src+N*j_src+offset;
		
	dst[k_dst] = src[k_src];
    }