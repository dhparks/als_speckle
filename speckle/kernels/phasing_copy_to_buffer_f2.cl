__kernel void execute(__global float2 *src, // source      (N,N)
                      __global float2 *dst, // destination (nf, nr, nc)
		      int r0,              // starting row
		      int c0,              // starting column
                      int N,               // frame in destination to which to copy source
		      int n)               // reconstruction is NxN
    {
    
       int i = get_global_id(0); // row
       int j = get_global_id(1); // col
       
       int rows = get_global_size(0);
       int cols = get_global_size(1);
       
       int idx_src = (i+r0)*N+(j+c0);
       int idx_dst = rows*cols*n+i*cols+j;
       
       dst[idx_dst] = src[idx_src];
    };