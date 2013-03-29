__kernel void execute(
    __global float* src_re, __global float* src_im, __global float* row_ave,
    __global float* out_re, __global float* out_im,
    __global int* pull, int fftrow, int frame) // the row and frame index

{	

        # define cm(a,b) (float2)(a.x*b.x-a.y*b.y,a.x*b.y+a.y*b.x)

	// i and j are the x and y coordinates of the pixel in out
	int row = get_global_id(0);
	int col = get_global_id(1);
	int N   = get_global_size(0); //this should be the length of the vector ie 512
        
        float d = row_ave[fftrow];
        float d3 = d*d*d;
	
	// make indices
        int i0 = fftrow*N;
        
	int i1 = i0+row;    // from src (wikipedia f1)
	int i2 = i0+col;    // from src (wikipedia f2)
	int i3 = row*N+col; // from pull
	int i4 = i0+pull[i3];  // pull this from src (f1+f2)
	int i5 = frame*N*N+i3; // this is the output index
	
	// pull the data

	float a1 = src_re[i1];
	float b1 = src_im[i1];
	float a2 = src_re[i2];
	float b2 = src_im[i2];
	float a3 = src_re[i4];
	float b3 = -src_im[i4];
        
	// put the output value in out
        float2 product = cm(cm((float2)(a1,b1), (float2)(a2,b2)),(float2)(a3,b3)); 
        out_re[i5] = product.x/d3;
        out_im[i5] = product.y/d3;
}