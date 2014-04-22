__kernel void execute(
    __global float2 *input,
    __global uchar *walls,
    __global uchar *poswalls,
    __global uchar *negwalls,
    __local  float *localmem,
    __global float *scratch
    )

{	
	#define compare(a,b) (1-sign(a)*sign(b))/2
	#define localindex(i,j,M) (1+i+(j+1)*(M+2))
	
	// it turns out that using local memory for this kernel does not
	// make things faster. presumably, this is because the
	// memory is already aligned? in the matrix transpose example,
	// the use of local memory prevents bank conflicts.
	// in this example, the compiler might already be able to send
	// an entire bank of results...
	
	// define the global and local indices for each worker
	int gi = get_global_id(0);
	int gj = get_global_id(1);
	int N  = get_global_size(0);
	int io = gi+N*gj;
	
	int li = get_local_id(0);
	int lj = get_local_id(1);
	int M  = get_local_size(0);
	
	// this is the index of each worker in the local memory
	int il = localindex(li,lj,M);

	// each worker needs to copy some data from global into local
	// workers on the edge have to copy more data than workers in the
	// center of the workgroup. for reasons that are unclear,
	// doing sign(input[io].x) causes a crash, but storing it first
	// in localmem then taking sign() seems ok.
	localmem[il] = input[io].x;
	localmem[il] = sign(localmem[il]);
	
	int gi_pad;
	int gj_pad;
	int right_side = (M+2)*(lj+1);
	int left_side  = (M+2)*(lj+1)+(M+1);
	int bottom_row = li+1;
	int top_row    = li+1+(M+2)*(M+1);
	
	if (li == 0) {
	  gi_pad = gi-1;
	  if (gi == 0) {gi_pad = N-1;}
	  localmem[left_side] = input[gi_pad+N*gj].x;
	  localmem[left_side] = sign(localmem[left_side]);
	}

	if (li == M-1) {
	  gi_pad = gi+1;
	  if (gi == N-1) {gi_pad = 0;}
	  localmem[right_side] = input[gi_pad+N*gj].x;
	  localmem[right_side] = sign(localmem[right_side]);
	}
	
	if (lj == 0) {
	  // copy pixels from below because we are on the bottom edge
	  gj_pad = gj-1;
	  if (gj == 0) {gj_pad = N-1;}
	  localmem[bottom_row] = input[gi+N*gj_pad].x;
	  localmem[bottom_row] = sign(localmem[bottom_row]);
	}
	
	if (lj == M-1) {
	  // copy pixels from above because we are on the top edge
	  gj_pad = gj+1;
	  if (gj == N-1) {gj_pad = 0;}
	  localmem[top_row] = input[gi+N*gj_pad].x;
	  localmem[top_row] = sign(localmem[top_row]);
	}
	
	// for the corner cases we have already checked the coordinates
	float corner = input[gi_pad+N*gj_pad].x;
	corner = sign(corner);
	int a = -1;
	if (li == 0   && lj == 0)   {a = 0;}
	if (li == M-1 && lj == 0)   {a = M+1;}
	if (li == 0   && lj == M-1) {a = (M+2)*(M+1);}
	if (li == M-1 && lj == M-1) {a = (M+2)*(M+2)-1;}
	if (a > -1) {localmem[a] = corner;}
	
	// signal that this worker is done copying to localmem
	barrier(CLK_LOCAL_MEM_FENCE);
	
	// The overprovisioned data for the workgroup has been copied
	// into local memory. For each pixel, check neighbors
	float v0   = localmem[(M+2)*(lj+1)+li+1];
	float comp = -1;
	for (int lx = 0; lx < 3; lx ++) {
		for (int ly = 0; ly < 3; ly++) {
			comp += (1-v0*localmem[(M+2)*(lj+ly)+li+lx])/2;
		}
	}

	// values to be written to output arrays
	uchar out    = 0;
	uchar posout = 0;
	uchar negout = 0;
	
	if (comp > 0.1f) {
		out = 1;
		if (v0  > 0.) {posout = 1;}
		if (v0 <= 0.) {negout = 1;}
	}
		
	// write to output arrays
	walls[io]    = out;
	poswalls[io] = posout;
	negwalls[io] = negout;
	
}
//}