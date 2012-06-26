__kernel void execute(
    __global float* input,
    __global float* walls,
    __global float* poswalls,
    __global float* negwalls,
    int N) // size

{	
	#define compare(a,b) (1-sign(a)*sign(b))/2
	
	// i and j are the x and y coordinates of the candidate pixel
	int i = get_global_id(0);
	int j = get_global_id(1);
	
	// grab the 8 nearest neighbors of point (i,j). use modulo arithmetic to enforce cyclic boundary conditions.
	// loops have been manually unrolled for speed increase
	int nx1 = i-1;
	int nx2 = i;
	int nx3 = i+1;
	int ny1 = j-1;
	int ny2 = j;
	int ny3 = j+1;
	
	if (nx1 < 0 || nx1 >= N) {nx1 = (nx1+N)%N;}
	if (nx2 < 0 || nx2 >= N) {nx2 = (nx2+N)%N;}
	if (nx3 < 0 || nx3 >= N) {nx3 = (nx3+N)%N;}
	
	if (ny1 < 0 || ny1 >= N) {ny1 = (ny1+N)%N;}
	if (ny2 < 0 || ny2 >= N) {ny2 = (ny2+N)%N;}
	if (ny3 < 0 || ny3 >= N) {ny3 = (ny3+N)%N;}
	
	float val11 = input[nx1+ny1*N]; 
	float val12 = input[nx1+ny2*N];
	float val13 = input[nx1+ny3*N];
	float val21 = input[nx2+ny1*N];
	float val22 = input[nx2+ny2*N]; // this is the candidate value; the others are the neighbors
	float val23 = input[nx2+ny3*N];
	float val31 = input[nx3+ny1*N];
	float val32 = input[nx3+ny2*N];
	float val33 = input[nx3+ny3*N]; 
	
	// the function "compare" returns 0 if both inputs are the same sign and 1 if the signs differ. this is how a domain wall is detected.
	float preout = compare(val11,val22)+compare(val12,val22)+compare(val13,val22)+
		       compare(val21,val22)                     +compare(val23,val22)+ //skip comparing val22 to itself!
		       compare(val31,val22)+compare(val32,val22)+compare(val33,val22);
				 
	// write to the output buffers
	float out = 0.0f;
	float posout = 0.0f;
	float negout = 0.0f;
	
	if (preout > 0.1f) {out = 1.0f;}
	if (preout > 0.1f && val22 > 0.0f)  {posout = 1.0f;}
	if (preout > 0.1f && val22 <= 0.0f) {negout = 1.0f;}
	
	walls[i+N*j]    = out;
	poswalls[i+N*j] = posout;
	negwalls[i+N*j] = negout;
}