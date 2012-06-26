__kernel void execute(
    __global float* input,
    __global float* output,
    int N)

    {	
	int i = 0;
	int j = 0;

	// grab the 8 nearest neighbors of point (i,j). use modulo arithmetic to enforce cyclic boundary conditions.
	int nx1 = i-1;
	int nx2 = i;
	int nx3 = i+1;
	int ny1 = j-1;
	int ny2 = j;
	int ny3 = j+1;
	
	if (nx1 < 0 || nx1 >= N) {nx1 = (nx1+N)%N;}
	if (nx3 < 0 || nx1 >= N) {nx3 = (nx3+N)%N;}
	if (ny1 < 0 || ny1 >= N) {ny1 = (ny1+N)%N;}
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
	
	int out = (val11+val12+val13+val21+val23+val31+val32+val33)/8;
	output[0] = out;
    }