__kernel void execute(
    __global float* input,
    __global float* maxima)

{	
	#define compare(a,b) (1-sign(a)*sign(b))/2
	
	// i and j are the x and y coordinates of the candidate pixel
	int i = get_global_id(0);
	int j = get_global_id(1);
	int I = get_global_size(0);
	int J = get_global_size(1);
	
	// grab the 8 nearest neighbors of point (i,j). use modulo arithmetic to enforce cyclic boundary conditions.
	// loops have been manually unrolled for speed increase
	int nx1 = i-1;
	int nx2 = i;
	int nx3 = i+1;
	int ny1 = j-1;
	int ny2 = j;
	int ny3 = j+1;
	
	if (nx1 < 0 || nx1 >= I) {nx1 = (nx1+I)%I;}
	if (nx2 < 0 || nx2 >= I) {nx2 = (nx2+I)%I;}
	if (nx3 < 0 || nx3 >= I) {nx3 = (nx3+I)%I;}
	
	if (ny1 < 0 || ny1 >= J) {ny1 = (ny1+J)%J;}
	if (ny2 < 0 || ny2 >= J) {ny2 = (ny2+J)%J;}
	if (ny3 < 0 || ny3 >= J) {ny3 = (ny3+J)%J;}
	
	float val11 = input[nx1+ny1*I]; 
	float val12 = input[nx1+ny2*I];
	float val13 = input[nx1+ny3*I];
	float val21 = input[nx2+ny1*I];
	float val22 = input[nx2+ny2*I]; // this is the candidate value; the others are the neighbors
	float val23 = input[nx2+ny3*I];
	float val31 = input[nx3+ny1*I];
	float val32 = input[nx3+ny2*I];
	float val33 = input[nx3+ny3*I];
	
	int greaters = 0;
	if (val22 > val11) {greaters++;}
	if (val22 > val12) {greaters++;}
	if (val22 > val13) {greaters++;}
	if (val22 > val21) {greaters++;}
	if (val22 > val23) {greaters++;}
	if (val22 > val31) {greaters++;}
	if (val22 > val32) {greaters++;}
	if (val22 > val33) {greaters++;}
	
	if (greaters == 8) {maxima[i+j*I] = 1.0f;}
	if (greaters < 8) {maxima[i+j*I] = 0.0f;}
}