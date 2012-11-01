__kernel void execute(
    __global float*  ua,  // array holding x coords in source plane. should include pixel pitch in meters.
    __global float*  va,  // array holding y coords in source plane. should include pixel pitch in meters.
    __global float2* out, // holds E(u,v)*exp(ikr)*(1/r^2). this will be summed to do the integration.
    const    float k,    // the wavevector magnitude 2pi/lambda
    const    float p,    // the pixel pitch
    const    float z2)   // the propagation distance in meters, squared!.
    
    // i, j are x, y in the displayed geometry
    // ua and va should run from 0 to N, not N/2 to -N/2. this is to stay consistent with i,j

{   
	// pull data from arrays
	int i = get_global_id(0);
	int j = get_global_id(1);
	int N = get_global_size(1);
	int index = j+i*N;
	float u  = ua[index];
	float v  = va[index];
	
	// compute integrand quantities
	float r2  = pown(i*p-u,2)+pown(j*p-v,2)+z2;
	float ir2 = native_recip(r2);
	float r   = sqrt(r2);

	// compute integrand
	out[index] = 
	
	
	
}