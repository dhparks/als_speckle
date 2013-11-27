__kernel void execute(
    __global float2* in,     
    __global float* denoms, 
    __global float2* out,    
    int mode)

{   

    int angles = 512;
    int i = get_global_id(0);
    float d = denoms[i];
    float v = native_recip(d*d);
    
    int row_start = i*angles;
    for (int j = 0; j < angles; j++) {
	
	int index  = row_start + j;
	float2 a   = in[index];
	float mag  = native_sqrt(a.x*a.x+a.y*a.y);
	out[index] = (float2)(mag*v,0.f);
	
	
    }

    //int index  = i+j*angles;
    //float d = denoms[j];
    //float v = native_recip(d*d);
    //float mag = 0;
    
    //if (mode == 0){ //wochner
	//float2 a = in[index];
	//mag = native_sqrt(a.x*a.x+a.y*a.y);
	//out[index] = (float2)(mag*v-1,0);
    //}
	
    //if (mode == 1){ // straight division to make ac value 1
	//out[index] = (float2)(in[index].x*v,0);
    //}
	
}