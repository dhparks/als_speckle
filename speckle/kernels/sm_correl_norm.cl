__kernel void execute(
    __global float2* in,     
    __global float* denoms, 
    __global float2* out,    
    int mode)

{   

    int angles = 512;
    int i = get_global_id(0);
    int j = get_global_id(1);

    int index  = i+j*angles;
    float d = denoms[j];
    float v = native_recip(d*d);
    float mag = 0;
    
    if (mode == 0){ //wochner
	float2 a = in[index];
	mag = hypot(a.x,a.y);
	out[index] = (float2)(mag*v-1,0);
    }
	
    if (mode == 1){ // straight division to make ac value 1
	out[index] = (float2)(in[index].x*v,0);
    }
	
}