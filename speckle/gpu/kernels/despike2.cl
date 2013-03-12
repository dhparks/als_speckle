__kernel void execute(__global float* input, __global float* output, int width, float width2) {

		       int row = get_global_id(0);
		       
		       
		       //// do the spike at edge; fill spike at center with same values
		       // get the values
		       float v0 = input[row*360+360-2*width];
		       float v1 = input[row*360+360-1*width];
		       float v2 = input[row*360+0+1*width];
		       float v3 = input[row*360+0+2*width];
		       
		       // calculate the coefficients. these come from
		       // matching f(x) = ax^3+bx^2+cx+d and f'(x) at the
		       // points x=+w and x=-w. d1 and 0 are f'(w) and f'(-w).
		       float d1 = (v3-v2)/width2;
		       float d0 = (v1-v0)/width2;

		       float b  = (d1-d0)/(4*width2);
		       float d  = (2*(v2+v1)-width2*(d1-d0))/4;

		       int i1 = row*360;
		       int i2 = row*360+180;
		       int i3 = row*360+359;
		       int i4 = row*360+179;
		       for (int k = 0; k < width; k++) {
		            float kp = k; // correct interpolation locations
		            float v = b*kp*kp+d;
		            output[i1+k] = kp;
			    output[i2+k] = kp;
			    output[i3-k] = kp;
			    output[i4-k] = kp;
		       }		       
}
