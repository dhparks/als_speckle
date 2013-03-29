__kernel void execute(__global float2* input, 
					  __global float2* output, 
					  int width, float width2) {

		       int row = get_global_id(0);
		       
		       int angles = 512;
		       int angles2 = angles/2;
		       
		       //// do the spike at center
		       // get the values
		       float v0 = input[row*angles+angles2-2*width].x;
		       float v1 = input[row*angles+angles2-1*width].x;
		       float v2 = input[row*angles+angles2+1*width].x;
		       float v3 = input[row*angles+angles2+2*width].x;
		       
		       // calculate the coefficients. these come from
		       // matching f(x) = ax^3+bx^2+cx+d and f'(x) at the
		       // points x=+w and x=-w. d1 and 0 are f'(w) and f'(-w).
		       float d1 = (v3-v2)/width2;
		       float d0 = (v1-v0)/width2;

		       float b  = (d1-d0)/(4*width2);
		       float d  = (2*(v2+v1)-width2*(d1-d0))/4;

		       int istart = row*angles+angles2-width;
		       for (int k = 0; k < 2*width; k++) {
		            output[istart+k] = (float2)(b*(k-width2)*(k-width2)+d,0);
		       }
		       
		       //// do the spike at edge
		       // get the values
		       v0 = input[row*angles+angles-2*width].x;
		       v1 = input[row*angles+angles-1*width].x;
		       v2 = input[row*angles       +1*width].x;
		       v3 = input[row*angles       +2*width].x;
		       
		       // calculate the coefficients. these come from
		       // matching f(x) = ax^3+bx^2+cx+d and f'(x) at the
		       // points x=+w and x=-w. d1 and 0 are f'(w) and f'(-w).
		       d1 = (v3-v2)/width2;
		       d0 = (v1-v0)/width2;

		       b  = (d1-d0)/(4*width2);
		       d  = (2*(v2+v1)-width2*(d1-d0))/4;

		       istart = row*angles;
		       for (int k2 = 0; k2 < width; k2++) {
		            //output[istart+k2] = b*k2*k2+d;
			    output[istart+k2] = (float2)(b*k2*k2+d,0);
		       }
		       
		       istart = row*angles+angles-width;
		       for (int k3 = 0; k3 < width; k3++) {
		            output[istart+k3] = (float2)(b*(k3-width2)*(k3-width2)+d,0);
		       }
		       
		       
		       
}
