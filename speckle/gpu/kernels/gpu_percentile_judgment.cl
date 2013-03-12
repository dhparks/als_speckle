__kernel void execute(
		    __global float *candidates, 
		    __global float *basis,
		    __global int   *out,
		    int   b_rows, 
		    int   b_cols)
    {
	// search a sorted list for where to insert a value
	// using a binary search. very fast.
        int i = get_global_id(0);
        int j = get_global_id(1);
	int rows  = get_global_size(0);
	int cols  = get_global_size(1);
		
        float val_target = candidates[i*cols+j];
		
	int t = 1;            // threshold that determines the success condition
	int i_upper = b_cols; // initial upper bound in binary search
	int i_lower = 0;      // initial lower bound in binary search
	int i_guess = 0;      // initial guess index
	int success = 0;      // bool value to stop while loop
	float v_guess = 0.0f; // value of initial guess
		
	while (success == 0)
	    {
		i_guess = (i_upper+i_lower)/2;
		v_guess = basis[i*b_cols+i_guess];
		if (val_target < v_guess) { i_upper = i_guess; }
		if (val_target > v_guess) { i_lower = i_guess; }
		if (val_target == v_guess) {success = 1;} // floating point, so never actually happens, but included for completeness
		if (i_upper-i_lower <= t)  {success = 1;}
	    }
        out[i*cols+j] = (i_upper+i_lower)/2;
    }