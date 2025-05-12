#pragma OPENCL EXTENSION cl_khr_fp64 : enable

#define TOLERANCE 1e-29
#define CTYPE double2
#define DISABLE_SKIP true
#define cpow_fake cpow  // dylans idee

inline CTYPE mul2(CTYPE a, CTYPE b) {
    return (CTYPE)(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

inline double abs2(CTYPE z) {
    return sqrt(z.x * z.x + z.y * z.y);
}

/*
w/v
w*vb/(v*vb)
(wx+wyi)*(vx-vyi)/|v|^2
wxvx-wxvyi+vxwyi+wyvy / (vxvx + vyvy)
*/
inline CTYPE div2(CTYPE a, CTYPE b) {
    CTYPE result = (CTYPE)((a.x * b.x + a.y * b.y) / (b.x * b.x + b.y * b.y), 
                           (a.y * b.x - a.x * b.y) / (b.x * b.x + b.y * b.y));
    return result;
}

inline CTYPE cpow(CTYPE z, CTYPE w) { // z^w = exp(w * log(z))
    double r = abs2(z);
    double theta = atan2(z.y, z.x);
    CTYPE log_z = (CTYPE)(log(r), theta);
    
    CTYPE w_log_z = mul2(w, log_z);
    
    // Compute exp(w * log(z))
    double exp_real = exp(w_log_z.x);
    double sin_imag = sin(w_log_z.y);
    double cos_imag = cos(w_log_z.y);
    
    return (CTYPE)(exp_real * cos_imag, exp_real * sin_imag);
}

inline CTYPE cpow_real(CTYPE z, double n) {
    double r = pow(z.x * z.x + z.y * z.y, n/2.0);
    double theta = n * atan2(z.y, z.x);
    return (CTYPE)(r * cos(theta), r * sin(theta));
}

inline CTYPE func(CTYPE z) {
    return (CTYPE)($f$);
}

inline CTYPE deriv(CTYPE z) {
    return (CTYPE)($d$);
}


__kernel void step(__global double *data, 
                   const int width,
                   __global double *roots,
                   const int num_roots) {

    int y = get_global_id(0);
    int x = get_global_id(1);
    int id = width * y + x;

    CTYPE z;
    z.x = data[id * 2 + 0];
    z.y = data[id * 2 + 1];

    bool skip = false;

    if (!DISABLE_SKIP) {
        for ( int i = 0; i < num_roots; i++ ) {
            CTYPE root;
            root.x = roots[i * 2 + 0];
            root.y = roots[i * 2 + 1];
            if (abs2(z - root) <= TOLERANCE) { skip = true; }
        }
    }

    CTYPE deriv_ = deriv(z);

    CTYPE f_z = func(z);

    if (abs2(deriv_) <= TOLERANCE) { skip = true; }
    if (skip && !DISABLE_SKIP) {  // consistency with cpu version  well that aged like milk...
        deriv_.x = 1.0;
        deriv_.y = 0.0;
        f_z.x = 0.0;
        f_z.y = 0.0;
    }

    CTYPE z_new;

    z_new = z - div2(f_z, deriv_);

    data[id * 2 + 0] = (double) z_new.x;
    data[id * 2 + 1] = (double) z_new.y;
/*
    fvalues[id * 2 + 0] = (double) f_z.x;
    fvalues[id * 2 + 1] = (double) f_z.y;

    derivs[id * 2 + 0] = (double) deriv_.x;
    derivs[id * 2 + 1] = (double) deriv_.y;*/
}

__kernel void step_n(__global double *data,
                     const int width,
                     __global double *roots,
                     const int num_roots,
                     const int repetitions) {

    int y = get_global_id(0);
    int x = get_global_id(1);
    int id = width * y + x;


    CTYPE z_new;
    CTYPE deriv_;
    CTYPE f_z;
    CTYPE root;

    bool skip;

    CTYPE z;
    z.x = data[id * 2 + 0];
    z.y = data[id * 2 + 1];

    for (int n_repetition = 0; n_repetition < repetitions; n_repetition++) {
        skip = false;

        if (!DISABLE_SKIP) {
            for ( int i = 0; i < num_roots; i++ ) {
                root.x = roots[i * 2 + 0];
                root.y = roots[i * 2 + 1];
                if (abs2(z - root) <= TOLERANCE) { skip = true; }
            }
        }

        deriv_ = deriv(z);

        f_z = func(z);

        if (abs2(deriv_) <= TOLERANCE) { skip = true; }
        if (skip && !DISABLE_SKIP) {  // consistency with cpu version  well that aged like milk...
            deriv_.x = 1.0;
            deriv_.y = 0.0;
            f_z.x = 0.0;
            f_z.y = 0.0;
        }

        z_new = z - div2(f_z, deriv_);

        data[id * 2 + 0] = (double) z_new.x;
        data[id * 2 + 1] = (double) z_new.y;
/*
        fvalues[id * 2 + 0] = (double) f_z.x;
        fvalues[id * 2 + 1] = (double) f_z.y;

        derivs[id * 2 + 0] = (double) deriv_.x;
        derivs[id * 2 + 1] = (double) deriv_.y;
*/
        z = z_new;
        if (skip) { break; }
    }
}
