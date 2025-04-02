#define TOLERANCE 1e-9

inline float2 mul2(float2 a, float2 b) {
    return (float2)(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}


inline float2 func(float2 z) {
    return ( mul2( mul2(z, z), z) - 1.0f);
}

inline float2 deriv(float2 z) {
    return (3.0f * mul2(z, z));
}


inline float abs2(float2 z) {
    return z.x * z.x + z.y * z.y;
}

inline float2 div2(float2 a, float2 b) {
    float denominator = b.x * b.x + b.y * b.y;
    float2 result;
    result.x = (a.x * b.x + a.y * b.y) / denominator;
    result.y = (a.y * b.x - a.x * b.y) / denominator;
    return result;
}


__kernel void step(__global double *data, 
                   const int width,
                   __global double *roots,
                   const int num_roots) {

    int y = get_global_id(0);
    int x = get_global_id(1);
    int id = width * y + x;

    float2 z;
    z.x = data[id * 2 + 0];
    z.y = data[id * 2 + 1];

    bool skip = false;

    for ( int i = 0; i < num_roots; i++ ) {
        float2 root;
        root.x = roots[i * 2 + 0];
        root.y = roots[i * 2 + 1];
        if (abs2(z - root) < TOLERANCE) { skip = true; }
    }

    float2 deriv_ = deriv(z);

    if (abs2(deriv_) < TOLERANCE) { skip = true; }
    if (skip) { return; }

    float2 f_z = func(z);

    float2 z_new;

    z_new = z - div2(f_z, deriv_);

    data[id * 2 + 0] = (double) z_new.x;
    data[id * 2 + 1] = (double) z_new.y;
}
