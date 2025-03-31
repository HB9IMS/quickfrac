#define TOLERANCE 1e-9

#define testing

#ifdef testing
#define func(z) (z * z * z - 1.0f)
#define deriv(z) (3.0f * z * z)
#endif

inline float2 func(float2 z) {
    return $func$; /* to be replaced at reading */
}

inline float2 deriv(float2 z) {
    return $deriv$; /* to be replaced at reading */
}


inline float abs2(float2 z) {
    return z.x * z.x + z.y * z.y;
}


__kernel void step(__global float2 *pixels,
	               const __global float2 *roots) {
    int id = get_global_id(0);
    float2 &z = pixels[id];
    bool skip = false;
    for (float2 root : roots) {  /* first time using this */
        if (abs2(z - root) < TOLERANCE) { skip = true; }
    }
    if (abs2(deriv_) < TOLERANCE) { skip = true; }
    if (skip) { return; }
    float2 f_z = func(z);
    float2 deriv_ = deriv(z);
    z = z - f_z / deriv_;
    return;
}