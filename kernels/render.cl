#define PIXTYPE uchar3


/* https://www.codespeedy.com/hsv-to-rgb-in-cpp/ */
PIXTYPE HSVtoRGB_old(float H, float S,float V){
    if(H>360 || H<0 || S>100 || S<0 || V>100 || V<0){
        PIXTYPE res;
        res.x = 0x70;
        res.y = 0x70;
        res.z = 0x70; 
        return res;
    }
    float s = S/100;
    float v = V/100;
    float C = s*v;
    float X = C*(1 - fabs((float) fmod(H/60.0, 2)-1));
    float m = v-C;
    float r, g, b;
    if(H >= 0 && H < 60){
        r = C;
        g = X;
        b = 0;
    }
    else if(H >= 60 && H < 120){
        r = X;
        g = C;
        b = 0;
    }
    else if(H >= 120 && H < 180){
        r = 0;
        g = C;
        b = X;
    }
    else if(H >= 180 && H < 240){
        r = 0;
        g = X;
        b = C;
    }
    else if(H >= 240 && H < 300){
        r = X;
        g = 0;
        b = C;
    }
    else{
        r = C;
        g = 0;
        b = X;
    }

    PIXTYPE res;
    res.x = (r+m)*255;
    res.y = (g+m)*255;
    res.z = (b+m)*255; 

    return res;
}


PIXTYPE Polar_to_RGB( float ang, float mag ) {
    float intensity = clamp(mag, 0.f, 1.f);
    int hue = ( (int)( ang * 6.0f )) % 6;
    float rem = fmod( ang * 6.0f, 1.0f ); // slope up
    float r, g, b;
    float mre = 1.0f - rem; // slope down
    // curve: high down low low up high
    switch ( hue ) {
        case 0: r = 1.f; g = rem; b= 0.f; break;
        case 1: r = mre; g = 1.f; b= 0.f; break;
        case 2: r = 0.f; g = 1.f; b= rem; break;
        case 3: r = 0.f; g = mre; b= 1.f; break;
        case 4: r = rem; g = 0.f; b= 1.f; break;
        case 5: r = 1.f; g = 0.f; b= mre; break;
    }
    PIXTYPE res;
    res.x = ( unsigned char )( 255 * (r * intensity) );
    res.y = ( unsigned char )( 255 * (g * intensity) );
    res.z = ( unsigned char )( 255 * (b * intensity) );
    return res;
}

__kernel void render(__global uchar *pixels,
                     __global double *data,
                     const int width) {

    int y = get_global_id(0);
    int x = get_global_id(1);
    int id = width * y + x;

    float2 z;
    z.x = data[id * 2 + 0];
    z.y = data[id * 2 + 1];

    float norm = sqrt( z.x * z.x + z.y * z.y );
    float angle = atan2( (float) z.y, (float) z.x) / (2 * M_PI);
    while (angle < 0) {
        angle += 1;
    }
    while (angle > 1) {
        angle -= 1;
    }

    PIXTYPE res;
    // check if coords are good:
    // res.x = (uchar) (255.0f * x / 640.f);
    // res.y = (uchar) (255.0f * y / width);
    // res.z = 0;

    // res.x = (uchar) ((z.x / 4.f + 0.5f) * 255);
    // res.y = (uchar) ((z.y / 4.f + 0.5f) * 255);
    // res.z = (uchar) 0;

    // check if angle and norm are looking ok
    // res.x = (uchar) (angle * 255.f);
    // res.y = (uchar) (norm / 6.f * 255.f);

    // check if angle is in bounds (should be full white; red = too high, green = too low)
    // res.x = (uchar) (0 ? (0.f <= angle) : 255);
    // res.y = (uchar) (0 ? (angle <= 1.f) : 255);
    // res.z = (uchar) (0 ? (0.f <= angle && angle <= 1.f) : 255);

    res = Polar_to_RGB(angle, norm);

    pixels[3 * id + 0] = res.x;
    pixels[3 * id + 1] = res.y;
    pixels[3 * id + 2] = res.z;
}

__kernel void render_subpixel_aa(__global uchar *pixels,
                                 __global double *data,
                                 const int width) {

    int x_ = get_global_id(0);
    int y_ = get_global_id(1);
    int z_ = get_global_id(2);
    int id = width * x_ + 3 * y_ + z_;

    float2 z;
    z.x = data[id * 2 + 0];
    z.y = data[id * 2 + 1];

    float norm = sqrt( z.x * z.x + z.y * z.y );
    float angle = atan2( (float) z.y, (float) z.x) / (2 * M_PI);
    while (angle < 0) {
        angle += 1;
    }
    while (angle > 1) {
        angle -= 1;
    }

    PIXTYPE res;

    res = Polar_to_RGB(angle, norm);

    switch ( z_ ) {
        case 0: pixels[id] = res.x; break;
        case 1: pixels[id] = res.y; break;
        case 2: pixels[id] = res.z; break;
    }
}