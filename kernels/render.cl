struct ubyte3 {
	unsigned char x, y, z;
}


/* https://www.codespeedy.com/hsv-to-rgb-in-cpp/ */
ubyte3 HSVtoRGB(float H, float S,float V){
    if(H>360 || H<0 || S>100 || S<0 || V>100 || V<0){
        return ubyte3{0,0,0};
    }
    float s = S/100;
    float v = V/100;
    float C = s*v;
    float X = C*(1-abs(fmod(H/60.0, 2)-1));
    float m = v-C;
    float r, g, b;
    if(H >= 0 && H < 60){
        r = C,
        g = X,
        b = 0;
    }
    else if(H >= 60 && H < 120){
        r = X,
        g = C,
        b = 0;
    }
    else if(H >= 120 && H < 180){
        r = 0,
        g = C,
        b = X;
    }
    else if(H >= 180 && H < 240){
        r = 0,
        g = X,
        b = C;
    }
    else if(H >= 240 && H < 300){
        r = X,
        g = 0,
        b = C;
    }
    else{
        r = C,
        g = 0,
        b = X;
    }

    int R = (r+m)*255;
    int G = (g+m)*255;
    int B = (b+m)*255;

    return ubyte3{R,G,B};
}

__kernel void render(__global ubyte3 *pixels,
                     const __global float2 *data) {
    int id = get_global_id(0);
    float2 &z = data[id];
    float norm = sqrt(z.x * z.x + z.y * z.y);
    float angle = atan2(z.y, z.x);
    pixels[id] = HSVtoRGB(angle * 57.29578f, 100.0f, max(norm, 1.0f) * 100.0f);
}