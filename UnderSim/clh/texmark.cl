__constant sampler_t directSampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

inline float sample(__read_only image2d_t image, int2 pos)
{
    float4 v = read_imagef(image, directSampler, pos);
    return v.x + v.y + v.z + v.w;
}

__kernel void benchmark2d(__read_only image2d_t image, __global uint *out)
{
    int2 gid = (int2) (get_global_id(0), get_global_id(1)) ADDR_MODIFIER;
    float sum = sample(image, gid);
    sum += sample(image, gid + (int2) (5, 5));
    sum += sample(image, gid + (int2) (10, 10));
    sum += sample(image, gid + (int2) (-5, 5));
    sum += sample(image, gid + (int2) (5, -5));
    sum += sample(image, gid + (int2) (0, 20));
    sum += sample(image, gid + (int2) (0, -20));
    sum += sample(image, gid + (int2) (-20, 0));
    sum += sample(image, gid + (int2) (20, 0));

    volatile float sink = sum;
}
