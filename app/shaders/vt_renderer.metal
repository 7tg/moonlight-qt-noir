#include <metal_stdlib>
using namespace metal;

struct Vertex
{
    float4 position [[ position ]];
    float2 texCoords;
};

struct CscParams
{
    half3x3 matrix;
    half3 offsets;
    half2 chromaOffset;
    half bitnessScaleFactor;
};

constexpr sampler s(coord::normalized, address::clamp_to_edge, filter::linear);

vertex Vertex vs_draw(constant Vertex *vertices [[ buffer(0) ]], uint id [[ vertex_id ]])
{
    return vertices[id];
}

fragment half4 ps_draw_biplanar(Vertex v [[ stage_in ]],
                                constant CscParams &cscParams [[ buffer(0) ]],
                                texture2d<half> luminancePlane [[ texture(0) ]],
                                texture2d<half> chrominancePlane [[ texture(1) ]])
{
    float2 chromaOffset = float2(cscParams.chromaOffset) / float2(luminancePlane.get_width(),
                                                                  luminancePlane.get_height());
    half3 yuv = half3(luminancePlane.sample(s, v.texCoords).r,
                      chrominancePlane.sample(s, v.texCoords + chromaOffset).rg);
    yuv *= cscParams.bitnessScaleFactor;
    yuv -= cscParams.offsets;

    return half4(yuv * cscParams.matrix, 1.0h);
}

fragment half4 ps_draw_triplanar(Vertex v [[ stage_in ]],
                                 constant CscParams &cscParams [[ buffer(0) ]],
                                 texture2d<half> luminancePlane [[ texture(0) ]],
                                 texture2d<half> chrominancePlaneU [[ texture(1) ]],
                                 texture2d<half> chrominancePlaneV [[ texture(2) ]])
{
    float2 chromaOffset = float2(cscParams.chromaOffset) / float2(luminancePlane.get_width(),
                                                                  luminancePlane.get_height());
    half3 yuv = half3(luminancePlane.sample(s, v.texCoords).r,
                      chrominancePlaneU.sample(s, v.texCoords + chromaOffset).r,
                      chrominancePlaneV.sample(s, v.texCoords + chromaOffset).r);
    yuv *= cscParams.bitnessScaleFactor;
    yuv -= cscParams.offsets;

    return half4(yuv * cscParams.matrix, 1.0h);
}

fragment half4 ps_draw_rgb(Vertex v [[ stage_in ]],
                           texture2d<half> rgbTexture [[ texture(0) ]])
{
    return rgbTexture.sample(s, v.texCoords);
}

// AMD FidelityFX FSR 1 RCAS (Robust Contrast Adaptive Sharpening), MIT licence,
// Copyright (c) 2021 Advanced Micro Devices, Inc. Applied 1:1 after MetalFX upscaling.
// `sharpness` is FSR's con.x = exp2(-stops): 1.0 strongest, 0.25 mild.
fragment float4 ps_draw_rcas(Vertex v [[ stage_in ]],
                             constant float &sharpness [[ buffer(0) ]],
                             texture2d<float> tex [[ texture(0) ]])
{
    int2 size = int2(tex.get_width(), tex.get_height());
    int2 p = clamp(int2(v.texCoords * float2(size)), int2(0), size - 1);
    float3 b = tex.read(uint2(clamp(p + int2(0, -1), int2(0), size - 1))).rgb;
    float3 d = tex.read(uint2(clamp(p + int2(-1, 0), int2(0), size - 1))).rgb;
    float4 ec = tex.read(uint2(p));
    float3 e = ec.rgb;
    float3 f = tex.read(uint2(clamp(p + int2(1, 0), int2(0), size - 1))).rgb;
    float3 h = tex.read(uint2(clamp(p + int2(0, 1), int2(0), size - 1))).rgb;

    float3 mn4 = min(min(b, d), min(f, h));
    float3 mx4 = max(max(b, d), max(f, h));
    float3 hitMin = min(mn4, e) / max(4.0 * mx4, 1e-5);
    float3 hitMax = (1.0 - max(mx4, e)) / min(4.0 * mn4 - 4.0, -1e-5);
    float3 lobeRGB = max(-hitMin, hitMax);
    const float RCAS_LIMIT = 0.25 - 1.0 / 16.0;
    float lobe = max(-RCAS_LIMIT, min(max(lobeRGB.r, max(lobeRGB.g, lobeRGB.b)), 0.0)) * sharpness;
    float3 c = (lobe * (b + d + f + h) + e) / (4.0 * lobe + 1.0);
    return float4(c, ec.a);
}
