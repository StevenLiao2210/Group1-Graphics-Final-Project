#version 410 core
in vec2 v_uv;
out vec4 FragColor;

uniform sampler2D u_scene;
uniform sampler2D u_bloomBlur;
uniform sampler2D u_volumetric;

uniform bool  u_enableBloom;
uniform bool  u_enableVolumetric;
uniform float u_bloomIntensity;
uniform float u_volumetricIntensity;
uniform float u_exposure;


void main()
{
    vec3 hdrColor   = texture(u_scene,     v_uv).rgb;
    vec3 bloomColor = texture(u_bloomBlur, v_uv).rgb;
    vec3 volColor   = vec3(0.0);

    if (u_enableVolumetric) {
        volColor = texture(u_volumetric, v_uv).rgb * u_volumetricIntensity;
    }

    // Treat volumetric as extra HDR light before tone mapping
    hdrColor += volColor;

    if (u_enableBloom) {
        hdrColor += bloomColor * u_bloomIntensity;
    }

    // simple exponential tone mapping
    vec3 mapped = vec3(1.0) - exp(-hdrColor * u_exposure);

    // gamma correction
    FragColor = vec4(mapped, 1.0);
}