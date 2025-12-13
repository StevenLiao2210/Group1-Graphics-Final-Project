#version 410 core
in vec2 v_uv;
out vec4 FragColor;

uniform sampler2D u_scene;         // occlusion buffer
uniform vec2      u_lightScreenPos;
uniform int       u_numSamples;
uniform float     u_exposure;
uniform float     u_decay;
uniform float     u_density;
uniform float     u_weight;

void main()
{
    vec2 coord = v_uv;
    vec2 delta = coord - u_lightScreenPos;
    delta *= (u_density / float(u_numSamples));

    float illumDecay = 1.0;
    float sum = 0.0;

    for (int i = 0; i < u_numSamples; ++i) {
        coord -= delta;
        float occ = texture(u_scene, coord).r;
        sum += occ * illumDecay * u_weight;
        illumDecay *= u_decay;
    }

    float result = sum * u_exposure;
    FragColor = vec4(vec3(result), 1.0);
}
