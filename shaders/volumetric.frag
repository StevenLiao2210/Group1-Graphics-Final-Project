#version 410 core
in vec2 v_uv;
out vec4 FragColor;

// This is now the *bright-pass / occlusion* texture
uniform sampler2D u_scene;          // g_brightColorTex

uniform vec2  u_lightScreenPos;     // [0,1] screen-space
uniform int   u_numSamples;         // e.g. 100
uniform float u_exposure;           // 0.0 ~ 1.0
uniform float u_decay;              // ~0.96815
uniform float u_density;           // ~0.926
uniform float u_weight;             // ~0.58767
uniform float u_sourceRadius;

// Optional: threshold on the occlusion buffer to ignore tiny noise
uniform float u_threshold;          // you already have this uniform

void main()
{
    // If light is off-screen, no rays
    if (u_lightScreenPos.x < 0.0 || u_lightScreenPos.x > 1.0 ||
        u_lightScreenPos.y < 0.0 || u_lightScreenPos.y > 1.0)
    {
        FragColor = vec4(0.0);
        return;
    }

    vec2 texCoord      = v_uv;
    vec2 delta         = (texCoord - u_lightScreenPos);
    delta             *= (u_density / float(u_numSamples));

    vec2  sampleCoord        = texCoord;
    float illuminationDecay  = 1.0;
    float accumulatedLumin   = 0.0;

    for (int i = 0; i < u_numSamples; ++i)
    {
        sampleCoord -= delta;

        // Sample the bright-pass/occlusion buffer
        vec3 c = texture(u_scene, sampleCoord).rgb;

        // Convert to a single scalar (luminance or just max component)
        float lum = max(max(c.r, c.g), c.b);


        float d = distance(sampleCoord, u_lightScreenPos);
        float sourceMask = smoothstep(u_sourceRadius, 0.0, d); // 1 near light, 0 far
        lum *= sourceMask;


        // Optional threshold to kill tiny noise
        if (lum > u_threshold)
        {
            lum *= illuminationDecay * u_weight;
            accumulatedLumin += lum;
        }

        illuminationDecay *= u_decay;
    }

    float intensity = accumulatedLumin * u_exposure;
    FragColor = vec4(vec3(intensity), 1.0);
}