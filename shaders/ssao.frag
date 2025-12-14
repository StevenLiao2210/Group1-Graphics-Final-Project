#version 410 core
in vec2 v_uv;
out float FragColor;

uniform sampler2D gPosition;  
uniform sampler2D gNormal;    
uniform sampler2D texNoise;

uniform vec3  u_samples[64]; 
uniform mat4  u_view;
uniform mat4  u_proj;
uniform float u_radius;
uniform float u_bias;
uniform vec2  u_noiseScale;

void main()
{
    vec3 worldPos = texture(gPosition, v_uv).xyz;
    vec3 worldN   = texture(gNormal,   v_uv).xyz;

    if (worldPos == vec3(0.0)) { FragColor = 1.0; return; }

    vec3 fragPos = (u_view * vec4(worldPos, 1.0)).xyz;
    vec3 normal  = normalize(mat3(u_view) * worldN);

    vec3 randomVec = normalize(texture(texNoise, v_uv * u_noiseScale).xyz);

    vec3 tangent   = normalize(randomVec - normal * dot(randomVec, normal));
    vec3 bitangent = cross(normal, tangent);
    mat3 TBN       = mat3(tangent, bitangent, normal);

    float occlusion = 0.0;

    for (int i = 0; i < 64; ++i)
    {
        vec3 samp = TBN * u_samples[i];
        vec3 samplePos = fragPos + samp * u_radius;

        vec4 offset = u_proj * vec4(samplePos, 1.0);
        offset.xyz /= offset.w;
        vec2 uv = offset.xy * 0.5 + 0.5;

        if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) continue;

        vec3 sampleWorldPos = texture(gPosition, uv).xyz;
        if (sampleWorldPos == vec3(0.0)) continue;

        float sampleDepth = (u_view * vec4(sampleWorldPos, 1.0)).z; // VIEW z
        float rangeCheck = smoothstep(0.0, 1.0, u_radius / abs(fragPos.z - sampleDepth));
        if (sampleDepth >= samplePos.z + u_bias)
            occlusion += rangeCheck;
    }

    float ao = 1.0 - (occlusion / 64.0);
    FragColor = clamp(ao, 0.0, 1.0);
}