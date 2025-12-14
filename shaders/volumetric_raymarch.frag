#version 410 core
in vec2 v_uv;
out vec4 FragColor;

uniform sampler2D gPosition;

uniform vec3 u_eye;

uniform vec3  u_volLightPos;       
uniform vec3  u_volLightColor;     
uniform float u_volLightIntensity;  

uniform int   u_steps;              // 64 ~ 256
uniform float u_maxDistance;        // e.g. 30~50
uniform float u_baseDensity;        // e.g. 0.02~0.08
uniform float u_heightFalloff;      // e.g. 0.05~0.2
uniform float u_anisotropy;         // HG g (-0.2 ~ 0.9)
uniform float u_extinction;         // absorption strength

const float PI = 3.14159265;

float phaseHG(float cosTheta, float g)
{
    float g2 = g * g;
    float denom = pow(max(1.0 + g2 - 2.0 * g * cosTheta, 1e-4), 1.5);
    return (1.0 - g2) / max(4.0 * PI * denom, 1e-4);
}

float fogDensityAt(vec3 p)
{
    return u_baseDensity * exp(-p.y * u_heightFalloff);
}

vec3 fallbackViewDir(vec2 uv)
{
    return normalize(vec3(uv * 2.0 - 1.0, -1.0));
}

uniform samplerCube u_pointShadowMap;
uniform float u_pointShadowFar;
uniform float u_shadowBias;

float pointLightShadow(vec3 samplePos, vec3 lightPos)
{
    vec3 toLight = samplePos - lightPos;
    float currentDepth = length(toLight);

    float closestDepth =
        texture(u_pointShadowMap, toLight).r * u_pointShadowFar;

    float bias = u_shadowBias;

    return (currentDepth - bias > closestDepth) ? 0.0 : 1.0;
}


void main()
{
    //vec3 worldEnd = texture(gPosition, v_uv).xyz;
    //bool hitGeometry = (length(worldEnd) > 1e-4);

    vec4 posData = texture(gPosition, v_uv);
    bool hitGeometry = (posData.a > 0.5);
    vec3 worldEnd = posData.xyz;

    vec3 rayStart = u_eye;
    vec3 rayDir;
    float marchDist;

    if (hitGeometry)
    {
        vec3 rayVec = worldEnd - rayStart;
        float dist = length(rayVec);

        if (dist < 1e-4)
        {
            rayDir = vec3(0.0, 0.0, -1.0);
            marchDist = u_maxDistance;
        }
        else
        {
            rayDir = rayVec / dist;
            marchDist = min(dist, u_maxDistance);
        }
    }
    else
    {
        rayDir = fallbackViewDir(v_uv);
        marchDist = u_maxDistance;
    }

    float stepLen = marchDist / max(float(u_steps), 1.0);
    vec3  stepVec = rayDir * stepLen;

    vec3 V = normalize(-rayDir);

    vec3 scattering = vec3(0.0);
    float T = 1.0; // transmittance

    vec3 p = rayStart;

    for (int i = 0; i < u_steps; ++i)
    {
        p += stepVec;

        float dens = fogDensityAt(p);
        if (dens < 1e-6) continue;

        float sigma = dens * u_extinction;

        vec3 toLight = u_volLightPos - p;
        float distL = length(toLight);
        if (distL < 1e-4) continue;

        vec3 L = toLight / distL;

        float atten = exp(-distL * 0.000002);


        float cosTheta = clamp(dot(V, L), -1.0, 1.0);
        float phase = phaseHG(cosTheta, u_anisotropy);

        float visibility = pointLightShadow(p, u_volLightPos);
        //float visibility = 1.0;


        vec3 inscatter =
            u_volLightColor *
            u_volLightIntensity *
            atten *
            dens *
            phase *
            visibility;

        scattering += T * inscatter * stepLen;

        T *= exp(-sigma * stepLen);
        if (T < 0.01) break;
    }
    FragColor = vec4(scattering, 1.0);
}
