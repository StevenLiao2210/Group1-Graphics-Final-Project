#version 410 core
in vec2 v_uv;

layout(location = 0) out vec4 FragColor;
layout(location = 1) out vec4 BrightColor;

uniform sampler2D gPosition;
uniform sampler2D gNormal;
uniform sampler2D gAmbient;
uniform sampler2D gDiffuse;
uniform sampler2D gSpecular;

uniform vec3 u_eye;
uniform vec3 u_lightPos;  
uniform vec3 u_Ia;
uniform vec3 u_Id;
uniform vec3 u_Is;

uniform float u_attConst;
uniform float u_attLinear;
uniform float u_attQuadratic;

uniform samplerCube u_shadowCube;
uniform float       u_far;

uniform sampler2D u_dirShadowMap;
uniform mat4 u_lightVP;       
uniform vec3 u_lightDir;      
uniform float u_shadowStrength;
uniform bool u_enableShadows;

uniform bool u_enableBloom;
uniform float u_bloomThreshold;

uniform bool u_enableSSAO;
uniform sampler2D u_ssaoTex;

uniform bool  u_enableSSR;
uniform mat4  u_view;
uniform mat4  u_proj;
uniform float u_ssrMaxDistance;
uniform int   u_ssrMaxSteps;
uniform float u_ssrStep;
uniform float u_ssrThickness;
uniform float u_ssrIntensity;

uniform vec3 u_rectCenter;
uniform vec3 u_rectNormal;
uniform vec3 u_rectTangent;
uniform vec3 u_rectBitangent;
uniform vec2 u_rectSize;       
uniform vec3 u_rectColor;
uniform int  u_rectSamples;
uniform bool u_enableAreaLight;

// Debug view mode
// 0 = lighting, 1 = pos, 2 = normal, 3 = ambient,
// 4 = diffuse, 5 = specular, 6 = SSAO
uniform int u_viewMode;

uniform bool u_enableToon;
uniform int  u_toonSteps;

uniform bool  u_enableEdges;
uniform float u_edgeDepthThreshold;
uniform float u_edgeNormalThreshold;
uniform float u_edgeStrength;

float toonQuantize(float x, int steps)
{
    steps = max(steps, 1);
    x = clamp(x, 0.0, 1.0);
    float s = float(max(steps - 1, 1));
    return floor(x * s + 0.5) / s;
}

float toonSpec(float specPow)
{
    float s = clamp(specPow, 0.0, 1.0);

    float hi = step(0.65, s);       // hard highlight
    float mid = step(0.35, s) * (1.0 - hi); // mid highlight
    return 0.0 + 0.5 * mid + 1.0 * hi;
}

float sobelEdge(vec2 uv)
{
    vec2 texel = 1.0 / vec2(textureSize(gNormal, 0));

    vec2 o[9] = vec2[](
        vec2(-1,-1), vec2(0,-1), vec2(1,-1),
        vec2(-1, 0), vec2(0, 0), vec2(1, 0),
        vec2(-1, 1), vec2(0, 1), vec2(1, 1)
    );

    float kx[9] = float[](
        -1, 0, 1,
        -2, 0, 2,
        -1, 0, 1
    );
    float ky[9] = float[](
        -1,-2,-1,
         0, 0, 0,
         1, 2, 1
    );

    float gxD = 0.0, gyD = 0.0;
    vec3  gxN = vec3(0.0);
    vec3  gyN = vec3(0.0);

    for (int i = 0; i < 9; ++i)
    {
        vec2 suv = uv + o[i] * texel;

        vec3 p = texture(gPosition, suv).xyz;
        vec3 n = texture(gNormal,   suv).xyz;

        float d = (p == vec3(0.0)) ? 1e6 : length(p - u_eye);

        gxD += d * kx[i];
        gyD += d * ky[i];

        gxN += n * kx[i];
        gyN += n * ky[i];
    }

    float magD = length(vec2(gxD, gyD));
    float magN = length(gxN) + length(gyN);

    float eD = smoothstep(u_edgeDepthThreshold,  u_edgeDepthThreshold * 2.0, magD);
    float eN = smoothstep(u_edgeNormalThreshold, u_edgeNormalThreshold * 2.0, magN);

    return clamp(max(eD, eN), 0.0, 1.0);
}

float calcDirShadow(vec3 worldPos, vec3 N)
{
    vec4 lp = u_lightVP * vec4(worldPos, 1.0);
    vec3 ndc = lp.xyz / lp.w;
    vec3 sc  = ndc * 0.5 + 0.5;

    if (sc.x < 0.0 || sc.x > 1.0 || sc.y < 0.0 || sc.y > 1.0 || sc.z < 0.0 || sc.z > 1.0)
        return 0.0;

    float NdotL = max(dot(N, u_lightDir), 0.0);
    float bias  = max(0.0025 * (1.0 - NdotL), 0.0005);

    vec2 texelSize = 1.0 / vec2(textureSize(u_dirShadowMap, 0));
    float shadow = 0.0;

    for (int y = -1; y <= 1; ++y)
    for (int x = -1; x <= 1; ++x)
    {
        float pcfDepth = texture(u_dirShadowMap, sc.xy + vec2(x,y) * texelSize).r;
        shadow += (sc.z - bias > pcfDepth) ? 1.0 : 0.0;
    }
    shadow /= 9.0;
    return shadow;
}

float calcPointShadow(vec3 worldPos)
{
    vec3 L = worldPos - u_lightPos;
    float currentDepth = length(L);
    float closestDepth = texture(u_shadowCube, L).r * u_far;

    float bias = 0.03;
    return (currentDepth - bias > closestDepth) ? 1.0 : 0.0;
}

vec3 evalRectAreaLightDiffuse(vec3 pos, vec3 N, vec3 Kd)
{
    const int MAX_SAMPLES = 32;
    int S = clamp(u_rectSamples, 1, MAX_SAMPLES);

    float halfW = 0.5 * u_rectSize.x;
    float halfH = 0.5 * u_rectSize.y;
    float area  = u_rectSize.x * u_rectSize.y;

    int grid = int(ceil(sqrt(float(S))));
    int used = 0;

    float accumG = 0.0;

    for (int j = 0; j < grid && used < S; ++j)
    for (int i = 0; i < grid && used < S; ++i, ++used)
    {
        float u = (float(i) + 0.5) / float(grid) - 0.5;
        float v = (float(j) + 0.5) / float(grid) - 0.5;

        vec3 samplePos =
            u_rectCenter +
            (u * 2.0 * halfW) * u_rectTangent +
            (v * 2.0 * halfH) * u_rectBitangent;

        vec3 Lvec = samplePos - pos;
        float d2  = dot(Lvec, Lvec);
        if (d2 <= 0.0) continue;

        vec3 wi = normalize(Lvec);

        float NdotL = max(dot(N, wi), 0.0);
        if (NdotL <= 0.0) continue;

        float NL_light = max(dot(-u_rectNormal, wi), 0.0);
        if (NL_light <= 0.0) continue;

        float G = (NdotL * NL_light) / d2;

        if (u_enableToon) {
            // Use the NdotL term; keep geometric term continuous
            float q = toonQuantize(NdotL, u_toonSteps);
            G *= (q / max(NdotL, 1e-6));
        }

        accumG += G;
    }

    if (used == 0) return vec3(0.0);

    float factor  = area / float(used);
    vec3 radiance = u_rectColor * accumG * factor;

    return Kd * radiance;
}

bool isFloorPixel(vec3 pos, vec3 normal)
{
    float epsY = 0.2;
    bool nearPlane = (pos.y > -epsY && pos.y < epsY);
    bool normalUp  = normal.y > 0.8;
    return nearPlane && normalUp;
}

float viewDepthOfPos(vec3 worldPos) {
    return -(u_view * vec4(worldPos, 1.0)).z; // positive forward depth
}

float viewDepthAtUV(vec2 uv) {
    vec3 p = texture(gPosition, uv).xyz;
    if (p == vec3(0.0)) return 1e9;
    return -(u_view * vec4(p, 1.0)).z;
}

float minNeighborDepth(vec2 uv) {
    ivec2 ts = textureSize(gPosition, 0);
    vec2 texel = 1.0 / vec2(ts);

    float m = 1e9;
    for (int y = -1; y <= 1; ++y) {
        for (int x = -1; x <= 1; ++x) {
            vec2 uv2 = uv + vec2(x, y) * texel;
            m = min(m, viewDepthAtUV(uv2));
        }
    }
    return m;
}

bool ssrSegmentOccluded(vec2 uv0, vec3 p0, vec2 uv1, vec3 p1)
{
    const int   N = 16;      // 8~24 (higher = stricter / slower)
    const float eps = 0.02;  // 0.01~0.06 (tune)

    for (int i = 1; i < N; ++i)
    {
        float a = float(i) / float(N);

        vec2 uvi = mix(uv0, uv1, a);
        vec3 pri = mix(p0,  p1,  a);   // point on the ray segment (world)

        // depth of ray segment at this screen location
        float zRay = viewDepthOfPos(pri);

        // depth buffer at this screen location
        float zBuf = viewDepthAtUV(uvi);
        if (zBuf >= 1e8) continue;

        // If buffer is closer than the ray segment -> something blocks it
        if (zBuf + eps < zRay)
            return true;
    }
    return false;
}

vec3 computeSSR(vec3 pos, vec3 normal, vec3 Ka, vec3 Kd, vec3 Ks, float NsRaw)
{
    vec3 N = normalize(normal);
    vec3 V = normalize(u_eye - pos);
    vec3 R = normalize(reflect(-V, N));

    vec3 rayOrigin = pos + N * 0.02;

    vec4 oClip = u_proj * u_view * vec4(rayOrigin, 1.0);
    vec2 uv0   = (oClip.xy / oClip.w) * 0.5 + 0.5;

    float tPrev = 0.0;
    float diffPrev = 0.0;
    bool  hasPrev = false;

    float t = 0.0;

    for (int i = 0; i < u_ssrMaxSteps; ++i)
    {
        if (t > u_ssrMaxDistance) break;

        vec3 samplePos = rayOrigin + R * t;

        vec4 clip = u_proj * u_view * vec4(samplePos, 1.0);
        if (clip.w <= 0.0) { t += u_ssrStep; continue; }

        vec3 ndc = clip.xyz / clip.w;
        if (ndc.x < -1.0 || ndc.x > 1.0 || ndc.y < -1.0 || ndc.y > 1.0) {
            t += u_ssrStep; continue;
        }

        vec2 uv = ndc.xy * 0.5 + 0.5;

        vec3 scenePos = texture(gPosition, uv).xyz;
        if (scenePos == vec3(0.0)) { t += u_ssrStep; continue; }

        vec3 sceneNormal = texture(gNormal, uv).xyz;

        if (isFloorPixel(scenePos, sceneNormal)) { t += u_ssrStep; continue; }

        if (dot(normalize(sceneNormal), -R) < 0.05) { t += u_ssrStep; continue; }

        float zRay   = -(u_view * vec4(samplePos, 1.0)).z; // positive forward depth
        float zScene = -(u_view * vec4(scenePos,  1.0)).z;

        float diff = zRay - zScene; // >0 means ray is behind geometry at that uv

        if (hasPrev && diff > 0.0 && diffPrev < 0.0)
        {
            float a = tPrev;
            float b = t;

            for (int it = 0; it < 6; ++it)
            {
                float m = 0.5 * (a + b);
                vec3 mp = rayOrigin + R * m;

                vec4 c  = u_proj * u_view * vec4(mp, 1.0);
                if (c.w <= 0.0) { a = m; continue; }

                vec3 n2 = c.xyz / c.w;
                vec2 uv2 = n2.xy * 0.5 + 0.5;

                vec3 sp2 = texture(gPosition, uv2).xyz;
                if (sp2 == vec3(0.0)) { a = m; continue; }

                float zR2 = -(u_view * vec4(mp,  1.0)).z;
                float zS2 = -(u_view * vec4(sp2, 1.0)).z;

                float d2 = zR2 - zS2;

                if (d2 > 0.0) b = m; else a = m;
            }

            vec3 hitPos = rayOrigin + R * b;
            vec4 hitClip = u_proj * u_view * vec4(hitPos, 1.0);
            vec3 hitNdc = hitClip.xyz / hitClip.w;
            vec2 hitUV  = hitNdc.xy * 0.5 + 0.5;

            if (ssrSegmentOccluded(uv0, rayOrigin, uv, scenePos)) {
                tPrev = t;
                diffPrev = diff;
                hasPrev = true;
                t += u_ssrStep;
                continue;
            }

            vec3 hitScenePos = texture(gPosition, hitUV).xyz;
            if (hitScenePos == vec3(0.0)) { t += u_ssrStep; continue; }

            if (ssrSegmentOccluded(uv0, rayOrigin, hitUV, hitScenePos)) {
                t += u_ssrStep;
                continue;
            }


            vec3 hitKa   = texture(gAmbient,  hitUV).rgb;
            vec3 hitKd   = texture(gDiffuse,  hitUV).rgb;
            vec4 hitSpec = texture(gSpecular, hitUV);
            vec3 hitKs   = hitSpec.rgb;
            float hitNsRaw = hitSpec.a;

            bool hitEmissive = (hitNsRaw < 0.0);
            float hitNs = max(hitNsRaw, 1.0);

            if (hitEmissive) {
                return hitKd; // emissive stored in Kd
            }

            vec3 hN = normalize(texture(gNormal, hitUV).xyz);
            vec3 Ld = normalize(u_lightDir);
            vec3 V2 = normalize(u_eye - texture(gPosition, hitUV).xyz);
            vec3 H2 = normalize(Ld + V2);

            float NdotL2 = max(dot(hN, Ld), 0.0);
            float NdotH2 = max(dot(hN, H2), 0.0);

            float ao2 = 1.0;
            if (u_enableSSAO) ao2 = texture(u_ssaoTex, hitUV).r;

            vec3 ambient2  = (hitKa * hitKd) * u_Ia * ao2;
            vec3 diffuse2  = u_Id * hitKd * NdotL2;
            vec3 specular2 = (NdotL2 > 0.0) ? (u_Is * hitKs * pow(NdotH2, hitNs)) : vec3(0.0);

            return ambient2 + diffuse2 + specular2;
        }

        if (diff > 0.0 && diff < u_ssrThickness)
        {
            float zHit = -(u_view * vec4(scenePos, 1.0)).z; 
            float zMin = minNeighborDepth(uv);              

            if (zMin + 0.05 < zHit) { // try 0.03~0.08
                // treat as no hit, keep marching
                tPrev = t;
                diffPrev = diff;
                hasPrev = true;
                t += u_ssrStep;
                continue;
            }
           
            vec3 hitKa   = texture(gAmbient, uv).rgb;
            vec3 hitKd   = texture(gDiffuse, uv).rgb;
            vec4 hitSpec = texture(gSpecular, uv);
            vec3 hitKs   = hitSpec.rgb;
            float hitNsRaw = hitSpec.a;

            bool hitEmissive = (hitNsRaw < 0.0);
            float hitNs = max(hitNsRaw, 1.0);

            if (hitEmissive) return hitKd;

            vec3 hN = normalize(sceneNormal);
            vec3 Ld = normalize(u_lightDir);
            vec3 V2 = normalize(u_eye - scenePos);
            vec3 H2 = normalize(Ld + V2);

            float NdotL2 = max(dot(hN, Ld), 0.0);
            float NdotH2 = max(dot(hN, H2), 0.0);

            float ao2 = 1.0;
            if (u_enableSSAO) ao2 = texture(u_ssaoTex, uv).r;

            vec3 ambient2  = (hitKa * hitKd) * u_Ia * ao2;
            vec3 diffuse2  = u_Id * hitKd * NdotL2;
            vec3 specular2 = (NdotL2 > 0.0) ? (u_Is * hitKs * pow(NdotH2, hitNs)) : vec3(0.0);

            return ambient2 + diffuse2 + specular2;
        }

        tPrev = t;
        diffPrev = diff;
        hasPrev = true;

        t += u_ssrStep;
    }

    return vec3(0.0);
}

void main()
{
    vec3 pos      = texture(gPosition, v_uv).xyz;
    vec3 normal   = texture(gNormal,   v_uv).xyz;
    vec3 Ka       = texture(gAmbient,  v_uv).rgb;
    vec3 Kd       = texture(gDiffuse,  v_uv).rgb;
    vec4 specData = texture(gSpecular, v_uv);
    vec3 Ks       = specData.rgb;
    float NsRaw   = specData.a;

    bool  isEmissive = (NsRaw < 0.0);
    float Ns = max(NsRaw, 1.0);

    if (pos == vec3(0.0)) {
        FragColor   = vec4(0.0);
        BrightColor = vec4(0.0);
        return;
    }

    if (u_viewMode == 1) { FragColor = vec4(normalize(pos) * 0.5 + 0.5, 1.0); BrightColor = vec4(0.0); return; }
    if (u_viewMode == 2) { FragColor = vec4(normalize(normal) * 0.5 + 0.5, 1.0); BrightColor = vec4(0.0); return; }
    if (u_viewMode == 3) { FragColor = vec4(Ka, 1.0); BrightColor = vec4(0.0); return; }
    if (u_viewMode == 4) { FragColor = vec4(Kd, 1.0); BrightColor = vec4(0.0); return; }
    if (u_viewMode == 5) { FragColor = vec4(Ks, 1.0); BrightColor = vec4(0.0); return; }
    if (u_viewMode == 6) {
        float aoDbg = u_enableSSAO ? texture(u_ssaoTex, v_uv).r : 1.0;
        FragColor = vec4(vec3(aoDbg), 1.0); BrightColor = vec4(0.0); return;
    }

    if (isEmissive) {
        vec3 emissive = Kd;
        FragColor = vec4(emissive, 1.0);

        vec3 bright = vec3(0.0);
        if (u_enableBloom) {
            float b = max(max(emissive.r, emissive.g), emissive.b);
            if (b > u_bloomThreshold) bright = emissive;
        }
        BrightColor = vec4(bright, 1.0);
        return;
    }

    vec3 N = normalize(normal);
    vec3 V = normalize(u_eye - pos);

    float ao = 1.0;
    if (u_enableSSAO) ao = texture(u_ssaoTex, v_uv).r;

    vec3 ambient = (Ka * Kd) * u_Ia * ao;

    float s = clamp(u_shadowStrength, 0.0, 1.0);

    vec3 Lp = normalize(u_lightPos - pos);
    vec3 Hp = normalize(Lp + V);

    float NdotLp = max(dot(N, Lp), 0.0);
    float qLp    = u_enableToon ? toonQuantize(NdotLp, u_toonSteps) : NdotLp;

    vec3 pointDiffuse = u_Id * Kd * qLp;

    vec3 pointSpecular = vec3(0.0);
    if (NdotLp > 0.0) {
        float specPow = pow(max(dot(N, Hp), 0.0), Ns);
        float specAmt = u_enableToon ? toonSpec(specPow) : specPow;
        pointSpecular = u_Is * Ks * specAmt;
    }

    vec3 directPoint = pointDiffuse + pointSpecular;

    float dist = length(u_lightPos - pos);
    float attenuation = 1.0 / (u_attConst + u_attLinear * dist + u_attQuadratic * dist * dist);
    directPoint *= attenuation;

    float pointShadow = 0.0;
    if (u_enableShadows) pointShadow = calcPointShadow(pos);
    vec3 pointWithShadow = directPoint * mix(s, 1.0, 1.0 - pointShadow);

    vec3 Ld = normalize(u_lightDir);
    vec3 Hd = normalize(Ld + V);

    float NdotLd = max(dot(N, Ld), 0.0);
    float qLd    = u_enableToon ? toonQuantize(NdotLd, u_toonSteps) : NdotLd;

    vec3 dirDiffuse = u_Id * Kd * qLd;

    vec3 dirSpecular = vec3(0.0);
    if (NdotLd > 0.0) {
        float specPow = pow(max(dot(N, Hd), 0.0), Ns);
        float specAmt = u_enableToon ? toonSpec(specPow) : specPow;
        dirSpecular = u_Is * Ks * specAmt;
    }

    vec3 directDir = dirDiffuse + dirSpecular;

    float dirShadow = 0.0;
    if (u_enableShadows) dirShadow = calcDirShadow(pos, N);
    vec3 dirWithShadow = directDir * mix(s, 1.0, 1.0 - dirShadow);

    vec3 directArea = vec3(0.0);
    if (u_enableAreaLight) {
        directArea = evalRectAreaLightDiffuse(pos, N, Kd) * u_Id;
    }

    vec3 lighting = ambient + pointWithShadow + dirWithShadow + directArea;

    if (u_enableEdges) {
        float e = sobelEdge(v_uv);
        lighting *= (1.0 - e * u_edgeStrength);
    }

    if (u_enableSSR && isFloorPixel(pos, normal)) {
        vec3 refl = computeSSR(pos, normal, Ka, Kd, Ks, NsRaw);
        lighting = mix(lighting, refl, u_ssrIntensity);
    }

    vec3 bright = vec3(0.0);
    if (u_enableBloom) {
        float b = max(max(lighting.r, lighting.g), lighting.b);
        if (b > u_bloomThreshold) bright = lighting;
    }

    FragColor   = vec4(lighting, 1.0);
    BrightColor = vec4(bright, 1.0);
}