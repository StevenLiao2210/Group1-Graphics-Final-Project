#version 430 core

in vec3 f_viewVertex;
in vec3 f_uv;
// NEW: view-space normal
in vec3 f_viewNormal;
// filter
in vec3 f_worldVertex;
in vec3 f_worldNormal;
flat in int f_objectType;

layout (location = 0) out vec4 fragColor;

layout(location = 2) uniform int pixelProcessId;
layout(location = 4) uniform sampler2D albedoTexture;

layout(location = 10) uniform int filterMode;

// NEW: normal mapping
layout(location = 6)  uniform sampler2D normalMap; 
layout(location = 11) uniform int useNormalMapping;
layout(location = 0)  uniform mat4 modelMat;
layout(location = 7)  uniform mat4 viewMat;


vec4 withFog(vec4 color) {
    const vec4  FOG_COLOR = vec4(0.0, 0.0, 0.0, 1.0);
    const float MAX_DIST  = 400.0;
    const float MIN_DIST  = 350.0;

    float dis = length(f_viewVertex);
    float fogFactor = (MAX_DIST - dis) / (MAX_DIST - MIN_DIST);
    fogFactor = clamp(fogFactor, 0.0, 1.0);
    fogFactor = fogFactor * fogFactor;

    vec4 colorWithFog = mix(FOG_COLOR, color, fogFactor);
    return colorWithFog;
}

// filter
vec4 worldSpaceVertexFilter() {
    vec3 n = normalize(f_worldVertex);
    vec3 color = n * 0.5 + 0.5;
    return vec4(color, 1.0);
}

vec4 worldSpaceNormalFilter() {
    vec3 n = normalize(f_worldNormal);
    vec3 color = n * 0.5 + 0.5;
    return vec4(color, 1.0);
}

vec4 diffuseFilter() {
    vec3 color;
    color = texture(albedoTexture, f_uv.xy).rgb;
    return vec4(color, 1.0);
}


vec4 specularFilter() {
    if (f_objectType == 0) {
        return vec4(1.0, 1.0, 1.0, 1.0);
    } else {
        return vec4(0.0, 0.0, 0.0, 1.0);
    }
}



// Generic Blinn-Phong using view-space N, V, light dir
vec4 blinnPhong(vec4 baseColor, vec3 N_in) {
    vec3 N = normalize(N_in);
    vec3 V = normalize(-f_viewVertex);           // camera at origin in view space

    // simple directional light in view space
    vec3 L = normalize(vec3(0.3, 0.7, 0.2));
    vec3 H = normalize(L + V);

    float diff = max(dot(N, L), 0.0);

    float spec = 0.0;
    if (diff > 0.0)
        spec = pow(max(dot(N, H), 0.0), 32.0);   // shininess

    vec3 ambient = 0.15 * baseColor.rgb;
    vec3 diffuse = diff * baseColor.rgb;
    vec3 specular = 0.5 * spec * vec3(1.0);

    vec3 color = ambient + diffuse + specular;
    return vec4(color, baseColor.a);
}

// Terrain pass: textured + Blinn-Phong + fog
void terrainPass() {
    vec4 texel = texture(albedoTexture, f_uv.xy);
    vec4 lit   = blinnPhong(texel, f_viewNormal);
    fragColor  = withFog(lit);
    fragColor.a = 1.0;
}

// Simple red (for frustum etc.)
void pureColor() {
    fragColor = withFog(vec4(1.0, 0.0, 0.0, 1.0));
}

// Dynamic objects: solid color + Blinn-Phong + fog
void blinnPhongObject() {

    vec4 base = texture(albedoTexture, f_uv.xy);

    vec3 N_geom = normalize(f_viewNormal);
    vec3 N = N_geom;

    if (useNormalMapping == 1) {

        vec3 n_ts = texture(normalMap, f_uv.xy).rgb;
        n_ts = n_ts * 2.0 - 1.0;

        vec3 up = (abs(N_geom.z) < 0.999) ? vec3(0.0, 0.0, 1.0) : vec3(0.0, 1.0, 0.0);
        vec3 T = normalize(cross(up, N_geom));
        vec3 B = cross(N_geom, T);
        mat3 TBN = mat3(T, B, N_geom);

        vec3 N_bump = normalize(TBN * n_ts);

        float strength = 0.5;  // 可以試 0.3 ~ 0.7 調整感覺
        N = normalize(mix(N_geom, N_bump, strength));
    }

    vec4 lit  = blinnPhong(base, N);
    fragColor = withFog(lit);
}
void main() {

    if (filterMode == 1) {
        fragColor = worldSpaceVertexFilter();
        return;
    }
    if (filterMode == 2) {
        fragColor = worldSpaceNormalFilter();
        return;
    }
    if (filterMode == 3) {
        fragColor = diffuseFilter();
        return;
    }
    if (filterMode == 4) {
        fragColor = specularFilter();
        return;
    }

    if (pixelProcessId == 5) {
        // used by view frustum lines etc.
        pureColor();
    }
    else if (pixelProcessId == 6) {
        // airplane + magic rock
        blinnPhongObject();
    }
    else if (pixelProcessId == 7) {
        // terrain
        terrainPass();
    }
    else {
        pureColor();
    }
}
