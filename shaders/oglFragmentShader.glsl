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
vec4 blinnPhong(vec4 baseColor) {
    vec3 N = normalize(f_viewNormal);
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
    vec4 lit   = blinnPhong(texel);
    fragColor  = withFog(lit);
    fragColor.a = 1.0;
}

// Simple red (for frustum etc.)
void pureColor() {
    fragColor = withFog(vec4(1.0, 0.0, 0.0, 1.0));
}

// Dynamic objects: solid color + Blinn-Phong + fog
void blinnPhongObject() {
    vec4 base = vec4(1.0, 0.3, 0.2, 1.0);  // tweak if you like
    vec4 lit  = blinnPhong(base);
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
