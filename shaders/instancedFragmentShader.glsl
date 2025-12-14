#version 430 core

in vec3 f_viewVertex;
in vec3 f_uv;
in vec3 f_viewNormal;
in vec3 f_worldVertex;
in vec3 f_worldNormal;
flat in int f_objectType;

layout (location = 0) out vec4 fragColor;

layout(location = 4) uniform sampler2D albedoTexture;
layout(location = 6) uniform sampler2D normalMap;

layout(location = 10) uniform int filterMode;
layout(location = 11) uniform int useNormalMapping;
layout(location = 12) uniform int hasAlphaTexture;  // New uniform for alpha handling

// Fog effect
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

vec4 applyGamma(vec4 color)
{
    // clamp to avoid NaNs
    color.rgb = max(color.rgb, vec3(0.0));
    // gamma = 2.0  (brightens the scene)
    color.rgb = pow(color.rgb, vec3(0.5));
    return color;
}

// Filter modes for debugging
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
    vec4 texel = texture(albedoTexture, f_uv.xy);
    return vec4(texel.rgb, 1.0);
}

vec4 specularFilter() {
    return vec4(1.0, 1.0, 1.0, 1.0);
}

// Blinn-Phong lighting
vec4 blinnPhong(vec4 baseColor, vec3 N_in) {
    vec3 N = normalize(N_in);
    vec3 V = normalize(-f_viewVertex);

    // Directional light in view space
    vec3 L = normalize(vec3(0.3, 0.7, 0.2));
    vec3 H = normalize(L + V);

    float diff = max(dot(N, L), 0.0);

    float spec = 0.0;
    if (diff > 0.0)
        spec = pow(max(dot(N, H), 0.0), 32.0);

    const vec3 Ia = vec3(0.2);   // ambient
    const vec3 Id = vec3(0.64);  // diffuse
    const vec3 Is = vec3(0.16);  // specular

    vec3 ambient = Ia * baseColor.rgb;
    vec3 diffuse = Id * diff * baseColor.rgb;

    // vec3 specular = 0.3 * spec * vec3(1.0);
    vec3 specular = vec3(0.0);

    vec3 color = ambient + diffuse + specular;
    return vec4(color, baseColor.a);
}

void main() {
    // Sample albedo texture
    vec4 texel = texture(albedoTexture, f_uv.xy);
    
    // Alpha testing for foliage (discard fully transparent pixels)
    if (hasAlphaTexture == 1) {
        if (texel.a < 0.5) {
            discard;
        }
    }
    
    // Filter modes
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
    
    // Normal processing
    vec3 N_geom = normalize(f_viewNormal);
    vec3 N = N_geom;
    
    if (useNormalMapping == 1) {
        vec3 n_ts = texture(normalMap, f_uv.xy).rgb;
        n_ts = n_ts * 2.0 - 1.0;
        
        // Approximate TBN
        vec3 up = (abs(N_geom.z) < 0.999) ? vec3(0.0, 0.0, 1.0) : vec3(0.0, 1.0, 0.0);
        vec3 T = normalize(cross(up, N_geom));
        vec3 B = cross(N_geom, T);
        mat3 TBN = mat3(T, B, N_geom);
        
        vec3 N_bump = normalize(TBN * n_ts);
        float strength = 0.5;
        N = normalize(mix(N_geom, N_bump, strength));
    }
    
    // Apply Blinn-Phong lighting
    vec4 lit = blinnPhong(texel, N);
    
    // Apply fog
    fragColor = applyGamma(withFog(lit));
    
    // For alpha blended foliage, preserve alpha
    if (hasAlphaTexture == 1) {
        fragColor.a = texel.a;
    }
}
