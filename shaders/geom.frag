#version 410 core
in vec3 v_worldPos;
in vec3 v_worldNorm;
in vec2 v_uv;

layout(location = 0) out vec4 gPosition;  // world-space position
layout(location = 1) out vec4 gNormal;    // world-space normal
layout(location = 2) out vec4 gAmbient;   // ambient color
layout(location = 3) out vec4 gDiffuse;   // diffuse color
layout(location = 4) out vec4 gSpecular;  // specular.rgb, shininess.a

uniform vec3  u_Ka;
uniform vec3  u_Kd;
uniform vec3  u_Ks;
uniform float u_Ns;

uniform sampler2D u_diffuseTex;
uniform bool      u_useTex;

uniform sampler2D u_normalMap;
uniform bool      u_useNormalMap;

void main()
{
    vec3 Ka = u_Ka;
    vec3 Kd = u_Kd;
    if (u_useTex) {
        vec4 tex = texture(u_diffuseTex, v_uv);
        if (tex.a < 0.5)
            discard; // alpha cutout, e.g., plant
        Kd = tex.rgb;          // only diffuse comes from texture
        // Ka stays as material ambient u_Ka
    }

    // Base world-space normal
    vec3 N = normalize(v_worldNorm);

    // --- Normal mapping (Trice only when enabled) ---
    if (u_useNormalMap) {
        // Build TBN in world space using derivatives
        vec3 dp1  = dFdx(v_worldPos);
        vec3 dp2  = dFdy(v_worldPos);
        vec2 duv1 = dFdx(v_uv);
        vec2 duv2 = dFdy(v_uv);

        float r = 1.0 / (duv1.x * duv2.y - duv1.y * duv2.x);
        vec3 T = normalize((dp1 * duv2.y - dp2 * duv1.y) * r);
        vec3 B = normalize((dp2 * duv1.x - dp1 * duv2.x) * r);

        vec3 nTex = texture(u_normalMap, v_uv).xyz * 2.0 - 1.0; // tangent-space
        mat3 TBN  = mat3(T, B, N);
        N = normalize(TBN * nTex);  // perturbed world-space normal
    }

    gPosition = vec4(v_worldPos, 1.0);
    gNormal   = vec4(N, 0.0);
    gAmbient  = vec4(Ka, 1.0);   // now pure material ambient
    gDiffuse  = vec4(Kd, 1.0);
    gSpecular = vec4(u_Ks, u_Ns);
}