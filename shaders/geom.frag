#version 410 core
in vec3 v_worldPos;
in vec3 v_worldNorm;
in vec2 v_uv;

layout(location = 0) out vec4 gPosition;  
layout(location = 1) out vec4 gNormal;    
layout(location = 2) out vec4 gAmbient;   
layout(location = 3) out vec4 gDiffuse;   
layout(location = 4) out vec4 gSpecular;  

uniform vec3 u_Ka;
uniform vec3 u_Kd;
uniform vec3 u_Ks;
uniform float u_Ns;

uniform sampler2D u_diffuseTex;
uniform bool      u_useTex;

uniform sampler2D u_normalMap;
uniform bool u_useNormalMap;

void main()
{
    vec3 Ka = u_Ka;
    vec3 Kd = u_Kd;
    if (u_useTex) {
        vec4 tex = texture(u_diffuseTex, v_uv);
        if (tex.a < 0.5)
            discard; 
        Kd = tex.rgb; 
    }

    vec3 N = normalize(v_worldNorm);

    if (u_useNormalMap) {
    vec3 nTex = texture(u_normalMap, v_uv).xyz * 2.0 - 1.0;

    nTex.y = -nTex.y;

    vec3 dp1  = dFdx(v_worldPos);
    vec3 dp2  = dFdy(v_worldPos);
    vec2 duv1 = dFdx(v_uv);
    vec2 duv2 = dFdy(v_uv);

    vec3 T = normalize(dp1 * duv2.y - dp2 * duv1.y);

    T = normalize(T - N * dot(N, T));

    vec3 B = normalize(cross(N, T));
    float handedness = (dot(cross(dp1, dp2), N) < 0.0) ? -1.0 : 1.0;
    B *= handedness;

    mat3 TBN = mat3(T, B, N);
    N = normalize(TBN * nTex);
}
    gPosition = vec4(v_worldPos, 1.0);
    gNormal   = vec4(N, 1.0);
    gAmbient  = vec4(Ka, 1.0);  
    gDiffuse  = vec4(Kd, 1.0);
    gSpecular = vec4(u_Ks, u_Ns);
}