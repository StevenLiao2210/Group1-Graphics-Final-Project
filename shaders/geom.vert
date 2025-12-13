#version 410 core
layout(location = 0) in vec3 a_pos;
layout(location = 1) in vec3 a_norm;
layout(location = 2) in vec2 a_uv;

out vec3 v_worldPos;
out vec3 v_worldNorm;
out vec2 v_uv;

uniform mat4 u_model;
uniform mat4 u_view;
uniform mat4 u_proj;

void main()
{
    vec4 wp = u_model * vec4(a_pos, 1.0);
    v_worldPos  = wp.xyz;
    v_worldNorm = mat3(transpose(inverse(u_model))) * a_norm;
    v_uv        = a_uv;

    gl_Position = u_proj * u_view * wp;
}