#version 410 core
layout(location = 0) in vec3 a_pos;

out vec3 v_worldPos;

uniform mat4 u_model;
uniform mat4 u_lightVP;   // per-face view-proj matrix

void main()
{
    vec4 wp = u_model * vec4(a_pos, 1.0);
    v_worldPos = wp.xyz;
    gl_Position = u_lightVP * wp;
}