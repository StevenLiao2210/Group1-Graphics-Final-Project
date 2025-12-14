#version 410 core
in vec3 v_worldPos;

uniform vec3 u_lightPos;
uniform float u_far;

void main()
{
    float dist = length(v_worldPos - u_lightPos);
    dist = dist / u_far;         
    gl_FragDepth = dist;
}