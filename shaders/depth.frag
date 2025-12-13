#version 410 core
in vec3 v_worldPos;

uniform vec3  u_lightPos;
uniform float u_far;

void main()
{
    // store radial distance / far in the cube depth
    float dist = length(v_worldPos - u_lightPos);
    dist = dist / u_far;          // 0..1
    gl_FragDepth = dist;
}