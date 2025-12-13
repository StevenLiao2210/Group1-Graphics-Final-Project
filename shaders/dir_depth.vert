#version 410 core
layout(location = 0) in vec3 a_pos;
uniform mat4 u_model;
uniform mat4 u_lightVP;
void main()
{
    gl_Position = u_lightVP * u_model * vec4(a_pos, 1.0);
}
