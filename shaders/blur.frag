#version 410 core
in vec2 v_uv;
out vec4 FragColor;

uniform sampler2D u_image;
uniform bool u_horizontal;

void main()
{
    float weight[5] = float[](0.227027, 0.1945946, 0.1216216, 0.054054, 0.016216);
    vec2 texelSize = 1.0 / vec2(textureSize(u_image, 0));

    vec3 result = texture(u_image, v_uv).rgb * weight[0];
    for (int i = 1; i < 5; ++i) {
        vec2 offset = u_horizontal
            ? vec2(texelSize.x * float(i), 0.0)
            : vec2(0.0, texelSize.y * float(i));

        result += texture(u_image, v_uv + offset).rgb * weight[i];
        result += texture(u_image, v_uv - offset).rgb * weight[i];
    }
    FragColor = vec4(result, 1.0);
}