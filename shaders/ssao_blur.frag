#version 410 core
in vec2 v_uv;
out float FragColor;

uniform sampler2D u_ssaoInput;

void main()
{
    vec2 texel = 1.0 / vec2(textureSize(u_ssaoInput, 0));
    float result = 0.0;

    // simple 4x4 box blur (fast + good enough for assignment)
    for (int y = -2; y <= 1; ++y)
    for (int x = -2; x <= 1; ++x)
    {
        vec2 off = vec2(x, y) * texel;
        result += texture(u_ssaoInput, v_uv + off).r;
    }
    result /= 16.0;

    FragColor = result;
}