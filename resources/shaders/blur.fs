#version 330 core
out vec4 FragColor;

in vec2 texCoord;

uniform sampler2D inputTexture;
uniform vec2 viewportSize;

void main()
{
    vec2 texelSize = 1.0 / viewportSize;
    
    // Gaussian blur kernel weights (9-tap)
    float weights[9] = float[](
        0.0625, 0.125, 0.0625,
        0.125,  0.25,  0.125,
        0.0625, 0.125, 0.0625
    );
    
    vec4 result = vec4(0.0);
    
    // Apply 3x3 Gaussian blur
    int index = 0;
    for (int y = -1; y <= 1; y++) {
        for (int x = -1; x <= 1; x++) {
            vec2 offset = vec2(float(x), float(y)) * texelSize;
            result += texture(inputTexture, texCoord + offset) * weights[index];
            index++;
        }
    }
    
    FragColor = result;
}
