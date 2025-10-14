#version 330 core
out vec4 FragColor;

in vec2 texCoord;

uniform sampler2D inputTexture1;
uniform sampler2D inputTexture2;
uniform vec2 viewportSize;
uniform float zoom;
uniform int mode;

void main()
{
    vec4 color1 = texture(inputTexture1, texCoord);
    vec4 color2 = texture(inputTexture2, texCoord);

    if (mode == 0) {
        vec3 mergedColor = clamp(color1.rgb * 0.5 + color2.rgb * 0.5, 0.0, 1.0);
        FragColor = vec4(mergedColor, 1.0f);
    } else if (mode == 1) {
        float brightness = clamp(dot(color2.rgb, vec3(1.0)), 0.0, 1.0);
        vec3 finalColor = mix(color1.rgb * 0.7, color2.rgb, brightness);
        FragColor = vec4(finalColor, 1.0f);
        //FragColor = vec4(color1.rgb * 0.4 + color2.rgb * 0.6, 1.0f);
    } else {
        FragColor = vec4(0.0);
    }


    // Calculate brightness of merged result
    //float brightness = clamp(dot(mergedColor, vec3(1.0)), 0.0, 1.0);
    
    // Use brightness as alpha to blend with screen background
    //vec3 finalColor = mix(screenBackground.rgb, mergedColor, brightness);

    //FragColor = vec4(finalColor, 1.0f);
}
