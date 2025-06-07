#version 450

layout(location = 0, index = 0) out vec4 outColor;
layout(location = 0, index = 1) out vec4 outBlend;

layout(location = 0) in RectData {
    vec4 bg_color;
} rect;

void main() {
    outColor = rect.bg_color;
    outBlend = vec4(vec3(rect.bg_color.a), 1.0);
}
