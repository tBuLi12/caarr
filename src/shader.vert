#version 450

layout(location = 0) in uvec2 pos;
layout(location = 1) in uvec2 size;
layout(location = 2) in vec4 bg_color;

layout(location = 0) out RectData {
    vec4 bg_color;
} rect;

layout(push_constant) uniform PushConstants {
    vec2 screen_size;
} constants;

vec2 positions[6] = vec2[](
    vec2(0.0, 0.0),
    vec2(1.0, 1.0),
    vec2(1.0, 0.0),
    vec2(0.0, 0.0),
    vec2(0.0, 1.0),
    vec2(1.0, 1.0)
);

void main() {
    vec2 in_rect = positions[gl_VertexIndex] * size;

    rect.bg_color = bg_color;
    
    gl_Position = vec4((in_rect + pos) * 2.0 / constants.screen_size - vec2(1.0), 0.0, 1.0);
}
