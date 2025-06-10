#version 450

struct Rect {
    uvec2 pos;
    uvec2 size;
    vec4 bg_color;
    uint parent_idx;
};

layout(set = 0, binding = 0) readonly buffer Rects {
    Rect data[];
} rects;

layout(location = 0) out RectData {
    vec4 bg_color;
} out_rect;

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
    uvec2 offset = uvec2(0);

    Rect rect = rects.data[gl_InstanceIndex];

    uint parent_idx = rect.parent_idx;
    while (parent_idx != 0) {
        Rect parent = rects.data[parent_idx - 1];
        offset += parent.pos;
        parent_idx = parent.parent_idx;
    }

    vec2 in_rect = positions[gl_VertexIndex] * rect.size;

    out_rect.bg_color = rect.bg_color;
    
    gl_Position = vec4((in_rect + rect.pos + offset) * 2.0 / constants.screen_size - vec2(1.0), 0.0, 1.0);
}
