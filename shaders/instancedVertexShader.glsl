#version 430 core

layout(location = 0) in vec3 v_vertex;
layout(location = 1) in vec3 v_normal;
layout(location = 2) in vec3 v_uv;

out vec3 f_viewVertex;
out vec3 f_uv;
out vec3 f_viewNormal;
out vec3 f_worldVertex;
out vec3 f_worldNormal;
flat out int f_objectType;

layout(location = 7) uniform mat4 viewMat;
layout(location = 8) uniform mat4 projMat;
layout(location = 13) uniform int visibleIndexOffset;  // Offset into visible indices buffer

// Instance data structure - must match CPU side
struct InstanceData {
    vec4 positionScale;
    vec4 rotationSinCos;
    vec4 boundingSphere;
};

// SSBO for all instance data
layout(std430, binding = 4) readonly buffer InstanceDataBuffer {
    InstanceData instances[];
};

// SSBO for visible instance indices
layout(std430, binding = 5) readonly buffer VisibleIndicesBuffer {
    uint visibleIndices[];
};

void main() {
    // Get the actual instance index from the visible indices buffer
    // Use visibleIndexOffset to find the correct position in the buffer
    uint visibleIdx = visibleIndices[visibleIndexOffset + gl_InstanceID];
    InstanceData instance = instances[visibleIdx];
    vec3 instancePos = instance.positionScale.xyz;
    float instanceScale = instance.positionScale.w;
    float sinY = instance.rotationSinCos.x;
    float cosY = instance.rotationSinCos.y;
    
    vec3 rotatedVertex = vec3(
        cosY * v_vertex.x + sinY * v_vertex.z,
        v_vertex.y,
        -sinY * v_vertex.x + cosY * v_vertex.z
    );
    vec3 scaledVertex = rotatedVertex * instanceScale;
    vec4 worldVertex = vec4(instancePos + scaledVertex, 1.0);
    
    vec3 rotatedNormal = vec3(
        cosY * v_normal.x + sinY * v_normal.z,
        v_normal.y,
        -sinY * v_normal.x + cosY * v_normal.z
    );
    
    // Pass to fragment shader
    vec3 worldNormalVec = normalize(rotatedNormal);
    f_worldVertex = worldVertex.xyz;
    f_worldNormal = worldNormalVec;
    f_objectType = 2;  // Instanced foliage/building type
    
    // View-space transformation
    vec4 viewVertex = viewMat * worldVertex;
    vec4 worldNormal = vec4(worldNormalVec, 0.0);
    vec4 viewNormal = viewMat * worldNormal;
    
    f_viewVertex = viewVertex.xyz;
    f_viewNormal = normalize(viewNormal.xyz);
    f_uv = v_uv;
    
    gl_Position = projMat * viewVertex;
}
