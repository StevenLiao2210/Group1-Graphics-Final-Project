#version 430 core

layout(local_size_x = 256) in;

// Instance data structure
struct InstanceData {
    mat4 modelMatrix;
    vec4 boundingSphere;  // xyz = center in local space, w = radius
};

// Frustum planes and camera data
layout(std140, binding = 0) uniform FrustumData {
    vec4 frustumPlanes[6];
    mat4 viewMatrix;
    mat4 projMatrix;
    uint totalInstances;
    uint meshInstanceOffset;
    uint meshInstanceCount;
    uint padding;
};

// Input: all instance data
layout(std430, binding = 1) readonly buffer InstanceDataBuffer {
    InstanceData instances[];
};

// Output: visible instance indices
layout(std430, binding = 2) buffer VisibleIndicesBuffer {
    uint visibleIndices[];
};

// Atomic counter for visible instances
layout(std430, binding = 3) buffer AtomicCounter {
    uint visibleCount;
};

// Check if sphere is inside or intersects frustum
// Returns true if visible (inside or intersecting)
bool sphereInFrustum(vec3 center, float radius) {
    for (int i = 0; i < 6; i++) {
        // Signed distance from center to plane
        float distance = dot(frustumPlanes[i].xyz, center) + frustumPlanes[i].w;
        // If sphere is completely behind any plane, it's outside
        if (distance < -radius) {
            return false;
        }
    }
    return true;
}

void main() {
    uint globalId = gl_GlobalInvocationID.x;
    
    // Check bounds
    if (globalId >= meshInstanceCount) {
        return;
    }
    
    // Calculate actual instance index
    uint instanceId = meshInstanceOffset + globalId;
    
    // Get instance data
    InstanceData instance = instances[instanceId];
    
    // Transform bounding sphere center to world space
    vec3 localCenter = instance.boundingSphere.xyz;
    vec4 worldCenterH = instance.modelMatrix * vec4(localCenter, 1.0);
    vec3 worldCenter = worldCenterH.xyz;
    
    // Calculate world-space radius by considering non-uniform scale
    // Extract scale from model matrix columns
    vec3 scale;
    scale.x = length(instance.modelMatrix[0].xyz);
    scale.y = length(instance.modelMatrix[1].xyz);
    scale.z = length(instance.modelMatrix[2].xyz);
    float maxScale = max(max(scale.x, scale.y), scale.z);
    float worldRadius = instance.boundingSphere.w * maxScale;
    
    // Perform frustum culling using bounding sphere
    if (sphereInFrustum(worldCenter, worldRadius)) {
        // Instance is visible - add to visible list
        uint index = atomicAdd(visibleCount, 1);
        visibleIndices[index] = instanceId;
    }
}
