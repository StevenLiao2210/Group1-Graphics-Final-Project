#include "IndirectRenderer.h"
#include "SceneManager.h"
#include "terrain/MyTerrainData.h"
#include "tiny_obj_loader.h"
#include "stb_image.h"

#include <iostream>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>

// Compute shader source for frustum culling - writes directly to draw command buffer
static const char* CULL_COMPUTE_SHADER_SOURCE = R"(
#version 430 core

layout(local_size_x = 64) in;

// Instance data structure
struct InstanceData {
    vec4 positionScale;    // xyz = translation, w = scale
    vec4 rotationSinCos;   // x = sin(yaw), y = cos(yaw)
    vec4 boundingSphere;   // xyz = center, w = radius
};

// Draw command structure
struct DrawCommand {
    uint count;
    uint instanceCount;
    uint firstIndex;
    uint baseVertex;
    uint baseInstance;
};

// Frustum planes - shared across all mesh dispatches
layout(std140, binding = 0) uniform FrustumData {
    vec4 frustumPlanes[6];
};

// Per-mesh uniforms (faster than UBO updates)
layout(location = 1) uniform uint meshInstanceOffset;
layout(location = 2) uniform uint meshInstanceCount;
layout(location = 3) uniform uint visibleIndexOffset;
layout(location = 4) uniform uint drawCommandIndex;
layout(location = 5) uniform vec3 cameraPosition;
layout(location = 6) uniform float maxViewDistance;

// Input: all instance data
layout(std430, binding = 1) readonly buffer InstanceDataBuffer {
    InstanceData instances[];
};

// Output: visible instance indices
layout(std430, binding = 2) buffer VisibleIndicesBuffer {
    uint visibleIndices[];
};

// Draw commands array - one per mesh
layout(std430, binding = 3) buffer DrawCommandBuffer {
    DrawCommand drawCmds[];
};

// Check if sphere is inside or intersects frustum
bool sphereInFrustum(vec3 center, float radius) {
    for (int i = 0; i < 6; i++) {
        float distance = dot(frustumPlanes[i].xyz, center) + frustumPlanes[i].w;
        if (distance < -radius) {
            return false;
        }
    }
    return true;
}

void main() {
    uint globalId = gl_GlobalInvocationID.x;
    
    if (globalId >= meshInstanceCount) {
        return;
    }
    
    uint instanceId = meshInstanceOffset + globalId;
    
    // Get instance data
    InstanceData instance = instances[instanceId];
    vec3 position = instance.positionScale.xyz;
    float uniformScale = instance.positionScale.w;
    vec3 localCenter = instance.boundingSphere.xyz;
    float sinY = instance.rotationSinCos.x;
    float cosY = instance.rotationSinCos.y;
    
    // Rotate local center around Y
    vec3 rotatedCenter = vec3(
        cosY * localCenter.x + sinY * localCenter.z,
        localCenter.y,
        -sinY * localCenter.x + cosY * localCenter.z
    );
    
    vec3 worldCenter = position + rotatedCenter * uniformScale;
    float worldRadius = instance.boundingSphere.w * uniformScale;
    
    // Distance-based culling
    if (maxViewDistance > 0.0) {
        float distToCamera = distance(worldCenter, cameraPosition);
        if (distToCamera - worldRadius > maxViewDistance) {
            return;
        }
    }
    
    // Frustum culling
    if (sphereInFrustum(worldCenter, worldRadius)) {
        // Add to visible list - atomically increment instance count in draw command
        uint index = atomicAdd(drawCmds[drawCommandIndex].instanceCount, 1);
        visibleIndices[visibleIndexOffset + index] = instanceId;
    }
}
)";

// Simple compute shader to reset all draw command instanceCounts to 0
static const char* RESET_COMPUTE_SHADER_SOURCE = R"(
#version 430 core

layout(local_size_x = 64) in;

struct DrawCommand {
    uint count;
    uint instanceCount;
    uint firstIndex;
    uint baseVertex;
    uint baseInstance;
};

layout(std430, binding = 0) buffer DrawCommandBuffer {
    DrawCommand drawCmds[];
};

layout(location = 0) uniform uint numMeshes;

void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx < numMeshes) {
        drawCmds[idx].instanceCount = 0;
    }
}
)";


IndirectRenderer::IndirectRenderer()
    : m_cullComputeProgram(0)
    , m_resetComputeProgram(0)
    , m_renderProgram(0)
    , m_instanceDataSSBO(0)
    , m_visibleIndicesSSBO(0)
    , m_drawCommandSSBO(0)
    , m_frustumUBO(0)
    , m_visibleInstanceCount(0)
    , m_totalInstanceCount(0)
    , m_finalized(false)
    , m_lastCullCameraPos(0.0f)
    , m_framesSinceLastCull(0)
    , m_recullDistance(5.0f)
    , m_recullFrameInterval(2)
    , m_hasCulledOnce(false)
    , m_collectStats(false)
    , m_statSampleInterval(60)
    , m_statSampleCounter(0)
    , m_maxViewDistance(250.0f)
{
}

IndirectRenderer::~IndirectRenderer() {
    if (m_cullComputeProgram) glDeleteProgram(m_cullComputeProgram);
    if (m_resetComputeProgram) glDeleteProgram(m_resetComputeProgram);
    if (m_renderProgram) glDeleteProgram(m_renderProgram);
    if (m_instanceDataSSBO) glDeleteBuffers(1, &m_instanceDataSSBO);
    if (m_visibleIndicesSSBO) glDeleteBuffers(1, &m_visibleIndicesSSBO);
    if (m_drawCommandSSBO) glDeleteBuffers(1, &m_drawCommandSSBO);
    if (m_frustumUBO) glDeleteBuffers(1, &m_frustumUBO);
    
    for (auto& mesh : m_meshes) {
        if (mesh.vao) glDeleteVertexArrays(1, &mesh.vao);
        if (mesh.vertexBuffer) glDeleteBuffers(1, &mesh.vertexBuffer);
        if (mesh.indexBuffer) glDeleteBuffers(1, &mesh.indexBuffer);
    }
}

// Helper to load shader source from file
static std::string loadShaderSource(const std::string& filepath) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        std::cerr << "Failed to open shader file: " << filepath << std::endl;
        return "";
    }
    std::stringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}

bool IndirectRenderer::initialize() {
    if (!createComputeShader()) return false;
    if (!createRenderShader()) return false;
    
    // Create frustum UBO (6 vec4 planes = 96 bytes, round up to 128)
    glGenBuffers(1, &m_frustumUBO);
    glBindBuffer(GL_UNIFORM_BUFFER, m_frustumUBO);
    glBufferData(GL_UNIFORM_BUFFER, 128, nullptr, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_UNIFORM_BUFFER, 0);
    
    return true;
}

bool IndirectRenderer::createRenderShader() {
    // Load vertex shader
    std::string vsSource = loadShaderSource("shaders/instancedVertexShader.glsl");
    if (vsSource.empty()) {
        std::cerr << "Failed to load instanced vertex shader" << std::endl;
        return false;
    }
    
    // Load fragment shader
    std::string fsSource = loadShaderSource("shaders/instancedFragmentShader.glsl");
    if (fsSource.empty()) {
        std::cerr << "Failed to load instanced fragment shader" << std::endl;
        return false;
    }
    
    // Compile vertex shader
    GLuint vsShader = glCreateShader(GL_VERTEX_SHADER);
    const char* vsSrc = vsSource.c_str();
    glShaderSource(vsShader, 1, &vsSrc, nullptr);
    glCompileShader(vsShader);
    
    GLint success;
    glGetShaderiv(vsShader, GL_COMPILE_STATUS, &success);
    if (!success) {
        char infoLog[1024];
        glGetShaderInfoLog(vsShader, 1024, nullptr, infoLog);
        std::cerr << "Instanced vertex shader compilation failed: " << infoLog << std::endl;
        return false;
    }
    
    // Compile fragment shader
    GLuint fsShader = glCreateShader(GL_FRAGMENT_SHADER);
    const char* fsSrc = fsSource.c_str();
    glShaderSource(fsShader, 1, &fsSrc, nullptr);
    glCompileShader(fsShader);
    
    glGetShaderiv(fsShader, GL_COMPILE_STATUS, &success);
    if (!success) {
        char infoLog[1024];
        glGetShaderInfoLog(fsShader, 1024, nullptr, infoLog);
        std::cerr << "Instanced fragment shader compilation failed: " << infoLog << std::endl;
        return false;
    }
    
    // Link program
    m_renderProgram = glCreateProgram();
    glAttachShader(m_renderProgram, vsShader);
    glAttachShader(m_renderProgram, fsShader);
    glLinkProgram(m_renderProgram);
    
    glGetProgramiv(m_renderProgram, GL_LINK_STATUS, &success);
    if (!success) {
        char infoLog[1024];
        glGetProgramInfoLog(m_renderProgram, 1024, nullptr, infoLog);
        std::cerr << "Instanced shader program linking failed: " << infoLog << std::endl;
        return false;
    }
    
    glDeleteShader(vsShader);
    glDeleteShader(fsShader);
    
    std::cout << "[IndirectRenderer] Render shader program created successfully" << std::endl;
    return true;
}

bool IndirectRenderer::createComputeShader() {
    GLint success;
    char infoLog[512];
    
    // Create cull compute shader
    GLuint cullShader = glCreateShader(GL_COMPUTE_SHADER);
    glShaderSource(cullShader, 1, &CULL_COMPUTE_SHADER_SOURCE, nullptr);
    glCompileShader(cullShader);
    
    glGetShaderiv(cullShader, GL_COMPILE_STATUS, &success);
    if (!success) {
        glGetShaderInfoLog(cullShader, 512, nullptr, infoLog);
        std::cerr << "Cull compute shader compilation failed: " << infoLog << std::endl;
        return false;
    }
    
    m_cullComputeProgram = glCreateProgram();
    glAttachShader(m_cullComputeProgram, cullShader);
    glLinkProgram(m_cullComputeProgram);
    
    glGetProgramiv(m_cullComputeProgram, GL_LINK_STATUS, &success);
    if (!success) {
        glGetProgramInfoLog(m_cullComputeProgram, 512, nullptr, infoLog);
        std::cerr << "Cull compute shader linking failed: " << infoLog << std::endl;
        return false;
    }
    glDeleteShader(cullShader);
    
    // Create reset compute shader
    GLuint resetShader = glCreateShader(GL_COMPUTE_SHADER);
    glShaderSource(resetShader, 1, &RESET_COMPUTE_SHADER_SOURCE, nullptr);
    glCompileShader(resetShader);
    
    glGetShaderiv(resetShader, GL_COMPILE_STATUS, &success);
    if (!success) {
        glGetShaderInfoLog(resetShader, 512, nullptr, infoLog);
        std::cerr << "Reset compute shader compilation failed: " << infoLog << std::endl;
        return false;
    }
    
    m_resetComputeProgram = glCreateProgram();
    glAttachShader(m_resetComputeProgram, resetShader);
    glLinkProgram(m_resetComputeProgram);
    
    glGetProgramiv(m_resetComputeProgram, GL_LINK_STATUS, &success);
    if (!success) {
        glGetProgramInfoLog(m_resetComputeProgram, 512, nullptr, infoLog);
        std::cerr << "Reset compute shader linking failed: " << infoLog << std::endl;
        return false;
    }
    glDeleteShader(resetShader);
    
    return true;
}

GLuint IndirectRenderer::loadTexture(const std::string& path) {
    if (path.empty()) return 0;
    
    int w, h, channels;
    stbi_set_flip_vertically_on_load(true);
    unsigned char* data = stbi_load(path.c_str(), &w, &h, &channels, 0);
    
    if (!data) {
        std::cerr << "[IndirectRenderer] Failed to load texture: " << path << std::endl;
        return 0;
    }
    
    GLenum format = GL_RGB;
    if (channels == 1) format = GL_RED;
    else if (channels == 3) format = GL_RGB;
    else if (channels == 4) format = GL_RGBA;
    
    GLuint tex = 0;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    
    glTexImage2D(GL_TEXTURE_2D, 0, 
                 (format == GL_RGBA ? GL_RGBA8 : GL_RGB8),
                 w, h, 0, format, GL_UNSIGNED_BYTE, data);
    
    glGenerateMipmap(GL_TEXTURE_2D);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    
    glBindTexture(GL_TEXTURE_2D, 0);
    stbi_image_free(data);
    
    return tex;
}

void IndirectRenderer::calculateBoundingSphere(const std::vector<float>& positions,
                                                glm::vec3& center, float& radius) {
    if (positions.empty()) {
        center = glm::vec3(0.0f);
        radius = 1.0f;
        return;
    }
    
    // Calculate center (average of all vertices)
    center = glm::vec3(0.0f);
    int numVerts = positions.size() / 3;
    for (int i = 0; i < numVerts; i++) {
        center.x += positions[i * 3 + 0];
        center.y += positions[i * 3 + 1];
        center.z += positions[i * 3 + 2];
    }
    center /= static_cast<float>(numVerts);
    
    // Calculate radius (max distance from center)
    radius = 0.0f;
    for (int i = 0; i < numVerts; i++) {
        glm::vec3 v(positions[i * 3 + 0], positions[i * 3 + 1], positions[i * 3 + 2]);
        float dist = glm::length(v - center);
        if (dist > radius) radius = dist;
    }
}

int IndirectRenderer::loadMesh(const std::string& objPath, const std::string& albedoTexPath,
                                const std::string& normalTexPath, bool hasAlpha) {
    tinyobj::attrib_t attrib;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> materials;
    std::string warn, err;
    
    bool ok = tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, objPath.c_str());
    if (!ok) {
        std::cerr << "[IndirectRenderer] Failed to load: " << objPath << std::endl;
        return -1;
    }
    
    std::vector<float> positions;
    std::vector<float> normals;
    std::vector<float> texcoords;
    
    for (const auto& shape : shapes) {
        size_t indexOffset = 0;
        for (size_t f = 0; f < shape.mesh.num_face_vertices.size(); ++f) {
            int fv = shape.mesh.num_face_vertices[f];
            for (int v = 0; v < fv; ++v) {
                tinyobj::index_t idx = shape.mesh.indices[indexOffset + v];
                
                positions.push_back(attrib.vertices[3 * idx.vertex_index + 0]);
                positions.push_back(attrib.vertices[3 * idx.vertex_index + 1]);
                positions.push_back(attrib.vertices[3 * idx.vertex_index + 2]);
                
                if (!attrib.normals.empty() && idx.normal_index >= 0) {
                    normals.push_back(attrib.normals[3 * idx.normal_index + 0]);
                    normals.push_back(attrib.normals[3 * idx.normal_index + 1]);
                    normals.push_back(attrib.normals[3 * idx.normal_index + 2]);
                }
                
                if (!attrib.texcoords.empty() && idx.texcoord_index >= 0) {
                    texcoords.push_back(attrib.texcoords[2 * idx.texcoord_index + 0]);
                    texcoords.push_back(attrib.texcoords[2 * idx.texcoord_index + 1]);
                }
            }
            indexOffset += fv;
        }
    }
    
    int numVerts = static_cast<int>(positions.size() / 3);
    if (numVerts == 0) return -1;
    
    if (static_cast<int>(normals.size()) != numVerts * 3)
        normals.assign(numVerts * 3, 0.0f);
    if (static_cast<int>(texcoords.size()) != numVerts * 2)
        texcoords.assign(numVerts * 2, 0.0f);
    
    // Create interleaved vertex data: pos(3) + normal(3) + uv(3)
    const int stride = 9;
    std::vector<float> vertexData(numVerts * stride);
    for (int i = 0; i < numVerts; i++) {
        vertexData[i * stride + 0] = positions[i * 3 + 0];
        vertexData[i * stride + 1] = positions[i * 3 + 1];
        vertexData[i * stride + 2] = positions[i * 3 + 2];
        vertexData[i * stride + 3] = normals[i * 3 + 0];
        vertexData[i * stride + 4] = normals[i * 3 + 1];
        vertexData[i * stride + 5] = normals[i * 3 + 2];
        vertexData[i * stride + 6] = texcoords[i * 2 + 0];
        vertexData[i * stride + 7] = texcoords[i * 2 + 1];
        vertexData[i * stride + 8] = 0.0f;
    }
    
    // Create index buffer (simple 0..N-1)
    std::vector<unsigned int> indices(numVerts);
    for (int i = 0; i < numVerts; i++) indices[i] = i;
    
    // Calculate bounding sphere
    glm::vec3 bsCenter;
    float bsRadius;
    calculateBoundingSphere(positions, bsCenter, bsRadius);
    
    // Create OpenGL objects
    MeshInfo mesh;
    mesh.name = objPath;
    mesh.numVertices = numVerts;
    mesh.numIndices = numVerts;
    mesh.boundingSphereCenter = bsCenter;
    mesh.boundingSphereRadius = bsRadius;
    mesh.hasAlpha = hasAlpha;
    mesh.instanceOffset = 0;
    mesh.instanceCount = 0;
    mesh.visibleIndexOffset = 0;
    
    // Create VAO
    glGenVertexArrays(1, &mesh.vao);
    glBindVertexArray(mesh.vao);
    
    // Create vertex buffer
    glGenBuffers(1, &mesh.vertexBuffer);
    glBindBuffer(GL_ARRAY_BUFFER, mesh.vertexBuffer);
    glBufferData(GL_ARRAY_BUFFER, vertexData.size() * sizeof(float), vertexData.data(), GL_STATIC_DRAW);
    
    // Set up vertex attributes
    // Position (location 0)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    // Normal (location 1)
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, stride * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);
    // UV (location 2)
    glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, stride * sizeof(float), (void*)(6 * sizeof(float)));
    glEnableVertexAttribArray(2);
    
    // Create index buffer
    glGenBuffers(1, &mesh.indexBuffer);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, mesh.indexBuffer);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(unsigned int), indices.data(), GL_STATIC_DRAW);
    
    glBindVertexArray(0);
    
    // Load textures
    mesh.albedoTexture = loadTexture(albedoTexPath);
    mesh.normalTexture = loadTexture(normalTexPath);
    
    int meshIndex = static_cast<int>(m_meshes.size());
    m_meshes.push_back(mesh);
    
    std::cout << "[IndirectRenderer] Loaded mesh: " << objPath 
              << " (verts: " << numVerts << ", bsRadius: " << bsRadius << ")" << std::endl;
    
    return meshIndex;
}

void IndirectRenderer::setInstances(int meshIndex, MyPoissonSample* sample, float baseY,
                                     float scaleMin, float scaleMax, const MyTerrainData* terrainData) {
    if (meshIndex < 0 || meshIndex >= static_cast<int>(m_meshes.size())) return;
    if (!sample) return;
    
    MeshInfo& mesh = m_meshes[meshIndex];
    mesh.instanceOffset = static_cast<int>(m_allInstances.size());
    mesh.instanceCount = sample->m_numSample;
    
    for (int i = 0; i < sample->m_numSample; i++) {
        float x = sample->m_positions[i * 3 + 0];
        float y = sample->m_positions[i * 3 + 1];
        float z = sample->m_positions[i * 3 + 2];
        
        // Get terrain height if available
        if (terrainData) {
            y = terrainData->height(x, z);
        } else {
            y = baseY;
        }
        
        float ry = sample->m_radians[i * 3 + 1];
        
        // Random scale
        float scale = scaleMin + (scaleMax - scaleMin) * (static_cast<float>(rand()) / RAND_MAX);
        
        float sinY = static_cast<float>(std::sin(ry));
        float cosY = static_cast<float>(std::cos(ry));
        
        InstanceData instance;
        instance.positionScale = glm::vec4(x, y, z, scale);
        instance.rotationSinCos = glm::vec4(sinY, cosY, 0.0f, 0.0f);
        instance.boundingSphere = glm::vec4(mesh.boundingSphereCenter, mesh.boundingSphereRadius);
        
        m_allInstances.push_back(instance);
    }
    
    m_totalInstanceCount = static_cast<int>(m_allInstances.size());
    std::cout << "[IndirectRenderer] Set " << sample->m_numSample << " instances for mesh " << meshIndex << std::endl;
}

void IndirectRenderer::addInstance(int meshIndex, const glm::mat4& modelMatrix) {
    if (meshIndex < 0 || meshIndex >= static_cast<int>(m_meshes.size())) return;
    
    MeshInfo& mesh = m_meshes[meshIndex];
    if (mesh.instanceCount == 0) {
        mesh.instanceOffset = static_cast<int>(m_allInstances.size());
    }
    mesh.instanceCount++;
    
    glm::vec3 translation = glm::vec3(modelMatrix[3]);
    float scale = glm::length(glm::vec3(modelMatrix[0]));
    float yaw = std::atan2(modelMatrix[0][2], modelMatrix[0][0]);
    InstanceData instance;
    instance.positionScale = glm::vec4(translation, scale);
    instance.rotationSinCos = glm::vec4(static_cast<float>(std::sin(yaw)),
                                        static_cast<float>(std::cos(yaw)),
                                        0.0f, 0.0f);
    instance.boundingSphere = glm::vec4(mesh.boundingSphereCenter, mesh.boundingSphereRadius);
    
    m_allInstances.push_back(instance);
    m_totalInstanceCount = static_cast<int>(m_allInstances.size());
}

void IndirectRenderer::finalizeInstances() {
    if (m_allInstances.empty()) {
        std::cerr << "[IndirectRenderer] No instances to finalize!" << std::endl;
        return;
    }
    
    // Create instance data SSBO
    glGenBuffers(1, &m_instanceDataSSBO);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_instanceDataSSBO);
    glBufferData(GL_SHADER_STORAGE_BUFFER, 
                 m_allInstances.size() * sizeof(InstanceData),
                 m_allInstances.data(), GL_STATIC_DRAW);
    
    // Calculate visible index offsets for each mesh
    int visibleOffset = 0;
    for (size_t i = 0; i < m_meshes.size(); i++) {
        m_meshes[i].visibleIndexOffset = visibleOffset;
        visibleOffset += m_meshes[i].instanceCount;
        m_meshes[i].chunks.clear();
    }
    
    // Create visible indices SSBO (max size = total instances)
    glGenBuffers(1, &m_visibleIndicesSSBO);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_visibleIndicesSSBO);
    glBufferData(GL_SHADER_STORAGE_BUFFER,
                 m_allInstances.size() * sizeof(GLuint),
                 nullptr, GL_DYNAMIC_DRAW);
    
    // Create a SINGLE draw command buffer containing all mesh draw commands
    std::vector<DrawElementsIndirectCommand> drawCommands(m_meshes.size());
    for (size_t i = 0; i < m_meshes.size(); i++) {
        drawCommands[i].count = m_meshes[i].numIndices;
        drawCommands[i].instanceCount = 0;  // Will be set by compute shader
        drawCommands[i].firstIndex = 0;
        drawCommands[i].baseVertex = 0;
        drawCommands[i].baseInstance = 0;
    }
    
    glGenBuffers(1, &m_drawCommandSSBO);
    glBindBuffer(GL_DRAW_INDIRECT_BUFFER, m_drawCommandSSBO);
    glBufferData(GL_DRAW_INDIRECT_BUFFER, 
                 drawCommands.size() * sizeof(DrawElementsIndirectCommand), 
                 drawCommands.data(), GL_DYNAMIC_DRAW);
    m_drawCommands = drawCommands;
    
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
    glBindBuffer(GL_DRAW_INDIRECT_BUFFER, 0);
    
    // Build per-mesh chunks for coarse CPU culling
    const GLuint chunkSize = 512;
    for (auto& mesh : m_meshes) {
        mesh.chunks.clear();
        if (mesh.instanceCount == 0) continue;

        GLuint processed = 0;
        while (processed < mesh.instanceCount) {
            MeshChunk chunk;
            chunk.instanceOffset = mesh.instanceOffset + processed;
            chunk.instanceCount = std::min(chunkSize, mesh.instanceCount - processed);

            glm::vec3 center(0.0f);
            for (GLuint i = 0; i < chunk.instanceCount; ++i) {
                const InstanceData& inst = m_allInstances[chunk.instanceOffset + i];
                center += glm::vec3(inst.positionScale);
            }
            center /= static_cast<float>(chunk.instanceCount);

            float radius = 0.0f;
            for (GLuint i = 0; i < chunk.instanceCount; ++i) {
                const InstanceData& inst = m_allInstances[chunk.instanceOffset + i];
                glm::vec3 pos = glm::vec3(inst.positionScale);
                float instRadius = inst.boundingSphere.w * inst.positionScale.w;
                float dist = glm::length(pos - center) + instRadius;
                radius = std::max(radius, dist);
            }

            chunk.boundsCenter = center;
            chunk.boundsRadius = radius + 1.0f; // small safety margin
            mesh.chunks.push_back(chunk);

            processed += chunk.instanceCount;
        }
    }

    m_finalized = true;
    std::cout << "[IndirectRenderer] Finalized " << m_totalInstanceCount << " total instances across " 
              << m_meshes.size() << " meshes" << std::endl;
}

void IndirectRenderer::setCullingThrottle(float distanceThreshold, int frameInterval) {
    m_recullDistance = std::max(0.0f, distanceThreshold);
    m_recullFrameInterval = std::max(1, frameInterval);
}

void IndirectRenderer::setStatsCollectionEnabled(bool enabled, int sampleInterval) {
    m_collectStats = enabled;
    if (sampleInterval > 0) {
        m_statSampleInterval = sampleInterval;
    }
    if (!enabled) {
        m_statSampleCounter = 0;
    }
}

void IndirectRenderer::setMaxViewDistance(float distanceMeters) {
    if (distanceMeters < 0.0f) {
        m_maxViewDistance = 0.0f;
    } else {
        m_maxViewDistance = distanceMeters;
    }
}

void IndirectRenderer::updateVisibleInstanceStats() {
    if (!m_drawCommandSSBO || m_meshes.empty()) {
        m_visibleInstanceCount = 0;
        return;
    }

    if (m_drawCommands.size() != m_meshes.size()) {
        m_drawCommands.resize(m_meshes.size());
    }

    glBindBuffer(GL_DRAW_INDIRECT_BUFFER, m_drawCommandSSBO);
    glGetBufferSubData(GL_DRAW_INDIRECT_BUFFER, 0,
                       m_drawCommands.size() * sizeof(DrawElementsIndirectCommand),
                       m_drawCommands.data());
    glBindBuffer(GL_DRAW_INDIRECT_BUFFER, 0);

    int visible = 0;
    for (const auto& cmd : m_drawCommands) {
        visible += static_cast<int>(cmd.instanceCount);
    }
    m_visibleInstanceCount = visible;
}

void IndirectRenderer::extractFrustumPlanes(const glm::mat4& viewProj, glm::vec4* planes) {
    // Extract frustum planes from view-projection matrix
    // Left plane
    planes[0] = glm::vec4(
        viewProj[0][3] + viewProj[0][0],
        viewProj[1][3] + viewProj[1][0],
        viewProj[2][3] + viewProj[2][0],
        viewProj[3][3] + viewProj[3][0]
    );
    // Right plane
    planes[1] = glm::vec4(
        viewProj[0][3] - viewProj[0][0],
        viewProj[1][3] - viewProj[1][0],
        viewProj[2][3] - viewProj[2][0],
        viewProj[3][3] - viewProj[3][0]
    );
    // Bottom plane
    planes[2] = glm::vec4(
        viewProj[0][3] + viewProj[0][1],
        viewProj[1][3] + viewProj[1][1],
        viewProj[2][3] + viewProj[2][1],
        viewProj[3][3] + viewProj[3][1]
    );
    // Top plane
    planes[3] = glm::vec4(
        viewProj[0][3] - viewProj[0][1],
        viewProj[1][3] - viewProj[1][1],
        viewProj[2][3] - viewProj[2][1],
        viewProj[3][3] - viewProj[3][1]
    );
    // Near plane
    planes[4] = glm::vec4(
        viewProj[0][3] + viewProj[0][2],
        viewProj[1][3] + viewProj[1][2],
        viewProj[2][3] + viewProj[2][2],
        viewProj[3][3] + viewProj[3][2]
    );
    // Far plane
    planes[5] = glm::vec4(
        viewProj[0][3] - viewProj[0][2],
        viewProj[1][3] - viewProj[1][2],
        viewProj[2][3] - viewProj[2][2],
        viewProj[3][3] - viewProj[3][2]
    );
    
    // Normalize planes
    for (int i = 0; i < 6; i++) {
        float len = glm::length(glm::vec3(planes[i]));
        if (len > 0.0001f) {
            planes[i] /= len;
        }
    }
}

void IndirectRenderer::prepareCulling(const glm::mat4& viewMatrix, const glm::mat4& projMatrix,
                                      const glm::vec3& cameraPos, bool enableCulling,
                                      bool updateStats) {
    if (!m_finalized || m_meshes.empty()) return;

    m_visibleInstanceCount = 0;

    glm::mat4 viewProj = projMatrix * viewMatrix;
    glm::vec4 frustumPlanes[6];
    extractFrustumPlanes(viewProj, frustumPlanes);
    
    bool runCullingThisFrame = enableCulling;
    if (enableCulling) {
        if (!m_hasCulledOnce) {
            runCullingThisFrame = true;
        } else {
            float moveDist = glm::length(cameraPos - m_lastCullCameraPos);
            if (moveDist < m_recullDistance && m_framesSinceLastCull < m_recullFrameInterval) {
                runCullingThisFrame = false;
            }
        }
    } else {
        m_framesSinceLastCull = 0;
        m_hasCulledOnce = false;
    }

    if (enableCulling && runCullingThisFrame) {
        // Phase 1: Reset draw command instanceCounts to 0
        glUseProgram(m_resetComputeProgram);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, m_drawCommandSSBO);
        glUniform1ui(0, static_cast<GLuint>(m_meshes.size()));
        glDispatchCompute(1, 1, 1);
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

        // Phase 2: Run culling compute shaders for all meshes
        glUseProgram(m_cullComputeProgram);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, m_instanceDataSSBO);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, m_visibleIndicesSSBO);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, m_drawCommandSSBO);

        glBindBuffer(GL_UNIFORM_BUFFER, m_frustumUBO);
        glBufferSubData(GL_UNIFORM_BUFFER, 0, sizeof(glm::vec4) * 6, frustumPlanes);
        glBindBufferBase(GL_UNIFORM_BUFFER, 0, m_frustumUBO);

        glUniform3fv(5, 1, glm::value_ptr(cameraPos));
        glUniform1f(6, m_maxViewDistance);

        for (size_t meshIdx = 0; meshIdx < m_meshes.size(); meshIdx++) {
            MeshInfo& mesh = m_meshes[meshIdx];
            if (mesh.instanceCount == 0) continue;

            auto chunkVisible = [&](const MeshChunk& chunk) {
                if (m_maxViewDistance > 0.0f) {
                    float dist = glm::length(chunk.boundsCenter - cameraPos);
                    if (dist - chunk.boundsRadius > m_maxViewDistance) {
                        return false;
                    }
                }
                for (int planeIdx = 0; planeIdx < 6; ++planeIdx) {
                    const glm::vec4& plane = frustumPlanes[planeIdx];
                    float distance = glm::dot(glm::vec3(plane), chunk.boundsCenter) + plane.w;
                    if (distance < -chunk.boundsRadius) {
                        return false;
                    }
                }
                return true;
            };

            if (mesh.chunks.empty()) {
                glUniform1ui(1, mesh.instanceOffset);
                glUniform1ui(2, mesh.instanceCount);
                glUniform1ui(3, mesh.visibleIndexOffset);
                glUniform1ui(4, static_cast<GLuint>(meshIdx));

                GLuint numGroups = (mesh.instanceCount + 63) / 64;
                glDispatchCompute(numGroups, 1, 1);
                continue;
            }

            for (const MeshChunk& chunk : mesh.chunks) {
                if (!chunkVisible(chunk)) {
                    continue;
                }

                glUniform1ui(1, chunk.instanceOffset);
                glUniform1ui(2, chunk.instanceCount);
                glUniform1ui(3, mesh.visibleIndexOffset);
                glUniform1ui(4, static_cast<GLuint>(meshIdx));

                GLuint numGroups = (chunk.instanceCount + 63) / 64;
                glDispatchCompute(numGroups, 1, 1);
            }
        }

        m_lastCullCameraPos = cameraPos;
        m_framesSinceLastCull = 0;
        m_hasCulledOnce = true;
    } else if (enableCulling && !runCullingThisFrame) {
        ++m_framesSinceLastCull;
    } else if (!enableCulling) {
        // No culling - set all instance counts to max and fill visible indices
        for (size_t meshIdx = 0; meshIdx < m_meshes.size(); meshIdx++) {
            MeshInfo& mesh = m_meshes[meshIdx];
            if (mesh.instanceCount == 0) continue;
            
            // Update draw command
            GLsizeiptr cmdOffset = meshIdx * sizeof(DrawElementsIndirectCommand);
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_drawCommandSSBO);
            glBufferSubData(GL_SHADER_STORAGE_BUFFER, 
                           cmdOffset + offsetof(DrawElementsIndirectCommand, instanceCount), 
                           sizeof(GLuint), &mesh.instanceCount);
            
            // Fill visible indices
            std::vector<GLuint> allIndices(mesh.instanceCount);
            for (GLuint i = 0; i < mesh.instanceCount; i++) {
                allIndices[i] = mesh.instanceOffset + i;
            }
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_visibleIndicesSSBO);
            glBufferSubData(GL_SHADER_STORAGE_BUFFER, 
                           mesh.visibleIndexOffset * sizeof(GLuint),
                           mesh.instanceCount * sizeof(GLuint), allIndices.data());
        }
    }
    
    glMemoryBarrier(GL_COMMAND_BARRIER_BIT | GL_SHADER_STORAGE_BARRIER_BIT);
    if (updateStats) {
        ++m_statSampleCounter;
        bool shouldReadStats = m_collectStats && (m_statSampleCounter % m_statSampleInterval == 0);
        if (shouldReadStats) {
            if (enableCulling) {
                updateVisibleInstanceStats();
            } else {
                m_visibleInstanceCount = m_totalInstanceCount;
            }
        }
    }
}

void IndirectRenderer::drawPrepared(const glm::mat4& viewMatrix, const glm::mat4& projMatrix,
                                    int filterMode, bool useNormalMapping) {
    if (!m_finalized || m_meshes.empty()) return;
    
    glUseProgram(m_renderProgram);
    
    // Set common uniforms ONCE
    glUniformMatrix4fv(7, 1, GL_FALSE, glm::value_ptr(viewMatrix));
    glUniformMatrix4fv(8, 1, GL_FALSE, glm::value_ptr(projMatrix));
    glUniform1i(10, filterMode);
    glUniform1i(11, useNormalMapping ? 1 : 0);
    glUniform1i(4, 0);  // albedoTexture sampler
    glUniform1i(6, 2);  // normalMap sampler
    
    // Bind instance data SSBOs ONCE
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, m_instanceDataSSBO);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, m_visibleIndicesSSBO);
    
    // Bind draw command buffer ONCE
    glBindBuffer(GL_DRAW_INDIRECT_BUFFER, m_drawCommandSSBO);
    
    for (size_t meshIdx = 0; meshIdx < m_meshes.size(); meshIdx++) {
        MeshInfo& mesh = m_meshes[meshIdx];
        if (mesh.instanceCount == 0) continue;
        
        // Set per-mesh uniforms
        glUniform1i(12, mesh.hasAlpha ? 1 : 0);
        glUniform1i(13, mesh.visibleIndexOffset);
        
        // Bind mesh VAO
        glBindVertexArray(mesh.vao);
        
        // Bind textures
        if (mesh.albedoTexture) {
            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D, mesh.albedoTexture);
        }
        if (mesh.normalTexture) {
            glActiveTexture(GL_TEXTURE2);
            glBindTexture(GL_TEXTURE_2D, mesh.normalTexture);
        }
        
        // Handle alpha for foliage
        if (mesh.hasAlpha) {
            glEnable(GL_BLEND);
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
            glDisable(GL_CULL_FACE);
        }
        
        // Draw using indirect command with offset into command buffer
        GLsizeiptr cmdOffset = meshIdx * sizeof(DrawElementsIndirectCommand);
        glDrawElementsIndirect(GL_TRIANGLES, GL_UNSIGNED_INT, (void*)cmdOffset);
        
        // Restore state
        if (mesh.hasAlpha) {
            glDisable(GL_BLEND);
            glEnable(GL_CULL_FACE);
        }
    }
    
    glBindVertexArray(0);
    glBindBuffer(GL_DRAW_INDIRECT_BUFFER, 0);
}

void IndirectRenderer::render(const glm::mat4& viewMatrix, const glm::mat4& projMatrix,
                               const glm::vec3& cameraPos, int filterMode, 
                               bool useNormalMapping, bool enableCulling,
                               bool updateStats) {
    prepareCulling(viewMatrix, projMatrix, cameraPos, enableCulling, updateStats);
    drawPrepared(viewMatrix, projMatrix, filterMode, useNormalMapping);
}
