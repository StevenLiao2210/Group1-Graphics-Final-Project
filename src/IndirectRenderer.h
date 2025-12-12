#pragma once

#include <glad/glad.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <vector>
#include <string>
#include "MyPoissonSample.h"

// Structure for indirect draw command
struct DrawElementsIndirectCommand {
    GLuint count;          // Number of indices
    GLuint instanceCount;  // Number of instances (will be filled by compute shader)
    GLuint firstIndex;     // Offset in index buffer
    GLuint baseVertex;     // Offset in vertex buffer
    GLuint baseInstance;   // Base instance for instanced arrays
};

struct MeshChunk {
    GLuint instanceOffset;
    GLuint instanceCount;
    glm::vec3 boundsCenter;
    float boundsRadius;
};

// Structure for instance data stored in SSBO (packed)
struct InstanceData {
    glm::vec4 positionScale;    // xyz = translation, w = uniform scale
    glm::vec4 rotationSinCos;   // x = sin(yaw), y = cos(yaw)
    glm::vec4 boundingSphere;   // xyz = center, w = radius
};

// Mesh information for multi-draw indirect
struct MeshInfo {
    std::string name;
    GLuint vao;
    GLuint vertexBuffer;
    GLuint indexBuffer;
    GLuint numIndices;
    GLuint numVertices;
    GLuint albedoTexture;
    GLuint normalTexture;
    glm::vec3 boundingSphereCenter;
    float boundingSphereRadius;
    bool hasAlpha;
    GLuint instanceOffset;    // Offset in the global instance buffer
    GLuint instanceCount;     // Number of instances for this mesh
    GLuint visibleIndexOffset; // Offset in the visible indices buffer for this mesh
    std::vector<MeshChunk> chunks;
};

class IndirectRenderer {
public:
    IndirectRenderer();
    ~IndirectRenderer();

    bool initialize();

    // Load mesh from OBJ file
    int loadMesh(const std::string& objPath, const std::string& albedoTexPath, 
                 const std::string& normalTexPath = "", bool hasAlpha = false);

    // Set instances from Poisson sample data
    void setInstances(int meshIndex, MyPoissonSample* sample, float baseY, 
                      float scaleMin = 1.0f, float scaleMax = 1.0f,
                      const class MyTerrainData* terrainData = nullptr);

    // Manually add instances with model matrices
    void addInstance(int meshIndex, const glm::mat4& modelMatrix);

    // Finalize instance data (upload to GPU)
    void finalizeInstances();

    // Prepare GPU state (compute culling) for a camera
    void prepareCulling(const glm::mat4& viewMatrix, const glm::mat4& projMatrix,
                        const glm::vec3& cameraPos, bool enableCulling = true,
                        bool updateStats = true);

    // Draw using the already prepared visible instance data
    void drawPrepared(const glm::mat4& viewMatrix, const glm::mat4& projMatrix,
                      int filterMode = 0, bool useNormalMapping = true);

    // Convenience render that does both preparing and drawing
    void render(const glm::mat4& viewMatrix, const glm::mat4& projMatrix, 
                const glm::vec3& cameraPos, int filterMode = 0, 
                bool useNormalMapping = true, bool enableCulling = true,
                bool updateStats = true);

    void setCullingThrottle(float distanceThreshold, int frameInterval);
    void setStatsCollectionEnabled(bool enabled, int sampleInterval = 60);
    void setMaxViewDistance(float distanceMeters);

    // Get total visible instance count (for debugging)
    int getVisibleInstanceCount() const { return m_visibleInstanceCount; }
    int getTotalInstanceCount() const { return m_totalInstanceCount; }

private:
    // Compute frustum planes from view-projection matrix
    void extractFrustumPlanes(const glm::mat4& viewProj, glm::vec4* planes);

    // Create compute shader for culling
    bool createComputeShader();
    
    // Create render shader program
    bool createRenderShader();

    // Load texture from file
    GLuint loadTexture(const std::string& path);

    // Calculate bounding sphere from vertices
    void calculateBoundingSphere(const std::vector<float>& positions, 
                                  glm::vec3& center, float& radius);

    // Read GPU draw command buffer to keep debug counters accurate
    void updateVisibleInstanceStats();

private:
    // Compute shader for culling
    GLuint m_cullComputeProgram;
    
    // Compute shader for resetting draw commands
    GLuint m_resetComputeProgram;
    
    // Render shader program
    GLuint m_renderProgram;

    // SSBOs
    GLuint m_instanceDataSSBO;       // All instance data
    GLuint m_visibleIndicesSSBO;     // Visible instance indices (per mesh)
    GLuint m_drawCommandSSBO;        // All draw commands in one buffer
    GLuint m_frustumUBO;             // Uniform buffer for frustum planes

    // Mesh data
    std::vector<MeshInfo> m_meshes;

    // All instances across all meshes
    std::vector<InstanceData> m_allInstances;

    // Draw commands
    std::vector<DrawElementsIndirectCommand> m_drawCommands;

    // Statistics
    int m_visibleInstanceCount;
    int m_totalInstanceCount;

    // Is finalized
    bool m_finalized;

    // Culling throttle parameters
    glm::vec3 m_lastCullCameraPos;
    int m_framesSinceLastCull;
    float m_recullDistance;
    int m_recullFrameInterval;
    bool m_hasCulledOnce;

    // Stats collection control
    bool m_collectStats;
    int m_statSampleInterval;
    int m_statSampleCounter;

    float m_maxViewDistance;
};
