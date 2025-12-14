#include <glad/glad.h>
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <iostream>
#include <vector>
#include <string>
#include <random>

#include <fstream>
#include <sstream>

// ==============================
// stb_image + tinyobjloader
// ==============================
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#define TINYOBJLOADER_IMPLEMENTATION
#include <tiny_obj_loader.h>

// ==============================
// Globals
// ==============================
ImVec4 g_clearColor = ImVec4(0.1f, 0.1f, 0.12f, 1.0f);

// Camera (initial values from assignment)
glm::vec3 g_eye = glm::vec3(4.0f, 1.0f, -1.5f);
glm::vec3 g_center = glm::vec3(3.0f, 1.0f, -1.5f);
glm::vec3 g_up = glm::vec3(0.0f, 1.0f, 0.0f);
float     g_fov = 45.0f;

// Mouse orbit
bool   g_mouseRot = false;
double g_lastX = 0.0, g_lastY = 0.0;

// Shader programs
GLuint g_geomProgram = 0; // geometry (G-buffer) pass
GLuint g_lightProgram = 0; // lighting pass
GLuint g_depthProgram = 0; // shadow-map depth pass
GLuint g_blurProgram = 0; // gaussian blur for bloom
GLuint g_finalProgram = 0; // final combine pass
GLuint g_volRaymarchProgram = 0; //volumetric lighting



// ==============================
// Directional shadow mapping
// ==============================
GLuint g_dirShadowFBO = 0;
GLuint g_dirShadowTex = 0;
//const int DIR_SHADOW_SIZE = 1024;
const int DIR_SHADOW_SIZE = 4096;


glm::mat4 g_lightVP = glm::mat4(1.0f); // light view-projection for directional shadow

GLuint g_dirDepthProgram = 0;

// ==============================
// Directional light camera (NEW / separate feature)
// ==============================
glm::vec3 g_dirLightEye = glm::vec3(-2.845f, 2.028f, -1.293f);
glm::vec3 g_dirLightCenter = glm::vec3(0.542f, -0.141f, -0.422f);
glm::vec3 g_dirLightUp = glm::vec3(0.0f, 1.0f, 0.0f);

float g_dirLightNear = 0.1f;
float g_dirLightFar = 10.0f;
float g_dirLightRange = 5.0f;

glm::mat4 g_dirLightVP = glm::mat4(1.0f);

// Shadow mapping globals
GLuint g_shadowFBO = 0;
GLuint g_shadowTex = 0;
const int SHADOW_MAP_SIZE = 1024;

const float g_pointShadowNear = 0.22f;
const float g_pointShadowFar = 18.0f;

glm::mat4 g_pointShadowMatrices[6];

// point light
//glm::vec3 g_lightEye = glm::vec3(1.87659f, 0.4625f, 0.103928f);
glm::vec3 g_pointLightPos = glm::vec3(1.87659f, 0.4625f, 0.103928f);
//glm::vec3 g_pointLightPos = glm::vec3(-2.845f * 5.0f, 2.028f * 2.5f, -1.293f * 5.0f);

glm::vec3 g_lightCenter = glm::vec3(0.0f, 0.5f, 0.0f);
glm::vec3 g_lightUp = glm::vec3(0.0f, 1.0f, 0.0f);
float     g_lightNear = 0.1f;
float     g_lightFar = 10.0f;
float     g_lightRange = 5.0f;  // ortho box half-size


GLuint g_shadowTexA = 0; // point light
GLuint g_shadowTexB = 0; // demo light (or 2nd point light)


// Point light sphere (visual)
GLuint g_lightSphereVAO = 0;
GLuint g_lightSphereVBO = 0;
GLuint g_lightSphereEBO = 0;
GLsizei g_lightSphereIndexCount = 0;

const float g_lightSphereRadius = 0.22f;  // from assignment

// Rectangular area light mesh (visible emitter)
GLuint g_areaRectVAO = 0;
GLuint g_areaRectVBO = 0;
GLuint g_areaRectEBO = 0;
GLsizei g_areaRectIndexCount = 0;

// Shadow strength + toggle (debug / tuning)
float g_shadowStrength = 0.0f;  // 0 = fully black shadow, 1 = no shadow dimming
bool  g_enableShadows = true;

// Triceratops transform (editable in ImGui)
glm::vec3 g_tricePos = glm::vec3(2.05f, 0.628725f, -1.9f);
float     g_triceScale = 0.001f;
float     g_triceYaw = 0.0f;   // degrees

// First mesh index that belongs to triceratops
size_t g_triceFirstMesh = (size_t)-1;

// ==============================
// G-buffer (Deferred shading)
// ==============================
GLuint g_gbufferFBO = 0;
GLuint g_gPositionTex = 0;
GLuint g_gNormalTex = 0;
GLuint g_gAmbientTex = 0;
GLuint g_gDiffuseTex = 0;
GLuint g_gSpecularTex = 0;
GLuint g_gDepthRBO = 0;
int    g_gbufferWidth = 0;
int    g_gbufferHeight = 0;
int g_viewMode = 0;

// Fullscreen quad for lighting pass
GLuint g_quadVAO = 0;
GLuint g_quadVBO = 0;

//Normal Mapping
bool   g_enableNormalMap = true;
GLuint g_triceNormalTex = 0;

GLuint g_hdrFBO = 0;
GLuint g_hdrColorTex = 0;
GLuint g_brightColorTex = 0;
GLuint g_pingpongFBO[2] = { 0, 0 };
GLuint g_pingpongTex[2] = { 0, 0 };
int    g_bloomWidth = 0;
int    g_bloomHeight = 0;

bool   g_enableBloom = true;
float  g_bloomThreshold = 0.8f;
float  g_bloomIntensity = 0.8f;
float  g_exposure = 1.5f;
int    g_blurIterations = 10;

// ==============================
// SSAO
// ==============================
GLuint g_ssaoProgram = 0;
GLuint g_ssaoFBO = 0;
GLuint g_ssaoTex = 0;
GLuint g_ssaoNoiseTex = 0;

std::vector<glm::vec3> g_ssaoKernel;
int   g_ssaoWidth = 0;
int   g_ssaoHeight = 0;

bool  g_enableSSAO = true;

const int   SSAO_KERNEL_SIZE = 64;
float g_ssaoRadius = 0.5f;    // spec
float g_ssaoBias = 0.025f;  // spec

GLuint g_ssaoBlurProgram = 0;
GLuint g_ssaoBlurFBO = 0;
GLuint g_ssaoBlurTex = 0;


// ==============================
// Screen-Space Reflection (SSR)
// ==============================
bool  g_enableSSR = true;   // toggle
float g_ssrMaxDistance = 6.0f;   // how far the ray can travel
int   g_ssrMaxSteps = 40;     // how many steps
float g_ssrStep = 0.15f;  // distance between samples
float g_ssrThickness = 0.15f;  // depth tolerance for hit
float g_ssrIntensity = 0.8f;   // how strong the reflection is

// ==============================
// Mesh struct
// ==============================
struct Mesh {
    GLuint vao = 0;
    GLuint vbo = 0;
    GLuint ebo = 0;
    GLsizei indexCount = 0;

    GLuint    diffuseTex = 0;             // map_Kd
    glm::vec3 Ka = glm::vec3(0.1f);       // ambient
    glm::vec3 Kd = glm::vec3(5.0f);       // diffuse
    glm::vec3 Ks = glm::vec3(0.0f);       // specular
    float     Ns = 32.0f;                 // shininess

    glm::mat4 model = glm::mat4(1.0f);    // base model (room uses this; trice overridden at draw)
};

std::vector<Mesh> g_meshes;

// ==============================
// Rectangular area light (world-space)
// ==============================
glm::vec3 g_areaCenter = glm::vec3(1.0f, 0.5f, -0.5f); // given
glm::vec2 g_areaSize = glm::vec2(1.0f, 1.0f);        // width, height
glm::vec3 g_areaEuler = glm::vec3(180.0f, 0.0f, 0.0f);  // pitch, yaw, roll in deg
glm::vec3 g_areaColor = glm::vec3(0.8f, 0.6f, 0.0f);  // given color
int       g_areaSamples = 16;                          // 4x4 stratified samples
bool      g_enableAreaLight = true;

// ==============================
// Volumetric light scattering ("God Rays")
// ==============================
GLuint g_volumetricFBO = 0;
GLuint g_volumetricTex = 0;
int    g_volWidth = 0;
int    g_volHeight = 0;
GLuint g_volumetricProgram = 0;

bool  g_enableVolumetric = false;
int   g_volNumSamples = 100;      // demo spec
float g_volExposure = 0.2f;     // exposure in formula
float g_volDecay = 0.96815f; // decay
float g_volDensity = 0.926f;   // density
float g_volWeight = 0.58767f; // weight
float g_volIntensity = 1.0f;     // how strong in final image

float g_volThreshold = 0.7f;   // brightness threshold
float g_volSourceRadius = 0.05f;  // on-screen radius around light

// Demo "sun" light for volumetric (from spec)
bool      g_useDemoVolLight = true;
glm::vec3 g_volDemoLightPos(
    -2.845f * 5.0f,
    2.028f * 2.5f,
    -1.293f * 5.0f
);
bool      g_showVolDemoSphere = true;


// ==============================
// NPR / Toon shading + Edges
// ==============================
bool  g_enableToon = true;
int   g_toonSteps = 3;     // spec says 3
bool  g_enableEdges = true;

float g_edgeDepthThreshold = 0.10f; // tune
float g_edgeNormalThreshold = 1.0f; // tune
float g_edgeStrength = 1.0f;         // 0..1

// ==============================
// TRUE volumetric fog (raymarch)
// ==============================
bool  g_enableVolFog = true;

int   g_volSteps = 256;
float g_volMaxDistance = 50.0f;

float g_volBaseDensity = 0.03f;   // main fog density
float g_volHeightFalloff = 1.0f; // height fog strength

float g_volAnisotropy = -0.2f;     // forward scattering (0 = isotropic)
float g_volExtinction = 1.0f;     // absorption + scattering

float g_volLightIntensity = 8.0f; // volumetric light strength

enum VolumetricMode {
    VOL_OFF = 0,
    VOL_GODRAYS = 1,
    VOL_FOG = 2
};

int g_volMode = VOL_FOG;

GLuint g_occFBO = 0;
GLuint g_occTex = 0;
GLuint g_occDepth = 0;
int g_occW = 0, g_occH = 0;

GLuint g_colorProgram = 0; // simple shader: output uniform color


static void initOcclusionBuffer(int w, int h) {
    w = std::max(1, w / 2);
    h = std::max(1, h / 2);
    if (w == g_occW && h == g_occH) return;
    g_occW = w; g_occH = h;

    if (g_occTex) glDeleteTextures(1, &g_occTex);
    if (g_occDepth) glDeleteRenderbuffers(1, &g_occDepth);
    if (g_occFBO) glDeleteFramebuffers(1, &g_occFBO);

    glGenFramebuffers(1, &g_occFBO);
    glBindFramebuffer(GL_FRAMEBUFFER, g_occFBO);

    glGenTextures(1, &g_occTex);
    glBindTexture(GL_TEXTURE_2D, g_occTex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, w, h, 0, GL_RGBA, GL_FLOAT, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, g_occTex, 0);

    glGenRenderbuffers(1, &g_occDepth);
    glBindRenderbuffer(GL_RENDERBUFFER, g_occDepth);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, w, h);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, g_occDepth);

    GLenum drawBuf = GL_COLOR_ATTACHMENT0;
    glDrawBuffers(1, &drawBuf);

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}


static GLuint createPointShadowCubemap()
{
    GLuint tex = 0;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_CUBE_MAP, tex);

    for (int i = 0; i < 6; ++i) {
        glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, 0,
            GL_DEPTH_COMPONENT24, SHADOW_MAP_SIZE, SHADOW_MAP_SIZE,
            0, GL_DEPTH_COMPONENT, GL_FLOAT, nullptr);
    }

    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);

    return tex;
}



static void initDirectionalShadowMap()
{
    glGenFramebuffers(1, &g_dirShadowFBO);

    glGenTextures(1, &g_dirShadowTex);
    glBindTexture(GL_TEXTURE_2D, g_dirShadowTex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT24,
        DIR_SHADOW_SIZE, DIR_SHADOW_SIZE, 0,
        GL_DEPTH_COMPONENT, GL_FLOAT, nullptr);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    // Crucial for directional shadows at edges:
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
    float border[4] = { 1,1,1,1 }; // outside = lit (depth = 1)
    glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, border);

    glBindFramebuffer(GL_FRAMEBUFFER, g_dirShadowFBO);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, g_dirShadowTex, 0);
    glDrawBuffer(GL_NONE);
    glReadBuffer(GL_NONE);

    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
        std::cerr << "ERROR: Directional shadow FBO not complete!\n";

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}


// ==============================
// GLFW error callback
// ==============================
static void glfw_error_callback(int error, const char* description)
{
    fprintf(stderr, "GLFW Error %d: %s\n", error, description);
}

// ==============================
// Texture loading helpers
// ==============================
static GLuint loadTexture2D(const std::string& path)
{
    int w, h, n;
    stbi_set_flip_vertically_on_load(true);
    unsigned char* data = stbi_load(path.c_str(), &w, &h, &n, 0);
    if (!data || w <= 0 || h <= 0) {
        std::cerr << "Failed to load texture: " << path << "\n";
        return 0;
    }

    std::cout << "Loaded texture " << path << "  size = "
        << w << "x" << h << "  channels = " << n << "\n";

    GLenum format = GL_RGB;
    GLenum internalFormat = GL_RGB8;

    if (n == 1) {
        format = GL_RED;
        internalFormat = GL_R8;
    }
    else if (n == 3) {
        format = GL_RGB;
        internalFormat = GL_RGB8;
    }
    else if (n == 4) {
        format = GL_RGBA;
        internalFormat = GL_RGBA8;
    }
    else {
        std::cerr << "Unsupported channel count (" << n
            << ") in texture: " << path << "\n";
        stbi_image_free(data);
        return 0;
    }

    GLuint tex = 0;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);

    glTexImage2D(GL_TEXTURE_2D, 0, internalFormat,
        w, h, 0, format, GL_UNSIGNED_BYTE, data);

    // grayscale swizzle
    if (n == 1) {
        GLint swizzleMask[] = { GL_RED, GL_RED, GL_RED, GL_ONE };
        glTexParameteriv(GL_TEXTURE_2D, GL_TEXTURE_SWIZZLE_RGBA, swizzleMask);
    }

    glGenerateMipmap(GL_TEXTURE_2D);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);

    stbi_image_free(data);
    return tex;
}

// Try full texName path relative to baseDir, then filename-only
static GLuint loadTextureSmart(const std::string& baseDir, const std::string& texName)
{
    if (texName.empty()) return 0;

    // 1) As-is relative to baseDir
    std::string full1 = baseDir + texName;
    GLuint tex = loadTexture2D(full1);
    if (tex) return tex;

    // 2) Only filename in same dir as OBJ
    size_t pos = texName.find_last_of("/\\");
    std::string fileOnly = (pos == std::string::npos ? texName : texName.substr(pos + 1));
    std::string full2 = baseDir + fileOnly;

    tex = loadTexture2D(full2);
    if (tex) return tex;

    std::cerr << "WARNING: could not load texture '" << texName
        << "' from baseDir '" << baseDir << "'\n";
    return 0;
}

// ==============================
// Shader compilation
// ==============================
static GLuint compileShader(GLenum type, const char* src)
{
    GLuint s = glCreateShader(type);
    glShaderSource(s, 1, &src, nullptr);
    glCompileShader(s);

    GLint ok;
    glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        GLint len;
        glGetShaderiv(s, GL_INFO_LOG_LENGTH, &len);
        std::string log(len, '\0');
        glGetShaderInfoLog(s, len, nullptr, log.data());
        std::cerr << "Shader compile error:\n" << log << "\n";
    }
    return s;
}

static GLuint createProgram(const char* vsSrc, const char* fsSrc)
{
    GLuint vs = compileShader(GL_VERTEX_SHADER, vsSrc);
    GLuint fs = compileShader(GL_FRAGMENT_SHADER, fsSrc);

    GLuint prog = glCreateProgram();
    glAttachShader(prog, vs);
    glAttachShader(prog, fs);
    glLinkProgram(prog);

    GLint ok;
    glGetProgramiv(prog, GL_LINK_STATUS, &ok);
    if (!ok) {
        GLint len;
        glGetProgramiv(prog, GL_INFO_LOG_LENGTH, &len);
        std::string log(len, '\0');
        glGetProgramInfoLog(prog, len, nullptr, log.data());
        std::cerr << "Program link error:\n" << log << "\n";
    }

    glDeleteShader(vs);
    glDeleteShader(fs);

    return prog;
}

// ==============================
// OBJ loader (appends meshes to g_meshes)
// ==============================
static void loadOBJScene(const std::string& objPath, const glm::mat4& modelMatrix)
{
    tinyobj::ObjReaderConfig config;
    config.triangulate = true;

    tinyobj::ObjReader reader;
    if (!reader.ParseFromFile(objPath, config)) {
        std::cerr << "TinyObj error: " << reader.Error() << "\n";
        return;
    }
    if (!reader.Warning().empty()) {
        std::cout << "TinyObj warning: " << reader.Warning() << "\n";
    }

    const auto& attrib = reader.GetAttrib();
    const auto& shapes = reader.GetShapes();
    const auto& materials = reader.GetMaterials();

    // base directory for MTL + textures
    size_t slashPos = objPath.find_last_of("/\\");
    std::string baseDir = (slashPos == std::string::npos)
        ? std::string()
        : objPath.substr(0, slashPos + 1);

    // preload materials: Ka/Kd/Ks/Ns + textures
    std::vector<GLuint>    matTexID(materials.size(), 0);
    std::vector<glm::vec3> matKa(materials.size(), glm::vec3(0.1f));
    std::vector<glm::vec3> matKd(materials.size(), glm::vec3(1.0f));
    std::vector<glm::vec3> matKs(materials.size(), glm::vec3(0.0f));
    std::vector<float>     matNs(materials.size(), 32.0f);

    for (size_t m = 0; m < materials.size(); ++m) {
        matKa[m] = glm::vec3(
            materials[m].ambient[0],
            materials[m].ambient[1],
            materials[m].ambient[2]);
        matKd[m] = glm::vec3(
            materials[m].diffuse[0],
            materials[m].diffuse[1],
            materials[m].diffuse[2]);
        matKs[m] = glm::vec3(
            materials[m].specular[0],
            materials[m].specular[1],
            materials[m].specular[2]);
        matNs[m] = (materials[m].shininess > 0.0f ? materials[m].shininess : 32.0f);

        if (!materials[m].diffuse_texname.empty()) {
            matTexID[m] = loadTextureSmart(baseDir, materials[m].diffuse_texname);
        }
    }

    struct Vertex {
        glm::vec3 pos;
        glm::vec3 norm;
        glm::vec2 uv;
    };

    if (sizeof(Vertex) != sizeof(float) * 8) {
        std::cerr << "ERROR: Vertex size mismatch (got " << sizeof(Vertex)
            << ", expected 32). Aborting load for " << objPath << "\n";
        return;
    }

    for (const auto& shape : shapes) {
        if (shape.mesh.indices.empty())
            continue;

        int matID = -1;
        if (!shape.mesh.material_ids.empty())
            matID = shape.mesh.material_ids[0];

        std::vector<Vertex>        vertices;
        std::vector<unsigned int>  indices;

        vertices.reserve(shape.mesh.indices.size());
        indices.reserve(shape.mesh.indices.size());

        for (size_t i = 0; i < shape.mesh.indices.size(); ++i) {
            const auto& idx = shape.mesh.indices[i];
            Vertex v{};

            // position
            if (idx.vertex_index >= 0) {
                v.pos = glm::vec3(
                    attrib.vertices[3 * idx.vertex_index + 0],
                    attrib.vertices[3 * idx.vertex_index + 1],
                    attrib.vertices[3 * idx.vertex_index + 2]
                );
            }
            else {
                v.pos = glm::vec3(0.0f);
            }

            // normal
            if (idx.normal_index >= 0) {
                v.norm = glm::vec3(
                    attrib.normals[3 * idx.normal_index + 0],
                    attrib.normals[3 * idx.normal_index + 1],
                    attrib.normals[3 * idx.normal_index + 2]
                );
            }
            else {
                v.norm = glm::vec3(0, 1, 0);
            }

            // texcoord
            if (idx.texcoord_index >= 0) {
                v.uv = glm::vec2(
                    attrib.texcoords[2 * idx.texcoord_index + 0],
                    attrib.texcoords[2 * idx.texcoord_index + 1]
                );
            }
            else {
                v.uv = glm::vec2(0.0f);
            }

            vertices.push_back(v);
            indices.push_back(static_cast<unsigned int>(vertices.size() - 1));
        }

        if (vertices.empty() || indices.empty()) continue;

        Mesh mesh;
        mesh.model = modelMatrix;

        glGenVertexArrays(1, &mesh.vao);
        glGenBuffers(1, &mesh.vbo);
        glGenBuffers(1, &mesh.ebo);

        glBindVertexArray(mesh.vao);

        glBindBuffer(GL_ARRAY_BUFFER, mesh.vbo);
        glBufferData(GL_ARRAY_BUFFER,
            vertices.size() * sizeof(Vertex),
            vertices.data(),
            GL_STATIC_DRAW);

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, mesh.ebo);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER,
            indices.size() * sizeof(unsigned int),
            indices.data(),
            GL_STATIC_DRAW);

        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE,
            sizeof(Vertex), (void*)offsetof(Vertex, pos));

        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE,
            sizeof(Vertex), (void*)offsetof(Vertex, norm));

        glEnableVertexAttribArray(2);
        glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE,
            sizeof(Vertex), (void*)offsetof(Vertex, uv));

        glBindVertexArray(0);

        mesh.indexCount = static_cast<GLsizei>(indices.size());

        if (matID >= 0 && matID < (int)materials.size()) {
            mesh.Ka = matKa[matID];
            mesh.Kd = matKd[matID];
            mesh.Ks = matKs[matID];
            mesh.Ns = matNs[matID];
            mesh.diffuseTex = matTexID[matID];
        }

        g_meshes.push_back(mesh);
    }

    std::cout << "Loaded OBJ: " << objPath
        << " (total meshes: " << g_meshes.size() << ")\n";
}

// ==============================
// Shadow map init
// ==============================
static void initShadowMap()
{
    glGenFramebuffers(1, &g_shadowFBO);
    glGenTextures(1, &g_shadowTex);

    glBindTexture(GL_TEXTURE_CUBE_MAP, g_shadowTex);
    for (int i = 0; i < 6; ++i) {
        glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, 0,
            GL_DEPTH_COMPONENT24,
            SHADOW_MAP_SIZE, SHADOW_MAP_SIZE,
            0, GL_DEPTH_COMPONENT, GL_FLOAT, nullptr);
    }

    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);

    glBindFramebuffer(GL_FRAMEBUFFER, g_shadowFBO);
    // Attach one face just to validate FBO; we'll reattach per-face later
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT,
        GL_TEXTURE_CUBE_MAP_POSITIVE_X, g_shadowTex, 0);
    glDrawBuffer(GL_NONE);
    glReadBuffer(GL_NONE);

    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
        std::cerr << "ERROR: Shadow FBO not complete!\n";
    }
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}


static void initBloomBuffers(int width, int height)
{
    if (width <= 0 || height <= 0) return;
    if (width == g_bloomWidth && height == g_bloomHeight) return;

    g_bloomWidth = width;
    g_bloomHeight = height;

    // Delete old
    if (g_hdrColorTex)   glDeleteTextures(1, &g_hdrColorTex);
    if (g_brightColorTex) glDeleteTextures(1, &g_brightColorTex);
    if (g_hdrFBO)        glDeleteFramebuffers(1, &g_hdrFBO);
    glDeleteTextures(2, g_pingpongTex);
    glDeleteFramebuffers(2, g_pingpongFBO);

    // HDR FBO (scene + bright)
    glGenFramebuffers(1, &g_hdrFBO);
    glBindFramebuffer(GL_FRAMEBUFFER, g_hdrFBO);

    // Scene color (HDR)
    glGenTextures(1, &g_hdrColorTex);
    glBindTexture(GL_TEXTURE_2D, g_hdrColorTex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, width, height, 0,
        GL_RGBA, GL_FLOAT, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
        GL_TEXTURE_2D, g_hdrColorTex, 0);

    // Bright color (HDR)
    glGenTextures(1, &g_brightColorTex);
    glBindTexture(GL_TEXTURE_2D, g_brightColorTex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, width, height, 0,
        GL_RGBA, GL_FLOAT, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1,
        GL_TEXTURE_2D, g_brightColorTex, 0);

    GLenum attachments[2] = { GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1 };
    glDrawBuffers(2, attachments);

    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
        std::cerr << "ERROR: HDR FBO not complete!\n";
    }

    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    // Ping-pong FBOs for blur
    glGenFramebuffers(2, g_pingpongFBO);
    glGenTextures(2, g_pingpongTex);
    for (int i = 0; i < 2; ++i) {
        glBindFramebuffer(GL_FRAMEBUFFER, g_pingpongFBO[i]);
        glBindTexture(GL_TEXTURE_2D, g_pingpongTex[i]);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, width, height, 0,
            GL_RGBA, GL_FLOAT, nullptr);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
            GL_TEXTURE_2D, g_pingpongTex[i], 0);

        if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
            std::cerr << "ERROR: Ping-pong FBO " << i << " not complete!\n";
        }
    }

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}


static void initSSAOKernelAndNoise()
{
    if (!g_ssaoKernel.empty())
        return;

    // Kernel
    std::uniform_real_distribution<float> rnd(0.0f, 1.0f);
    std::default_random_engine           gen;

    g_ssaoKernel.reserve(SSAO_KERNEL_SIZE);
    for (int i = 0; i < SSAO_KERNEL_SIZE; ++i) {
        glm::vec3 sample(
            rnd(gen) * 2.0f - 1.0f,
            rnd(gen) * 2.0f - 1.0f,
            rnd(gen));        // hemisphere (z >= 0)
        sample = glm::normalize(sample);
        sample *= rnd(gen);   // scale by random [0,1]

        // bias samples closer to origin
        float scale = float(i) / float(SSAO_KERNEL_SIZE);
        scale = glm::mix(0.1f, 1.0f, scale * scale);
        sample *= scale;

        g_ssaoKernel.push_back(sample);
    }

    // Noise texture (4x4)
    std::vector<glm::vec3> noiseData;
    noiseData.reserve(16);
    for (int i = 0; i < 16; ++i) {
        glm::vec3 noise(
            rnd(gen) * 2.0f - 1.0f,
            rnd(gen) * 2.0f - 1.0f,
            0.0f);
        noiseData.push_back(noise);
    }

    if (g_ssaoNoiseTex)
        glDeleteTextures(1, &g_ssaoNoiseTex);

    glGenTextures(1, &g_ssaoNoiseTex);
    glBindTexture(GL_TEXTURE_2D, g_ssaoNoiseTex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB16F, 4, 4, 0,
        GL_RGB, GL_FLOAT, noiseData.data());
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
}

static void initSSAOBuffer(int width, int height)
{
    if (width <= 0 || height <= 0) return;
    if (width == g_ssaoWidth && height == g_ssaoHeight) return;

    g_ssaoWidth = width;
    g_ssaoHeight = height;

    if (g_ssaoTex)  glDeleteTextures(1, &g_ssaoTex);
    if (g_ssaoFBO)  glDeleteFramebuffers(1, &g_ssaoFBO);

    glGenFramebuffers(1, &g_ssaoFBO);
    glBindFramebuffer(GL_FRAMEBUFFER, g_ssaoFBO);

    glGenTextures(1, &g_ssaoTex);
    glBindTexture(GL_TEXTURE_2D, g_ssaoTex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_R8, width, height, 0,
        GL_RED, GL_UNSIGNED_BYTE, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
        GL_TEXTURE_2D, g_ssaoTex, 0);

    GLenum drawBuf = GL_COLOR_ATTACHMENT0;
    glDrawBuffers(1, &drawBuf);

    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
        std::cerr << "ERROR: SSAO FBO not complete!\n";
    }
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

static void initSSAOBlurBuffer(int width, int height)
{
    if (width <= 0 || height <= 0) return;

    if (g_ssaoBlurTex) glDeleteTextures(1, &g_ssaoBlurTex);
    if (g_ssaoBlurFBO) glDeleteFramebuffers(1, &g_ssaoBlurFBO);

    glGenFramebuffers(1, &g_ssaoBlurFBO);
    glBindFramebuffer(GL_FRAMEBUFFER, g_ssaoBlurFBO);

    glGenTextures(1, &g_ssaoBlurTex);
    glBindTexture(GL_TEXTURE_2D, g_ssaoBlurTex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_R8, width, height, 0, GL_RED, GL_UNSIGNED_BYTE, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, g_ssaoBlurTex, 0);

    GLenum drawBuf = GL_COLOR_ATTACHMENT0;
    glDrawBuffers(1, &drawBuf);

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}


static void initVolumetricBuffer(int width, int height)
{
    if (width <= 0 || height <= 0) return;
    if (width == g_volWidth && height == g_volHeight) return;

    g_volWidth = width;
    g_volHeight = height;

    if (g_volumetricTex) glDeleteTextures(1, &g_volumetricTex);
    if (g_volumetricFBO) glDeleteFramebuffers(1, &g_volumetricFBO);

    glGenFramebuffers(1, &g_volumetricFBO);
    glBindFramebuffer(GL_FRAMEBUFFER, g_volumetricFBO);

    glGenTextures(1, &g_volumetricTex);
    glBindTexture(GL_TEXTURE_2D, g_volumetricTex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F,
        width, height, 0, GL_RGBA, GL_FLOAT, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
        GL_TEXTURE_2D, g_volumetricTex, 0);

    GLenum drawBuf = GL_COLOR_ATTACHMENT0;
    glDrawBuffers(1, &drawBuf);

    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
        std::cerr << "ERROR: Volumetric FBO not complete!\n";
    }

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}



// ==============================
// G-buffer init / resize
// ==============================
static void initGBuffer(int width, int height)
{
    if (width <= 0 || height <= 0) return;

    g_gbufferWidth = width;
    g_gbufferHeight = height;

    // Delete old resources if they exist
    if (g_gPositionTex) glDeleteTextures(1, &g_gPositionTex);
    if (g_gNormalTex)   glDeleteTextures(1, &g_gNormalTex);
    if (g_gAmbientTex)  glDeleteTextures(1, &g_gAmbientTex);
    if (g_gDiffuseTex)  glDeleteTextures(1, &g_gDiffuseTex);
    if (g_gSpecularTex) glDeleteTextures(1, &g_gSpecularTex);
    if (g_gDepthRBO)    glDeleteRenderbuffers(1, &g_gDepthRBO);
    if (g_gbufferFBO)   glDeleteFramebuffers(1, &g_gbufferFBO);

    glGenFramebuffers(1, &g_gbufferFBO);
    glBindFramebuffer(GL_FRAMEBUFFER, g_gbufferFBO);

    // Position (RGBA16F)
    glGenTextures(1, &g_gPositionTex);
    glBindTexture(GL_TEXTURE_2D, g_gPositionTex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F,
        width, height, 0, GL_RGBA, GL_FLOAT, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
        GL_TEXTURE_2D, g_gPositionTex, 0);

    // Normal (RGBA16F)
    glGenTextures(1, &g_gNormalTex);
    glBindTexture(GL_TEXTURE_2D, g_gNormalTex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F,
        width, height, 0, GL_RGBA, GL_FLOAT, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1,
        GL_TEXTURE_2D, g_gNormalTex, 0);

    // Ambient (RGBA8)
    glGenTextures(1, &g_gAmbientTex);
    glBindTexture(GL_TEXTURE_2D, g_gAmbientTex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8,
        width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT2,
        GL_TEXTURE_2D, g_gAmbientTex, 0);

    // Diffuse (RGBA8)
    glGenTextures(1, &g_gDiffuseTex);
    glBindTexture(GL_TEXTURE_2D, g_gDiffuseTex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8,
        width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT3,
        GL_TEXTURE_2D, g_gDiffuseTex, 0);

    // Specular (RGBA16F – RGB = Ks, A = Ns)
    glGenTextures(1, &g_gSpecularTex);
    glBindTexture(GL_TEXTURE_2D, g_gSpecularTex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F,
        width, height, 0, GL_RGBA, GL_FLOAT, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT4,
        GL_TEXTURE_2D, g_gSpecularTex, 0);

    // Depth buffer
    glGenRenderbuffers(1, &g_gDepthRBO);
    glBindRenderbuffer(GL_RENDERBUFFER, g_gDepthRBO);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, width, height);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT,
        GL_RENDERBUFFER, g_gDepthRBO);

    GLenum attachments[5] = {
        GL_COLOR_ATTACHMENT0,
        GL_COLOR_ATTACHMENT1,
        GL_COLOR_ATTACHMENT2,
        GL_COLOR_ATTACHMENT3,
        GL_COLOR_ATTACHMENT4
    };
    glDrawBuffers(5, attachments);

    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
        std::cerr << "ERROR: G-buffer FBO not complete!\n";
    }

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    initBloomBuffers(width, height);
    initSSAOBuffer(width, height);
    initSSAOBlurBuffer(width, height);
    initVolumetricBuffer(width, height);

}

// ==============================
// Fullscreen quad
// ==============================
static void initFullscreenQuad()
{
    if (g_quadVAO != 0) return;

    float quadVerts[] = {
        // positions   // uvs
        -1.0f, -1.0f,  0.0f, 0.0f,
         1.0f, -1.0f,  1.0f, 0.0f,
        -1.0f,  1.0f,  0.0f, 1.0f,
         1.0f,  1.0f,  1.0f, 1.0f
    };

    glGenVertexArrays(1, &g_quadVAO);
    glGenBuffers(1, &g_quadVBO);

    glBindVertexArray(g_quadVAO);
    glBindBuffer(GL_ARRAY_BUFFER, g_quadVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(quadVerts), quadVerts, GL_STATIC_DRAW);

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(2 * sizeof(float)));

    glBindVertexArray(0);
}

static void initLightSphere()
{
    if (g_lightSphereVAO != 0) return;

    const int stacks = 16;
    const int slices = 32;

    struct V {
        glm::vec3 pos;
        glm::vec3 norm;
        glm::vec2 uv;
    };

    std::vector<V> verts;
    std::vector<unsigned> idx;

    for (int i = 0; i <= stacks; i++) {
        float v = float(i) / stacks;
        float theta = v * 3.1415926f;

        for (int j = 0; j <= slices; j++) {
            float u = float(j) / slices;
            float phi = u * 6.2831853f;

            glm::vec3 n(
                sin(theta) * cos(phi),
                cos(theta),
                sin(theta) * sin(phi));

            verts.push_back({ n, n, glm::vec2(u, v) });
        }
    }

    for (int i = 0; i < stacks; i++) {
        for (int j = 0; j < slices; j++) {
            int row1 = i * (slices + 1);
            int row2 = (i + 1) * (slices + 1);

            idx.push_back(row1 + j);
            idx.push_back(row2 + j);
            idx.push_back(row2 + j + 1);

            idx.push_back(row1 + j);
            idx.push_back(row2 + j + 1);
            idx.push_back(row1 + j + 1);
        }
    }

    g_lightSphereIndexCount = idx.size();

    glGenVertexArrays(1, &g_lightSphereVAO);
    glGenBuffers(1, &g_lightSphereVBO);
    glGenBuffers(1, &g_lightSphereEBO);

    glBindVertexArray(g_lightSphereVAO);

    glBindBuffer(GL_ARRAY_BUFFER, g_lightSphereVBO);
    glBufferData(GL_ARRAY_BUFFER, verts.size() * sizeof(V), verts.data(), GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, g_lightSphereEBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, idx.size() * sizeof(unsigned), idx.data(), GL_STATIC_DRAW);

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(V), (void*)0);

    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(V), (void*)offsetof(V, norm));

    glEnableVertexAttribArray(2);
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(V), (void*)offsetof(V, uv));

    glBindVertexArray(0);
}

// ==============================
// Area rectangle mesh init
// ==============================
static void initAreaRectMesh()
{
    if (g_areaRectVAO != 0) return;

    struct V {
        glm::vec3 pos;
        glm::vec3 norm;
        glm::vec2 uv;
    };

    // Unit quad in local space, in the XY plane, facing +Z
    V verts[4] = {
        { glm::vec3(-1.0f, -1.0f, 0.0f), glm::vec3(0, 0, 1), glm::vec2(0.0f, 0.0f) },
        { glm::vec3(1.0f, -1.0f, 0.0f), glm::vec3(0, 0, 1), glm::vec2(1.0f, 0.0f) },
        { glm::vec3(1.0f,  1.0f, 0.0f), glm::vec3(0, 0, 1), glm::vec2(1.0f, 1.0f) },
        { glm::vec3(-1.0f,  1.0f, 0.0f), glm::vec3(0, 0, 1), glm::vec2(0.0f, 1.0f) },
    };

    unsigned indices[6] = { 0, 1, 2, 0, 2, 3 };
    g_areaRectIndexCount = 6;

    glGenVertexArrays(1, &g_areaRectVAO);
    glGenBuffers(1, &g_areaRectVBO);
    glGenBuffers(1, &g_areaRectEBO);

    glBindVertexArray(g_areaRectVAO);

    glBindBuffer(GL_ARRAY_BUFFER, g_areaRectVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(verts), verts, GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, g_areaRectEBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(V), (void*)offsetof(V, pos));

    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(V), (void*)offsetof(V, norm));

    glEnableVertexAttribArray(2);
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(V), (void*)offsetof(V, uv));

    glBindVertexArray(0);
}

// Helper to get current trice model matrix
static glm::mat4 getTriceModel()
{
    glm::mat4 m(1.0f);
    m = glm::translate(m, g_tricePos);
    m = glm::rotate(m, glm::radians(g_triceYaw), glm::vec3(0.0f, 1.0f, 0.0f));
    m = glm::scale(m, glm::vec3(g_triceScale));
    return m;
}

// Area rectangle model matrix (matching area light basis)
static glm::mat4 getAreaRectModel()
{
    glm::mat4 M(1.0f);

    // Translate to center
    M = glm::translate(M, g_areaCenter);

    // Same rotation order as buildAreaLightBasis()
    float pitch = glm::radians(g_areaEuler.x);
    float yaw = glm::radians(g_areaEuler.y);
    float roll = glm::radians(g_areaEuler.z);

    M = glm::rotate(M, yaw, glm::vec3(0, 1, 0)); // Yaw
    M = glm::rotate(M, pitch, glm::vec3(1, 0, 0)); // Pitch
    M = glm::rotate(M, roll, glm::vec3(0, 0, 1)); // Roll

    // Scale unit quad [-1,1]x[-1,1] -> actual size (width, height)
    M = glm::scale(M, glm::vec3(g_areaSize.x * 0.5f,
        g_areaSize.y * 0.5f,
        1.0f));

    return M;
}

// ==============================
// Rendering
// ==============================

// Build the 6 view-projection matrices for the point light
static void buildPointShadowMatrices(const glm::vec3& lightPos)
{
    glm::mat4 proj = glm::perspective(glm::radians(90.0f),
        1.0f,
        g_pointShadowNear,
        g_pointShadowFar);

    g_pointShadowMatrices[0] = proj *
        glm::lookAt(lightPos, lightPos + glm::vec3(1, 0, 0), glm::vec3(0, -1, 0));
    g_pointShadowMatrices[1] = proj *
        glm::lookAt(lightPos, lightPos + glm::vec3(-1, 0, 0), glm::vec3(0, -1, 0));
    g_pointShadowMatrices[2] = proj *
        glm::lookAt(lightPos, lightPos + glm::vec3(0, 1, 0), glm::vec3(0, 0, 1));
    g_pointShadowMatrices[3] = proj *
        glm::lookAt(lightPos, lightPos + glm::vec3(0, -1, 0), glm::vec3(0, 0, -1));
    g_pointShadowMatrices[4] = proj *
        glm::lookAt(lightPos, lightPos + glm::vec3(0, 0, 1), glm::vec3(0, -1, 0));
    g_pointShadowMatrices[5] = proj *
        glm::lookAt(lightPos, lightPos + glm::vec3(0, 0, -1), glm::vec3(0, -1, 0));
}

// Depth pass: render scene into cube shadow map
static void renderShadowPass(const glm::vec3& lightPos, GLuint targetCubeTex)
{
    buildPointShadowMatrices(lightPos);

    glViewport(0, 0, SHADOW_MAP_SIZE, SHADOW_MAP_SIZE);
    glBindFramebuffer(GL_FRAMEBUFFER, g_shadowFBO);
    glEnable(GL_DEPTH_TEST);

    glUseProgram(g_depthProgram);

    GLint locModel = glGetUniformLocation(g_depthProgram, "u_model");
    GLint locLightVP = glGetUniformLocation(g_depthProgram, "u_lightVP");
    GLint locLightPos = glGetUniformLocation(g_depthProgram, "u_lightPos");
    GLint locFar = glGetUniformLocation(g_depthProgram, "u_far");

    glUniform3fv(locLightPos, 1, glm::value_ptr(lightPos));
    glUniform1f(locFar, g_pointShadowFar);

    glm::mat4 triceModel = getTriceModel();

    glEnable(GL_CULL_FACE);
    glCullFace(GL_FRONT); // helps reduce acne

    // Render for each cube face
    for (int face = 0; face < 6; ++face) {
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT,
            GL_TEXTURE_CUBE_MAP_POSITIVE_X + face,
            targetCubeTex, 0);
        glClear(GL_DEPTH_BUFFER_BIT);

        glUniformMatrix4fv(locLightVP, 1, GL_FALSE,
            glm::value_ptr(g_pointShadowMatrices[face]));

        for (size_t i = 0; i < g_meshes.size(); ++i) {
            const Mesh& mesh = g_meshes[i];
            if (mesh.vao == 0 || mesh.indexCount <= 0) continue;

            bool isTrice = (g_triceFirstMesh != (size_t)-1 && i >= g_triceFirstMesh);
            glm::mat4 model = isTrice ? triceModel : mesh.model;

            glBindVertexArray(mesh.vao);
            glUniformMatrix4fv(locModel, 1, GL_FALSE, glm::value_ptr(model));
            glDrawElements(GL_TRIANGLES, mesh.indexCount, GL_UNSIGNED_INT, 0);
        }
    }

    glCullFace(GL_BACK);
    glDisable(GL_CULL_FACE);

    glBindVertexArray(0);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

static void buildAreaLightBasis(glm::vec3& outNormal, glm::vec3& outTangent, glm::vec3& outBitangent)
{
    // Start as a +Z facing quad
    glm::mat4 R(1.0f);
    float pitch = glm::radians(g_areaEuler.x); // rotate around X
    float yaw = glm::radians(g_areaEuler.y); // rotate around Y
    float roll = glm::radians(g_areaEuler.z); // rotate around Z

    R = glm::rotate(R, yaw, glm::vec3(0, 1, 0)); // Yaw
    R = glm::rotate(R, pitch, glm::vec3(1, 0, 0)); // Pitch
    R = glm::rotate(R, roll, glm::vec3(0, 0, 1)); // Roll

    outNormal = glm::normalize(glm::vec3(R * glm::vec4(0, 0, 1, 0)));
    outTangent = glm::normalize(glm::vec3(R * glm::vec4(1, 0, 0, 0)));
    outBitangent = glm::normalize(glm::vec3(R * glm::vec4(0, 1, 0, 0)));
}

static void renderDirectionalShadowPass()
{
    // Build light view/proj from your spec-controlled globals:
    glm::mat4 lightView = glm::lookAt(g_dirLightEye, g_dirLightCenter, g_dirLightUp);

    // Ortho “range” box (Range = 5 means [-5,5] on x/y in light space)
    glm::mat4 lightProj = glm::ortho(-g_dirLightRange, g_dirLightRange,
        -g_dirLightRange, g_dirLightRange,
        g_dirLightNear, g_dirLightFar);
    g_dirLightVP = lightProj * lightView;

    glViewport(0, 0, DIR_SHADOW_SIZE, DIR_SHADOW_SIZE);
    glBindFramebuffer(GL_FRAMEBUFFER, g_dirShadowFBO);
    glEnable(GL_DEPTH_TEST);
    glClear(GL_DEPTH_BUFFER_BIT);

    glUseProgram(g_dirDepthProgram);

    GLint locModel = glGetUniformLocation(g_dirDepthProgram, "u_model");
    GLint locLightVP = glGetUniformLocation(g_dirDepthProgram, "u_lightVP");
    glUniformMatrix4fv(locLightVP, 1, GL_FALSE, glm::value_ptr(g_dirLightVP)); // FIX


    glm::mat4 triceModel = getTriceModel();

    // Optional: slope acne reduction trick
    glEnable(GL_CULL_FACE);
    glCullFace(GL_FRONT);

    for (size_t i = 0; i < g_meshes.size(); ++i) {
        const Mesh& mesh = g_meshes[i];
        if (mesh.vao == 0 || mesh.indexCount <= 0) continue;

        bool isTrice = (g_triceFirstMesh != (size_t)-1 && i >= g_triceFirstMesh);
        glm::mat4 model = isTrice ? triceModel : mesh.model;

        glBindVertexArray(mesh.vao);
        glUniformMatrix4fv(locModel, 1, GL_FALSE, glm::value_ptr(model));
        glDrawElements(GL_TRIANGLES, mesh.indexCount, GL_UNSIGNED_INT, 0);
    }

    glCullFace(GL_BACK);
    glDisable(GL_CULL_FACE);

    glBindVertexArray(0);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}


// Main display: deferred shading pipeline
static void on_display(GLFWwindow* window)
{

    glm::vec3 lightA = g_pointLightPos;
    glm::vec3 lightB = g_volDemoLightPos; // or another real point light

    renderShadowPass(lightA, g_shadowTexA);
    renderShadowPass(lightB, g_shadowTexB);

    int display_w, display_h;
    glfwGetFramebufferSize(window, &display_w, &display_h);

    // Resize G-buffer if needed
    if (display_w != g_gbufferWidth || display_h != g_gbufferHeight) {
        initGBuffer(display_w, display_h);
        initOcclusionBuffer(display_w, display_h);
    }

    float aspect = (display_h > 0) ? (float)display_w / (float)display_h : 1.0f;
    glm::mat4 proj = glm::perspective(glm::radians(g_fov), aspect, 0.1f, 100.0f);
    glm::mat4 view = glm::lookAt(g_eye, g_center, g_up);

    glm::vec3 lightPos = g_pointLightPos; // point light position follows sphere

    // 1) Shadow pass (point light cube)
    renderDirectionalShadowPass();

    /*renderShadowPass(lightPos);
    renderShadowPass(g_volDemoLightPos);*/


    // 2) Geometry pass -> G-buffers
    glBindFramebuffer(GL_FRAMEBUFFER, g_gbufferFBO);
    glViewport(0, 0, g_gbufferWidth, g_gbufferHeight);
    glEnable(GL_DEPTH_TEST);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glUseProgram(g_geomProgram);
    glUniformMatrix4fv(glGetUniformLocation(g_geomProgram, "u_view"), 1, GL_FALSE, glm::value_ptr(view));
    glUniformMatrix4fv(glGetUniformLocation(g_geomProgram, "u_proj"), 1, GL_FALSE, glm::value_ptr(proj));

    glm::mat4 triceModel = getTriceModel();

    for (size_t i = 0; i < g_meshes.size(); ++i) {
        const Mesh& mesh = g_meshes[i];
        if (mesh.vao == 0 || mesh.indexCount <= 0) continue;

        bool isTrice = (g_triceFirstMesh != (size_t)-1 && i >= g_triceFirstMesh);
        glm::mat4 model = isTrice ? triceModel : mesh.model;

        // For trice: enable normal map if checkbox is on and texture is valid
        bool useNormalMap = isTrice && g_enableNormalMap && (g_triceNormalTex != 0);
        glUniform1i(glGetUniformLocation(g_geomProgram, "u_useNormalMap"),
            useNormalMap ? 1 : 0);

        if (useNormalMap) {
            glActiveTexture(GL_TEXTURE1);
            glBindTexture(GL_TEXTURE_2D, g_triceNormalTex);
            glUniform1i(glGetUniformLocation(g_geomProgram, "u_normalMap"), 1);
        }
        else {
            // No normal map for this mesh
            glActiveTexture(GL_TEXTURE1);
            glBindTexture(GL_TEXTURE_2D, 0);
        }

        glBindVertexArray(mesh.vao);
        glUniformMatrix4fv(glGetUniformLocation(g_geomProgram, "u_model"), 1, GL_FALSE, glm::value_ptr(model));

        glm::vec3 Ka = mesh.Ka;
        glm::vec3 Kd = mesh.Kd;
        glm::vec3 Ks = mesh.Ks;
        float     Ns = mesh.Ns;
        bool      useTex = (mesh.diffuseTex != 0);

        if (isTrice) {
            Kd = glm::vec3(0.2f, 0.9f, 0.2f);
            useTex = false; // keep solid green dino in deferred too
        }

        glUniform3fv(glGetUniformLocation(g_geomProgram, "u_Ka"), 1, glm::value_ptr(Ka));
        glUniform3fv(glGetUniformLocation(g_geomProgram, "u_Kd"), 1, glm::value_ptr(Kd));
        glUniform3fv(glGetUniformLocation(g_geomProgram, "u_Ks"), 1, glm::value_ptr(Ks));
        glUniform1f(glGetUniformLocation(g_geomProgram, "u_Ns"), Ns);

        glUniform1i(glGetUniformLocation(g_geomProgram, "u_useTex"), useTex ? 1 : 0);
        if (useTex) {
            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D, mesh.diffuseTex);
            glUniform1i(glGetUniformLocation(g_geomProgram, "u_diffuseTex"), 0);
        }

        glDrawElements(GL_TRIANGLES, mesh.indexCount, GL_UNSIGNED_INT, 0);
    }

    // ---- Emissive Area Rect (visible rectangle light) ----
    if (g_areaRectVAO != 0 && g_enableAreaLight)
    {
        glm::mat4 model = getAreaRectModel();

        glBindVertexArray(g_areaRectVAO);

        glUniformMatrix4fv(glGetUniformLocation(g_geomProgram, "u_model"),
            1, GL_FALSE, glm::value_ptr(model));

        glm::vec3 Kd = g_areaColor * 20.0f;  // tweak for bloom
        glm::vec3 Ka = glm::vec3(0.0f);
        glm::vec3 Ks = glm::vec3(0.0f);
        float Ns = -1.0f;                    // negative => emissive

        glUniform3fv(glGetUniformLocation(g_geomProgram, "u_Ka"), 1, glm::value_ptr(Ka));
        glUniform3fv(glGetUniformLocation(g_geomProgram, "u_Kd"), 1, glm::value_ptr(Kd));
        glUniform3fv(glGetUniformLocation(g_geomProgram, "u_Ks"), 1, glm::value_ptr(Ks));
        glUniform1f(glGetUniformLocation(g_geomProgram, "u_Ns"), Ns);

        glUniform1i(glGetUniformLocation(g_geomProgram, "u_useTex"), 0);
        glUniform1i(glGetUniformLocation(g_geomProgram, "u_useNormalMap"), 0);

        glDrawElements(GL_TRIANGLES, g_areaRectIndexCount, GL_UNSIGNED_INT, 0);
    }

    // ---- Emissive Point Light Sphere ----
    if (g_lightSphereVAO != 0)
    {
        glm::mat4 model(1.0f);
        model = glm::translate(model, g_pointLightPos);     // same position as point light
        model = glm::scale(model, glm::vec3(g_lightSphereRadius));

        glBindVertexArray(g_lightSphereVAO);

        glUniformMatrix4fv(glGetUniformLocation(g_geomProgram, "u_model"),
            1, GL_FALSE, glm::value_ptr(model));

        // emissive color stored in Kd
        glm::vec3 Kd = glm::vec3(50.0f);  // bright white so bloom works
        glm::vec3 Ka = glm::vec3(0.0f);
        glm::vec3 Ks = glm::vec3(0.0f);
        float Ns = -1.0f;    // negative shininess marks emissive

        glUniform3fv(glGetUniformLocation(g_geomProgram, "u_Ka"), 1, glm::value_ptr(Ka));
        glUniform3fv(glGetUniformLocation(g_geomProgram, "u_Kd"), 1, glm::value_ptr(Kd));
        glUniform3fv(glGetUniformLocation(g_geomProgram, "u_Ks"), 1, glm::value_ptr(Ks));
        glUniform1f(glGetUniformLocation(g_geomProgram, "u_Ns"), Ns);

        glUniform1i(glGetUniformLocation(g_geomProgram, "u_useTex"), 0);
        glUniform1i(glGetUniformLocation(g_geomProgram, "u_useNormalMap"), 0);

        glDrawElements(GL_TRIANGLES, g_lightSphereIndexCount, GL_UNSIGNED_INT, 0);
    }

    // NEW: small emissive sphere at volumetric demo light position
    if (g_lightSphereVAO != 0 && g_showVolDemoSphere)
    {
        glm::mat4 model(1.0f);
        model = glm::translate(model, g_volDemoLightPos);
        model = glm::scale(model, glm::vec3(g_lightSphereRadius * 0.6f));

        glBindVertexArray(g_lightSphereVAO);

        glUniformMatrix4fv(glGetUniformLocation(g_geomProgram, "u_model"),
            1, GL_FALSE, glm::value_ptr(model));

        glm::vec3 Kd = glm::vec3(40.0f); // bright white "sun"
        glm::vec3 Ka = glm::vec3(0.0f);
        glm::vec3 Ks = glm::vec3(0.0f);
        float Ns = -1.0f; // emissive

        glUniform3fv(glGetUniformLocation(g_geomProgram, "u_Ka"), 1, glm::value_ptr(Ka));
        glUniform3fv(glGetUniformLocation(g_geomProgram, "u_Kd"), 1, glm::value_ptr(Kd));
        glUniform3fv(glGetUniformLocation(g_geomProgram, "u_Ks"), 1, glm::value_ptr(Ks));
        glUniform1f(glGetUniformLocation(g_geomProgram, "u_Ns"), Ns);

        glUniform1i(glGetUniformLocation(g_geomProgram, "u_useTex"), 0);
        glUniform1i(glGetUniformLocation(g_geomProgram, "u_useNormalMap"), 0);

        glDrawElements(GL_TRIANGLES, g_lightSphereIndexCount, GL_UNSIGNED_INT, 0);
    }

    glBindVertexArray(0);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    // 2.5) SSAO pass
    if (g_enableSSAO) {
        glBindFramebuffer(GL_FRAMEBUFFER, g_ssaoFBO);
        glViewport(0, 0, g_ssaoWidth, g_ssaoHeight);
        glClear(GL_COLOR_BUFFER_BIT);
        glDisable(GL_DEPTH_TEST);

        glUseProgram(g_ssaoProgram);

        // G-buffer inputs
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, g_gPositionTex);
        glUniform1i(glGetUniformLocation(g_ssaoProgram, "gPosition"), 0);

        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_2D, g_gNormalTex);
        glUniform1i(glGetUniformLocation(g_ssaoProgram, "gNormal"), 1);

        glActiveTexture(GL_TEXTURE2);
        glBindTexture(GL_TEXTURE_2D, g_ssaoNoiseTex);
        glUniform1i(glGetUniformLocation(g_ssaoProgram, "texNoise"), 2);

        // uniforms
        glm::vec2 noiseScale(
            (float)g_ssaoWidth / 4.0f,
            (float)g_ssaoHeight / 4.0f);
        glUniform2fv(glGetUniformLocation(g_ssaoProgram, "u_noiseScale"),
            1, glm::value_ptr(noiseScale));
        glUniformMatrix4fv(glGetUniformLocation(g_ssaoProgram, "u_view"), 1,
            GL_FALSE, glm::value_ptr(view));
        glUniformMatrix4fv(glGetUniformLocation(g_ssaoProgram, "u_proj"), 1,
            GL_FALSE, glm::value_ptr(proj));
        glUniform1f(glGetUniformLocation(g_ssaoProgram, "u_radius"), g_ssaoRadius);
        glUniform1f(glGetUniformLocation(g_ssaoProgram, "u_bias"), g_ssaoBias);

        // kernel samples
        for (int i = 0; i < SSAO_KERNEL_SIZE; ++i) {
            std::string name = "u_samples[" + std::to_string(i) + "]";
            glUniform3fv(glGetUniformLocation(g_ssaoProgram, name.c_str()),
                1, glm::value_ptr(g_ssaoKernel[i]));
        }

        glBindVertexArray(g_quadVAO);
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
        glBindVertexArray(0);
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
    }

    if (g_enableSSAO) {
        glBindFramebuffer(GL_FRAMEBUFFER, g_ssaoBlurFBO);
        glViewport(0, 0, g_ssaoWidth, g_ssaoHeight);
        glClear(GL_COLOR_BUFFER_BIT);
        glDisable(GL_DEPTH_TEST);

        glUseProgram(g_ssaoBlurProgram);

        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, g_ssaoTex);
        glUniform1i(glGetUniformLocation(g_ssaoBlurProgram, "u_ssaoInput"), 0);

        glBindVertexArray(g_quadVAO);
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
        glBindVertexArray(0);

        glBindFramebuffer(GL_FRAMEBUFFER, 0);
    }



    // 3) Lighting pass -> HDR FBO (scene + bright)
    glBindFramebuffer(GL_FRAMEBUFFER, g_hdrFBO);
    glViewport(0, 0, display_w, display_h);
    glDisable(GL_DEPTH_TEST);
    glClear(GL_COLOR_BUFFER_BIT);

    glUseProgram(g_lightProgram);

    // Bind G-buffer textures
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, g_gPositionTex);
    glUniform1i(glGetUniformLocation(g_lightProgram, "gPosition"), 0);

    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, g_gNormalTex);
    glUniform1i(glGetUniformLocation(g_lightProgram, "gNormal"), 1);

    glActiveTexture(GL_TEXTURE2);
    glBindTexture(GL_TEXTURE_2D, g_gAmbientTex);
    glUniform1i(glGetUniformLocation(g_lightProgram, "gAmbient"), 2);

    glActiveTexture(GL_TEXTURE3);
    glBindTexture(GL_TEXTURE_2D, g_gDiffuseTex);
    glUniform1i(glGetUniformLocation(g_lightProgram, "gDiffuse"), 3);

    glActiveTexture(GL_TEXTURE4);
    glBindTexture(GL_TEXTURE_2D, g_gSpecularTex);
    glUniform1i(glGetUniformLocation(g_lightProgram, "gSpecular"), 4);

    // Shadow cube map
    glActiveTexture(GL_TEXTURE5);
    glBindTexture(GL_TEXTURE_CUBE_MAP, g_shadowTexA);
    glUniform1i(glGetUniformLocation(g_lightProgram, "u_shadowCube"), 5);
    glUniform1f(glGetUniformLocation(g_lightProgram, "u_far"), g_pointShadowFar);

    // SSAO texture
    glActiveTexture(GL_TEXTURE6);
    glBindTexture(GL_TEXTURE_2D, g_enableSSAO ? g_ssaoBlurTex : 0);
    glUniform1i(glGetUniformLocation(g_lightProgram, "u_ssaoTex"), 6);
    glUniform1i(glGetUniformLocation(g_lightProgram, "u_enableSSAO"),
        g_enableSSAO ? 1 : 0);


    // ---- Directional shadow map + light matrices ----

    // Bind shadow map (2D depth)
    glActiveTexture(GL_TEXTURE7);
    glBindTexture(GL_TEXTURE_2D, g_dirShadowTex);
    glUniform1i(glGetUniformLocation(g_lightProgram, "u_dirShadowMap"), 7);

    // Pass light VP (computed in renderDirectionalShadowPass())
    glUniformMatrix4fv(glGetUniformLocation(g_lightProgram, "u_lightVP"),
        1, GL_FALSE, glm::value_ptr(g_dirLightVP));

    // Pass light direction (world space).
    // Convention: u_lightDir = direction FROM surface TO light
    // Explanation: (eye -> center) is the direction the light rays travel.
    // From surface to light is opposite of ray travel, so use (center - eye).
    glm::vec3 lightDir = glm::normalize(g_dirLightEye - g_dirLightCenter);
    //glm::vec3 lightDir = glm::normalize(g_dirLightCenter - g_dirLightEye);

    glUniform3fv(glGetUniformLocation(g_lightProgram, "u_lightDir"),
        1, glm::value_ptr(lightDir));



    // Camera / light uniforms
    glUniform3fv(glGetUniformLocation(g_lightProgram, "u_eye"), 1, glm::value_ptr(g_eye));

    glm::vec3 Ia(0.02f, 0.02f, 0.02f);
    glm::vec3 Id(0.9f, 0.9f, 0.9f);
    glm::vec3 Is(0.4f, 0.4f, 0.4f);

    glUniform3fv(glGetUniformLocation(g_lightProgram, "u_lightPos"), 1, glm::value_ptr(lightPos));
    glUniform3fv(glGetUniformLocation(g_lightProgram, "u_Ia"), 1, glm::value_ptr(Ia));
    glUniform3fv(glGetUniformLocation(g_lightProgram, "u_Id"), 1, glm::value_ptr(Id));
    glUniform3fv(glGetUniformLocation(g_lightProgram, "u_Is"), 1, glm::value_ptr(Is));

    glUniform1f(glGetUniformLocation(g_lightProgram, "u_attConst"), 1.0f);
    glUniform1f(glGetUniformLocation(g_lightProgram, "u_attLinear"), 0.7f);
    glUniform1f(glGetUniformLocation(g_lightProgram, "u_attQuadratic"), 0.14f);

    glUniform1f(glGetUniformLocation(g_lightProgram, "u_shadowStrength"), g_shadowStrength);
    glUniform1i(glGetUniformLocation(g_lightProgram, "u_enableShadows"), g_enableShadows ? 1 : 0);
    glUniform1i(glGetUniformLocation(g_lightProgram, "u_viewMode"), g_viewMode);

    glUniform1i(glGetUniformLocation(g_lightProgram, "u_enableBloom"), g_enableBloom ? 1 : 0);
    glUniform1f(glGetUniformLocation(g_lightProgram, "u_bloomThreshold"), g_bloomThreshold);

    // View / proj for SSR
    glUniformMatrix4fv(glGetUniformLocation(g_lightProgram, "u_view"),
        1, GL_FALSE, glm::value_ptr(view));
    glUniformMatrix4fv(glGetUniformLocation(g_lightProgram, "u_proj"),
        1, GL_FALSE, glm::value_ptr(proj));

    // SSR params
    glUniform1i(glGetUniformLocation(g_lightProgram, "u_enableSSR"), g_enableSSR ? 1 : 0);
    glUniform1f(glGetUniformLocation(g_lightProgram, "u_ssrMaxDistance"), g_ssrMaxDistance);
    glUniform1i(glGetUniformLocation(g_lightProgram, "u_ssrMaxSteps"), g_ssrMaxSteps);
    glUniform1f(glGetUniformLocation(g_lightProgram, "u_ssrStep"), g_ssrStep);
    glUniform1f(glGetUniformLocation(g_lightProgram, "u_ssrThickness"), g_ssrThickness);
    glUniform1f(glGetUniformLocation(g_lightProgram, "u_ssrIntensity"), g_ssrIntensity);

    // ----- area light basis & uniforms (before draw!) -----
    {
        glm::vec3 rectN, rectT, rectB;
        buildAreaLightBasis(rectN, rectT, rectB);

        glUniform3fv(glGetUniformLocation(g_lightProgram, "u_rectCenter"), 1, glm::value_ptr(g_areaCenter));
        glUniform3fv(glGetUniformLocation(g_lightProgram, "u_rectNormal"), 1, glm::value_ptr(rectN));
        glUniform3fv(glGetUniformLocation(g_lightProgram, "u_rectTangent"), 1, glm::value_ptr(rectT));
        glUniform3fv(glGetUniformLocation(g_lightProgram, "u_rectBitangent"), 1, glm::value_ptr(rectB));
        glUniform2fv(glGetUniformLocation(g_lightProgram, "u_rectSize"), 1, glm::value_ptr(g_areaSize));
        glUniform3fv(glGetUniformLocation(g_lightProgram, "u_rectColor"), 1, glm::value_ptr(g_areaColor));
        glUniform1i(glGetUniformLocation(g_lightProgram, "u_rectSamples"), g_areaSamples);
        glUniform1i(glGetUniformLocation(g_lightProgram, "u_enableAreaLight"), g_enableAreaLight ? 1 : 0);
    }

    glUniform1i(glGetUniformLocation(g_lightProgram, "u_enableToon"), g_enableToon ? 1 : 0);
    glUniform1i(glGetUniformLocation(g_lightProgram, "u_toonSteps"), g_toonSteps);

    glUniform1i(glGetUniformLocation(g_lightProgram, "u_enableEdges"), g_enableEdges ? 1 : 0);
    glUniform1f(glGetUniformLocation(g_lightProgram, "u_edgeDepthThreshold"), g_edgeDepthThreshold);
    glUniform1f(glGetUniformLocation(g_lightProgram, "u_edgeNormalThreshold"), g_edgeNormalThreshold);
    glUniform1f(glGetUniformLocation(g_lightProgram, "u_edgeStrength"), g_edgeStrength);


    glBindVertexArray(g_quadVAO);
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    glBindVertexArray(0);

    // 4) Blur bright texture (gaussian ping-pong)
    bool horizontal = true;
    bool firstIter = true;
    int  iterations = glm::clamp(g_blurIterations, 1, 20);

    glUseProgram(g_blurProgram);
    for (int i = 0; i < iterations; ++i) {
        glBindFramebuffer(GL_FRAMEBUFFER, g_pingpongFBO[horizontal ? 1 : 0]);
        glUniform1i(glGetUniformLocation(g_blurProgram, "u_horizontal"), horizontal ? 1 : 0);

        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D,
            firstIter ? g_brightColorTex
            : g_pingpongTex[horizontal ? 0 : 1]);
        glUniform1i(glGetUniformLocation(g_blurProgram, "u_image"), 0);

        glBindVertexArray(g_quadVAO);
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

        horizontal = !horizontal;
        if (firstIter) firstIter = false;
    }

    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    // --- 4.5) Volumetric light scattering ("god rays") ---
    glm::vec2 lightScreenPos(-1.0f, -1.0f);
    bool doVol = g_enableVolumetric;
    float edgeFade = 1.0f;

    glm::vec3 volLightPos = g_useDemoVolLight ? g_volDemoLightPos : lightPos;


    

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    //end

    //slide demo
    // =======================================================
    // SCREEN-SPACE GOD RAYS (SLIDES VERSION)
    // =======================================================
    // =======================================================
    // GOD RAYS (Screen-space) — GPU Gems style
    // Requires:
    //   - g_occFBO + g_occTex (initOcclusionBuffer)
    //   - g_volumetricFBO + g_volumetricTex (initVolumetricBuffer)
    //   - g_colorProgram (simple solid color shader)
    //   - g_volumetricProgram (radial blur shader)
    //   - g_quadVAO
    // =======================================================
    if (g_volMode == VOL_GODRAYS)
    {
        // Choose which light position drives the shafts
        glm::vec3 lightWorld = g_useDemoVolLight ? g_volDemoLightPos : g_pointLightPos;

        // ---------------------------------------------------
        // 1) OCCLUSION MASK PASS (downsampled)
        //    - Render all occluders black
        //    - Render light as small white blob
        // ---------------------------------------------------
        glBindFramebuffer(GL_FRAMEBUFFER, g_occFBO);
        glViewport(0, 0, g_occW, g_occH);
        glEnable(GL_DEPTH_TEST);

        glClearColor(0, 0, 0, 1);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        glUseProgram(g_colorProgram);
        glUniformMatrix4fv(glGetUniformLocation(g_colorProgram, "u_view"), 1, GL_FALSE, glm::value_ptr(view));
        glUniformMatrix4fv(glGetUniformLocation(g_colorProgram, "u_proj"), 1, GL_FALSE, glm::value_ptr(proj));

        // Draw scene as BLACK (occluders)
        glUniform4f(glGetUniformLocation(g_colorProgram, "u_color"), 0, 0, 0, 1);

        glm::mat4 triceModel = getTriceModel();
        for (size_t i = 0; i < g_meshes.size(); ++i) {
            const Mesh& mesh = g_meshes[i];
            if (!mesh.vao || mesh.indexCount <= 0) continue;

            bool isTrice = (g_triceFirstMesh != (size_t)-1 && i >= g_triceFirstMesh);
            glm::mat4 model = isTrice ? triceModel : mesh.model;

            glUniformMatrix4fv(glGetUniformLocation(g_colorProgram, "u_model"), 1, GL_FALSE, glm::value_ptr(model));
            glBindVertexArray(mesh.vao);
            glDrawElements(GL_TRIANGLES, mesh.indexCount, GL_UNSIGNED_INT, 0);
        }

        // Draw the light as WHITE (keep it SMALL so you get long shafts)
        glUniform4f(glGetUniformLocation(g_colorProgram, "u_color"), 1, 1, 1, 1);

        if (g_lightSphereVAO != 0) {
            glm::mat4 lm(1.0f);
            lm = glm::translate(lm, lightWorld);
            lm = glm::scale(lm, glm::vec3(g_lightSphereRadius * 0.25f)); // << smaller = better shafts
            glUniformMatrix4fv(glGetUniformLocation(g_colorProgram, "u_model"), 1, GL_FALSE, glm::value_ptr(lm));
            glBindVertexArray(g_lightSphereVAO);
            glDrawElements(GL_TRIANGLES, g_lightSphereIndexCount, GL_UNSIGNED_INT, 0);
        }

        glBindVertexArray(0);
        glBindFramebuffer(GL_FRAMEBUFFER, 0);

        // ---------------------------------------------------
        // 2) Compute LIGHT SCREEN POS (UV 0..1)
        // ---------------------------------------------------
        glm::vec4 clip = proj * view * glm::vec4(lightWorld, 1.0f);

        // If behind camera, just clear volumetric and skip
        if (clip.w <= 0.0f) {
            glBindFramebuffer(GL_FRAMEBUFFER, g_volumetricFBO);
            glViewport(0, 0, g_volWidth, g_volHeight);
            glDisable(GL_DEPTH_TEST);
            glClearColor(0, 0, 0, 0);
            glClear(GL_COLOR_BUFFER_BIT);
            glBindFramebuffer(GL_FRAMEBUFFER, 0);
        }
        else
        {
            glm::vec3 ndc = glm::vec3(clip) / clip.w;          // -1..1
            glm::vec2 uv = glm::vec2(ndc.x, ndc.y) * 0.5f + 0.5f; // 0..1

            // Clamp to screen (GPU Gems style still works fine clamped)
            glm::vec2 lightScreenPos = glm::clamp(uv, glm::vec2(0.0f), glm::vec2(1.0f));

            // Optional edge fade (prevents ugly smear when light is off-screen-ish)
            glm::vec2 clamped = lightScreenPos;
            float edgeFade = 1.0f - glm::clamp(glm::length(uv - clamped) * 2.0f, 0.0f, 1.0f);

            // ---------------------------------------------------
            // 3) RADIAL BLUR PASS -> g_volumetricTex
            // ---------------------------------------------------
            glBindFramebuffer(GL_FRAMEBUFFER, g_volumetricFBO);
            glViewport(0, 0, g_volWidth, g_volHeight);
            glDisable(GL_DEPTH_TEST);

            glClearColor(0, 0, 0, 0);
            glClear(GL_COLOR_BUFFER_BIT);

            glUseProgram(g_volumetricProgram);

            // Input = occlusion mask
            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D, g_occTex);
            glUniform1i(glGetUniformLocation(g_volumetricProgram, "u_scene"), 0);

            glUniform2fv(glGetUniformLocation(g_volumetricProgram, "u_lightScreenPos"), 1, &lightScreenPos.x);

            // GPU Gems params
            glUniform1i(glGetUniformLocation(g_volumetricProgram, "u_numSamples"), g_volNumSamples);
            glUniform1f(glGetUniformLocation(g_volumetricProgram, "u_exposure"), g_volExposure * edgeFade);
            glUniform1f(glGetUniformLocation(g_volumetricProgram, "u_decay"), g_volDecay);
            glUniform1f(glGetUniformLocation(g_volumetricProgram, "u_density"), g_volDensity);
            glUniform1f(glGetUniformLocation(g_volumetricProgram, "u_weight"), g_volWeight);

            glBindVertexArray(g_quadVAO);
            glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
            glBindVertexArray(0);

            glBindFramebuffer(GL_FRAMEBUFFER, 0);
        }
    }





    if (g_volMode == VOL_FOG)
    {
        glBindFramebuffer(GL_FRAMEBUFFER, g_volumetricFBO);
        glViewport(0, 0, g_volWidth, g_volHeight);
        glDisable(GL_DEPTH_TEST);

        glClearColor(0, 0, 0, 0);
        glClear(GL_COLOR_BUFFER_BIT);

        glUseProgram(g_volRaymarchProgram);

        // --------------------------------------------------
        // Bind POINT LIGHT shadow cubemap for volumetric fog
        // --------------------------------------------------
        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_CUBE_MAP, g_shadowTexB);
        glUniform1i(
            glGetUniformLocation(g_volRaymarchProgram, "u_pointShadowMap"),
            1
        );

        // Shadow far plane (must match depth pass)
        glUniform1f(
            glGetUniformLocation(g_volRaymarchProgram, "u_pointShadowFar"),
            g_pointShadowFar
        );

        // Small bias to avoid self-shadowing in fog
        glUniform1f(
            glGetUniformLocation(g_volRaymarchProgram, "u_shadowBias"),
            0.05f
        );



        // gPosition
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, g_gPositionTex);
        glUniform1i(glGetUniformLocation(g_volRaymarchProgram, "gPosition"), 0);

        // camera
        glUniform3fv(glGetUniformLocation(g_volRaymarchProgram, "u_eye"),
            1, glm::value_ptr(g_eye));

        // volumetric point light = demo sphere
        glm::vec3 volPos = g_volDemoLightPos;
        //renderShadowPass(volPos);
        glUniform3fv(glGetUniformLocation(g_volRaymarchProgram, "u_volLightPos"),
            1, glm::value_ptr(volPos));
        /*glUniform3fv(glGetUniformLocation(g_volRaymarchProgram, "u_volLightPos"),
            1, glm::value_ptr(g_volDemoLightPos));*/
        glUniform3f(glGetUniformLocation(g_volRaymarchProgram, "u_volLightColor"),
            1.0f, 1.0f, 1.0f);
        glUniform1f(glGetUniformLocation(g_volRaymarchProgram, "u_volLightIntensity"),
            g_volLightIntensity);

        // fog params
        glUniform1i(glGetUniformLocation(g_volRaymarchProgram, "u_steps"), g_volSteps);
        glUniform1f(glGetUniformLocation(g_volRaymarchProgram, "u_maxDistance"), g_volMaxDistance);
        glUniform1f(glGetUniformLocation(g_volRaymarchProgram, "u_baseDensity"), g_volBaseDensity);
        glUniform1f(glGetUniformLocation(g_volRaymarchProgram, "u_heightFalloff"), g_volHeightFalloff);
        glUniform1f(glGetUniformLocation(g_volRaymarchProgram, "u_anisotropy"), g_volAnisotropy);
        glUniform1f(glGetUniformLocation(g_volRaymarchProgram, "u_extinction"), g_volExtinction);
        glUniform1f(glGetUniformLocation(g_volRaymarchProgram, "u_volAtten"), 0.002f);

        glBindVertexArray(g_quadVAO);
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
        glBindVertexArray(0);

        glBindFramebuffer(GL_FRAMEBUFFER, 0);
    }




    /*if (g_enableVolumetric && !doVol) {
        glBindFramebuffer(GL_FRAMEBUFFER, g_volumetricFBO);
        glViewport(0, 0, display_w, display_h);
        glDisable(GL_DEPTH_TEST);
        glClearColor(0, 0, 0, 0);
        glClear(GL_COLOR_BUFFER_BIT);
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
    }*/


    // 5) Final combine: HDR scene + blurred bloom -> default framebuffer
    glViewport(0, 0, display_w, display_h);

    glClearColor(g_clearColor.x * g_clearColor.w,
        g_clearColor.y * g_clearColor.w,
        g_clearColor.z * g_clearColor.w,
        g_clearColor.w);
    glClear(GL_COLOR_BUFFER_BIT);

    glUseProgram(g_finalProgram);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, g_hdrColorTex);
    glUniform1i(glGetUniformLocation(g_finalProgram, "u_scene"), 0);

    GLuint blurredTex = g_pingpongTex[horizontal ? 0 : 1];
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, blurredTex);
    glUniform1i(glGetUniformLocation(g_finalProgram, "u_bloomBlur"), 1);

    // Volumetric texture (may be empty if disabled/off-screen)
    glActiveTexture(GL_TEXTURE2);
    glBindTexture(GL_TEXTURE_2D, g_volumetricTex);
    glUniform1i(glGetUniformLocation(g_finalProgram, "u_volumetric"), 2);

    glUniform1i(glGetUniformLocation(g_finalProgram, "u_enableBloom"),
        g_enableBloom ? 1 : 0);

    glUniform1i(glGetUniformLocation(g_finalProgram, "u_enableVolumetric"),
        (g_volMode != VOL_OFF) ? 1 : 0);

    glUniform1f(glGetUniformLocation(g_finalProgram, "u_bloomIntensity"),
        g_bloomIntensity);

    glUniform1f(glGetUniformLocation(g_finalProgram, "u_volumetricIntensity"),
        g_volLightIntensity);

    glUniform1f(glGetUniformLocation(g_finalProgram, "u_exposure"),
        g_exposure);




    glBindVertexArray(g_quadVAO);
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    glBindVertexArray(0);

    glUseProgram(0);
}

// ==============================
// ImGui UI
// ==============================
static void on_gui()
{
    ImGui::Begin("Control Panel");

    ImGui::Text("Camera");
    ImGui::DragFloat3("Eye", glm::value_ptr(g_eye), 0.05f);
    ImGui::DragFloat3("Center", glm::value_ptr(g_center), 0.05f);
    ImGui::SliderFloat("FOV", &g_fov, 20.0f, 80.0f);

    ImGui::Separator();
    ImGui::Text("Triceratops Transform");
    ImGui::DragFloat3("Trice Pos", &g_tricePos.x, 0.01f);
    ImGui::DragFloat("Trice Scale", &g_triceScale,
        0.0001f, 0.00001f, 0.01f, "%.5f");
    ImGui::SliderFloat("Trice Yaw", &g_triceYaw, -180.0f, 180.0f);
    ImGui::Checkbox("Normal map (Trice)", &g_enableNormalMap);

    ImGui::Separator();
    ImGui::Text("Point Light (Fire Place)");
    ImGui::DragFloat3("Point Pos", &g_pointLightPos.x, 0.05f);
    //ImGui::DragFloat3("Light Center", glm::value_ptr(g_lightCenter), 0.05f);
    //ImGui::SliderFloat("Light Range", &g_lightRange, 1.0f, 10.0f);

    ImGui::Separator();
    ImGui::Text("Directional Light Camera (2D shadow)");
    ImGui::DragFloat3("Dir Eye", &g_dirLightEye.x, 0.05f);
    ImGui::DragFloat3("Dir Look at", &g_dirLightCenter.x, 0.05f);
    //ImGui::DragFloat3("Dir Up", &g_dirLightUp.x, 0.05f);

    ImGui::SliderFloat("Dir Near", &g_dirLightNear, 0.01f, 2.0f);
    ImGui::SliderFloat("Dir Far", &g_dirLightFar, 1.0f, 50.0f);
    ImGui::SliderFloat("Dir Range", &g_dirLightRange, 1.0f, 20.0f);

    ImGui::Checkbox("Enable Shadows", &g_enableShadows);
    ImGui::SliderFloat("Shadow Strength", &g_shadowStrength, 0.0f, 1.0f);

    ImGui::Separator();
    ImGui::Text("Bloom / HDR");
    ImGui::Checkbox("Enable Bloom", &g_enableBloom);
    ImGui::SliderFloat("Bloom Threshold", &g_bloomThreshold, 0.1f, 5.0f);
    ImGui::SliderFloat("Bloom Intensity", &g_bloomIntensity, 0.0f, 3.0f);
    ImGui::SliderInt("Blur Iterations", &g_blurIterations, 1, 20);
    ImGui::SliderFloat("Exposure", &g_exposure, 0.1f, 5.0f);

    ImGui::Separator();
    ImGui::Text("SSAO");
    ImGui::Checkbox("Enable SSAO", &g_enableSSAO);
    ImGui::SliderFloat("SSAO Radius", &g_ssaoRadius, 0.1f, 1.0f);
    ImGui::SliderFloat("SSAO Bias", &g_ssaoBias, 0.0f, 0.1f);

    ImGui::Separator();
    ImGui::Text("Screen-Space Reflection (floor)");
    ImGui::Checkbox("Enable SSR (floor)", &g_enableSSR);
    ImGui::SliderFloat("SSR Intensity", &g_ssrIntensity, 0.0f, 1.0f);
    ImGui::SliderFloat("SSR Max Distance", &g_ssrMaxDistance, 1.0f, 10.0f);
    ImGui::SliderInt("SSR Max Steps", &g_ssrMaxSteps, 10, 80);
    ImGui::SliderFloat("SSR Step", &g_ssrStep, 0.05f, 0.5f);
    ImGui::SliderFloat("SSR Thickness", &g_ssrThickness, 0.01f, 0.4f);

    ImGui::Separator();
    ImGui::Text("Rectangular Area Light");
    ImGui::Checkbox("Enable Area Light", &g_enableAreaLight);
    //ImGui::DragFloat3("Center", &g_areaCenter.x, 0.01f);
    //ImGui::DragFloat2("Size (W,H)", &g_areaSize.x, 0.01f, 0.01f, 5.0f);
    ImGui::DragFloat3("Euler (pitch,yaw,roll)", &g_areaEuler.x, 0.5f, -180.0f, 180.0f);
    ImGui::ColorEdit3("Area Color", &g_areaColor.x);
    //ImGui::SliderInt("Samples", &g_areaSamples, 1, 32);

    ImGui::Separator();
    ImGui::Text("Volumetric Light");

    ImGui::Checkbox("Enable Volumetric Light", &g_enableVolFog);
    ImGui::SliderFloat("Vol Light Intensity", &g_volLightIntensity, 0.0f, 50.0f);

    /*ImGui::Separator();
    ImGui::Text("Volumetric");*/

    /*const char* volModes[] = {
        "Off",
        "God Rays (Screen-space)",
        "True Fog (Raymarch)"
    };
    ImGui::Combo("Volumetric Mode", &g_volMode, volModes, 3);*/

    /*if (g_volMode == VOL_GODRAYS) {
        ImGui::SliderFloat("Exposure", &g_volExposure, 0.0f, 1.0f);
        ImGui::SliderFloat("Density", &g_volDensity, 0.0f, 2.0f);
        ImGui::SliderFloat("Decay", &g_volDecay, 0.8f, 1.0f);
    }*/

    if (g_volMode == VOL_FOG) {
        ImGui::SliderFloat("Fog Density", &g_volBaseDensity, 0.0f, 0.2f);
        //ImGui::SliderFloat("Light Intensity", &g_volLightIntensity, 0.0f, 50.0f);
    }


    /*ImGui::SliderInt("March Steps", &g_volSteps, 16, 256);
    ImGui::SliderFloat("Max Distance", &g_volMaxDistance, 5.0f, 80.0f);

    ImGui::SliderFloat("Base Density", &g_volBaseDensity, 0.0f, 2.0f);
    ImGui::SliderFloat("Height Falloff", &g_volHeightFalloff, 0.0f, 1.0f);

    ImGui::SliderFloat("Anisotropy (g)", &g_volAnisotropy, -0.2f, 0.9f);
    ImGui::SliderFloat("Extinction", &g_volExtinction, 0.1f, 5.0f);*/


    // NEW: demo sun light
    //ImGui::Separator();
    //ImGui::Text("Volumetric Demo Light");
    //ImGui::Checkbox("Use demo light pos", &g_useDemoVolLight);
    //ImGui::Checkbox("Show demo light sphere", &g_showVolDemoSphere);
    ImGui::DragFloat3("Demo light world pos", &g_volDemoLightPos.x, 0.05f);
    /*ImGui::SliderFloat("Vol Source radius", &g_volSourceRadius, 0.0f, 0.3f);
    ImGui::SliderFloat("Vol Threshold", &g_volThreshold, 0.0f, 2.0f);*/

    ImGui::Separator();
    ImGui::Text("NPR / Toon");
    ImGui::Checkbox("Enable Toon Shading", &g_enableToon);
    ImGui::SliderInt("Toon Steps", &g_toonSteps, 2, 6);

    ImGui::Checkbox("Enable Edge Detection", &g_enableEdges);
    ImGui::SliderFloat("Edge Depth Thresh", &g_edgeDepthThreshold, 0.001f, 0.5f);
    ImGui::SliderFloat("Edge Normal Thresh", &g_edgeNormalThreshold, 0.01f, 1.0f);
    ImGui::SliderFloat("Edge Strength", &g_edgeStrength, 0.0f, 1.0f);



    ImGui::Separator();
    ImGui::Text("G-Buffer Debug");

    const char* modes[] = {
        "Lighting",
        "World Pos",
        "World Normal",
        "Ambient",
        "Diffuse",
        "Specular",
        "SSAO"
    };
    ImGui::Combo("View", &g_viewMode, modes, IM_ARRAYSIZE(modes));

    ImGui::Separator();
    ImGui::ColorEdit3("Clear color", (float*)&g_clearColor);

    ImGui::Separator();
    ImGui::Text("Controls:");
    ImGui::BulletText("WASD: move camera on plane");
    ImGui::BulletText("Q/E: move camera down/up");
    ImGui::BulletText("RMB drag: orbit camera");

    ImGui::End();
}

// ==============================
// Input callbacks
// ==============================
static void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
    if (action != GLFW_PRESS && action != GLFW_REPEAT) return;

    const float step = 0.1f;
    glm::vec3 forward = glm::normalize(g_center - g_eye);
    glm::vec3 right = glm::normalize(glm::cross(forward, g_up));

    if (key == GLFW_KEY_W) {
        g_eye += forward * step;
        g_center += forward * step;
    }
    if (key == GLFW_KEY_S) {
        g_eye -= forward * step;
        g_center -= forward * step;
    }
    if (key == GLFW_KEY_A) {
        g_eye -= right * step;
        g_center -= right * step;
    }
    if (key == GLFW_KEY_D) {
        g_eye += right * step;
        g_center += right * step;
    }
    if (key == GLFW_KEY_Q) {
        g_eye.y -= step;
        g_center.y -= step;
    }
    if (key == GLFW_KEY_E) {
        g_eye.y += step;
        g_center.y += step;
    }
}

static void mouse_button_callback(GLFWwindow* window, int button, int action, int mods)
{
    if (button == GLFW_MOUSE_BUTTON_RIGHT) {
        if (action == GLFW_PRESS) {
            g_mouseRot = true;
            glfwGetCursorPos(window, &g_lastX, &g_lastY);
        }
        else if (action == GLFW_RELEASE) {
            g_mouseRot = false;
        }
    }
}

static void cursor_pos_callback(GLFWwindow* window, double xpos, double ypos)
{
    if (!g_mouseRot) return;

    float dx = float(xpos - g_lastX);
    float dy = float(ypos - g_lastY);
    g_lastX = xpos;
    g_lastY = ypos;

    glm::vec3 dir = g_eye - g_center;
    float     radius = glm::length(dir);
    if (radius < 1e-3f) return;

    float yaw = atan2(dir.z, dir.x);
    float pitch = asin(dir.y / radius);

    const float sens = 0.005f;
    yaw -= dx * sens;
    pitch -= dy * sens;
    pitch = glm::clamp(pitch, -1.4f, 1.4f);

    dir.x = radius * cos(pitch) * cos(yaw);
    dir.y = radius * sin(pitch);
    dir.z = radius * cos(pitch) * sin(yaw);

    g_eye = g_center + dir;
}

static std::string loadTextFile(const std::string& path)
{
    std::ifstream file(path);
    if (!file.is_open()) {
        std::cerr << "ERROR: failed to open shader file: " << path << "\n";
        return std::string();
    }
    std::stringstream ss;
    ss << file.rdbuf();
    return ss.str();
}

static GLuint createProgramFromFiles(const std::string& vsPath, const std::string& fsPath)
{
    std::string vsSrcStr = loadTextFile(vsPath);
    std::string fsSrcStr = loadTextFile(fsPath);
    if (vsSrcStr.empty() || fsSrcStr.empty()) {
        std::cerr << "ERROR: shader source empty. VS=" << vsPath << " FS=" << fsPath << "\n";
        return 0;
    }

    const char* vsSrc = vsSrcStr.c_str();
    const char* fsSrc = fsSrcStr.c_str();

    GLuint vs = compileShader(GL_VERTEX_SHADER, vsSrc);
    GLuint fs = compileShader(GL_FRAGMENT_SHADER, fsSrc);

    GLuint prog = glCreateProgram();
    glAttachShader(prog, vs);
    glAttachShader(prog, fs);
    glLinkProgram(prog);

    GLint ok;
    glGetProgramiv(prog, GL_LINK_STATUS, &ok);
    if (!ok) {
        GLint len;
        glGetProgramiv(prog, GL_INFO_LOG_LENGTH, &len);
        std::string log(len, '\0');
        glGetProgramInfoLog(prog, len, nullptr, log.data());
        std::cerr << "Program link error (" << vsPath << ", " << fsPath << "):\n" << log << "\n";
    }

    glDeleteShader(vs);
    glDeleteShader(fs);
    return prog;
}

// ==============================
// main()
// ==============================
int main(int, char**)
{
    glfwSetErrorCallback(glfw_error_callback);
    if (!glfwInit())
        return 1;

    const char* glsl_version = "#version 410";
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 1);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);

    float main_scale = ImGui_ImplGlfw_GetContentScaleForMonitor(glfwGetPrimaryMonitor());
    GLFWwindow* window = glfwCreateWindow(
        (int)(1280 * main_scale), (int)(800 * main_scale),
        "Final_Framework", nullptr, nullptr);
    if (window == nullptr)
        return 1;

    glfwMakeContextCurrent(window);
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        std::cerr << "Failed to initialize GLAD\n";
        return -1;
    }
    glfwSwapInterval(1); // vsync

    // Callbacks
    glfwSetKeyCallback(window, key_callback);
    glfwSetMouseButtonCallback(window, mouse_button_callback);
    glfwSetCursorPosCallback(window, cursor_pos_callback);

    // ImGui init
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;

    ImGui::StyleColorsDark();
    ImGuiStyle& style = ImGui::GetStyle();
    style.ScaleAllSizes(main_scale);
    style.FontScaleDpi = main_scale;

    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init(glsl_version);

    // Our GL resources
    g_geomProgram = createProgramFromFiles("./shaders/geom.vert", "./shaders/geom.frag");
    g_lightProgram = createProgramFromFiles("./shaders/quad.vert", "./shaders/light.frag");
    g_blurProgram = createProgramFromFiles("./shaders/quad.vert", "./shaders/blur.frag");
    g_finalProgram = createProgramFromFiles("./shaders/quad.vert", "./shaders/final.frag");
    g_depthProgram = createProgramFromFiles("./shaders/depth.vert", "./shaders/depth.frag");
    g_dirDepthProgram = createProgramFromFiles("./shaders/dir_depth.vert", "./shaders/dir_depth.frag");
    g_ssaoProgram = createProgramFromFiles("./shaders/quad.vert", "./shaders/ssao.frag");
    g_ssaoBlurProgram = createProgramFromFiles("./shaders/quad.vert", "./shaders/ssao_blur.frag");
    g_volumetricProgram = createProgramFromFiles("./shaders/quad.vert", "./shaders/volumetric.frag");
    g_volRaymarchProgram = createProgramFromFiles("./shaders/quad.vert", "./shaders/volumetric_raymarch.frag");
    g_colorProgram = createProgramFromFiles("./shaders/color.vert", "./shaders/color.frag");


    initSSAOKernelAndNoise();
    initShadowMap();
    initDirectionalShadowMap();
    initFullscreenQuad();
    initLightSphere();
    initAreaRectMesh();


    g_shadowTexA = createPointShadowCubemap();
    g_shadowTexB = createPointShadowCubemap();

    int fbw, fbh;
    glfwGetFramebufferSize(window, &fbw, &fbh);
    initGBuffer(fbw, fbh);
    initOcclusionBuffer(fbw, fbh);

    // Load models
    loadOBJScene("./assets/indoor_model/Grey_White_Room.obj", glm::mat4(1.0f));
    g_triceFirstMesh = g_meshes.size(); // triceratops starts here
    loadOBJScene("./assets/indoor_model/trice.obj", glm::mat4(1.0f));

    g_triceNormalTex = loadTexture2D("./assets/indoor_model/tricnorm.jpg");

    // Main loop
    while (!glfwWindowShouldClose(window))
    {
        glfwPollEvents();
        if (glfwGetWindowAttrib(window, GLFW_ICONIFIED) != 0) {
            ImGui_ImplGlfw_Sleep(10);
            continue;
        }

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        on_gui();
        on_display(window);

        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        glfwSwapBuffers(window);
    }

    // Cleanup
    for (auto& m : g_meshes) {
        if (m.diffuseTex) glDeleteTextures(1, &m.diffuseTex);
        if (m.ebo)        glDeleteBuffers(1, &m.ebo);
        if (m.vbo)        glDeleteBuffers(1, &m.vbo);
        if (m.vao)        glDeleteVertexArrays(1, &m.vao);
    }

    if (g_shadowTex)    glDeleteTextures(1, &g_shadowTex);
    if (g_shadowFBO)    glDeleteFramebuffers(1, &g_shadowFBO);
    if (g_depthProgram) glDeleteProgram(g_depthProgram);

    if (g_gPositionTex) glDeleteTextures(1, &g_gPositionTex);
    if (g_gNormalTex)   glDeleteTextures(1, &g_gNormalTex);
    if (g_gAmbientTex)  glDeleteTextures(1, &g_gAmbientTex);
    if (g_gDiffuseTex)  glDeleteTextures(1, &g_gDiffuseTex);
    if (g_gSpecularTex) glDeleteTextures(1, &g_gSpecularTex);
    if (g_gDepthRBO)    glDeleteRenderbuffers(1, &g_gDepthRBO);
    if (g_gbufferFBO)   glDeleteFramebuffers(1, &g_gbufferFBO);

    if (g_geomProgram)  glDeleteProgram(g_geomProgram);
    if (g_lightProgram) glDeleteProgram(g_lightProgram);

    if (g_quadVBO)      glDeleteBuffers(1, &g_quadVBO);
    if (g_quadVAO)      glDeleteVertexArrays(1, &g_quadVAO);

    if (g_triceNormalTex) glDeleteTextures(1, &g_triceNormalTex);

    if (g_hdrColorTex)    glDeleteTextures(1, &g_hdrColorTex);
    if (g_brightColorTex) glDeleteTextures(1, &g_brightColorTex);
    if (g_hdrFBO)         glDeleteFramebuffers(1, &g_hdrFBO);
    glDeleteTextures(2, g_pingpongTex);
    glDeleteFramebuffers(2, g_pingpongFBO);

    if (g_blurProgram)    glDeleteProgram(g_blurProgram);
    if (g_finalProgram)   glDeleteProgram(g_finalProgram);

    if (g_ssaoTex)      glDeleteTextures(1, &g_ssaoTex);
    if (g_ssaoNoiseTex) glDeleteTextures(1, &g_ssaoNoiseTex);
    if (g_ssaoFBO)      glDeleteFramebuffers(1, &g_ssaoFBO);
    if (g_ssaoProgram)  glDeleteProgram(g_ssaoProgram);

    if (g_lightSphereEBO) glDeleteBuffers(1, &g_lightSphereEBO);
    if (g_lightSphereVBO) glDeleteBuffers(1, &g_lightSphereVBO);
    if (g_lightSphereVAO) glDeleteVertexArrays(1, &g_lightSphereVAO);

    if (g_areaRectEBO) glDeleteBuffers(1, &g_areaRectEBO);
    if (g_areaRectVBO) glDeleteBuffers(1, &g_areaRectVBO);
    if (g_areaRectVAO) glDeleteVertexArrays(1, &g_areaRectVAO);

    if (g_volumetricTex)   glDeleteTextures(1, &g_volumetricTex);
    if (g_volumetricFBO)   glDeleteFramebuffers(1, &g_volumetricFBO);
    if (g_volumetricProgram) glDeleteProgram(g_volumetricProgram);

    if (g_dirShadowTex) glDeleteTextures(1, &g_dirShadowTex);
    if (g_dirShadowFBO) glDeleteFramebuffers(1, &g_dirShadowFBO);
    if (g_dirDepthProgram) glDeleteProgram(g_dirDepthProgram);


    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
