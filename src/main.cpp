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


// ==============================
// Directional shadow mapping
// ==============================
GLuint g_dirShadowFBO = 0;
GLuint g_dirShadowTex = 0;
const int DIR_SHADOW_SIZE = 1024;

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
const float g_pointShadowFar = 10.0f;

glm::mat4 g_pointShadowMatrices[6];

// point light
//glm::vec3 g_lightEye = glm::vec3(1.87659f, 0.4625f, 0.103928f);
glm::vec3 g_pointLightPos = glm::vec3(1.87659f, 0.4625f, 0.103928f);
glm::vec3 g_lightCenter = glm::vec3(0.0f, 0.5f, 0.0f);
glm::vec3 g_lightUp = glm::vec3(0.0f, 1.0f, 0.0f);
float     g_lightNear = 0.1f;
float     g_lightFar = 10.0f;
float     g_lightRange = 5.0f;  // ortho box half-size

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

float g_edgeDepthThreshold = 0.08f; // tune
float g_edgeNormalThreshold = 0.35f; // tune
float g_edgeStrength = 1.0f;         // 0..1


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
// Shaders
// ==============================

static const char* kDirDepthVS = R"(#version 410 core
layout(location = 0) in vec3 a_pos;
uniform mat4 u_model;
uniform mat4 u_lightVP;
void main()
{
    gl_Position = u_lightVP * u_model * vec4(a_pos, 1.0);
}
)";

static const char* kDirDepthFS = R"(#version 410 core
void main() { }
)";

// Geometry pass: output world-pos, world-normal, ambient, diffuse, specular
static const char* kGeomVertexShader = R"(#version 410 core
layout(location = 0) in vec3 a_pos;
layout(location = 1) in vec3 a_norm;
layout(location = 2) in vec2 a_uv;

out vec3 v_worldPos;
out vec3 v_worldNorm;
out vec2 v_uv;

uniform mat4 u_model;
uniform mat4 u_view;
uniform mat4 u_proj;

void main()
{
    vec4 wp = u_model * vec4(a_pos, 1.0);
    v_worldPos  = wp.xyz;
    v_worldNorm = mat3(transpose(inverse(u_model))) * a_norm;
    v_uv        = a_uv;

    gl_Position = u_proj * u_view * wp;
}
)";

static const char* kGeomFragmentShader = R"(#version 410 core
in vec3 v_worldPos;
in vec3 v_worldNorm;
in vec2 v_uv;

layout(location = 0) out vec4 gPosition;  // world-space position
layout(location = 1) out vec4 gNormal;    // world-space normal
layout(location = 2) out vec4 gAmbient;   // ambient color
layout(location = 3) out vec4 gDiffuse;   // diffuse color
layout(location = 4) out vec4 gSpecular;  // specular.rgb, shininess.a

uniform vec3  u_Ka;
uniform vec3  u_Kd;
uniform vec3  u_Ks;
uniform float u_Ns;

uniform sampler2D u_diffuseTex;
uniform bool      u_useTex;

uniform sampler2D u_normalMap;
uniform bool      u_useNormalMap;

void main()
{
    vec3 Ka = u_Ka;
    vec3 Kd = u_Kd;
    if (u_useTex) {
        vec4 tex = texture(u_diffuseTex, v_uv);
        if (tex.a < 0.5)
            discard; // alpha cutout, e.g., plant
        Kd = tex.rgb;          // only diffuse comes from texture
        // Ka stays as material ambient u_Ka
    }

    // Base world-space normal
    vec3 N = normalize(v_worldNorm);

    // --- Normal mapping (Trice only when enabled) ---
    if (u_useNormalMap) {
        // Build TBN in world space using derivatives
        vec3 dp1  = dFdx(v_worldPos);
        vec3 dp2  = dFdy(v_worldPos);
        vec2 duv1 = dFdx(v_uv);
        vec2 duv2 = dFdy(v_uv);

        float r = 1.0 / (duv1.x * duv2.y - duv1.y * duv2.x);
        vec3 T = normalize((dp1 * duv2.y - dp2 * duv1.y) * r);
        vec3 B = normalize((dp2 * duv1.x - dp1 * duv2.x) * r);

        vec3 nTex = texture(u_normalMap, v_uv).xyz * 2.0 - 1.0; // tangent-space
        mat3 TBN  = mat3(T, B, N);
        N = normalize(TBN * nTex);  // perturbed world-space normal
    }

    gPosition = vec4(v_worldPos, 1.0);
    gNormal   = vec4(N, 0.0);
    gAmbient  = vec4(Ka, 1.0);   // now pure material ambient
    gDiffuse  = vec4(Kd, 1.0);
    gSpecular = vec4(u_Ks, u_Ns);
}
)";


// Lighting pass: fullscreen quad using G-buffers + shadow map
static const char* kLightVertexShader = R"(#version 410 core
layout(location = 0) in vec2 a_pos;
layout(location = 1) in vec2 a_uv;

out vec2 v_uv;

void main()
{
    v_uv = a_uv;
    gl_Position = vec4(a_pos, 0.0, 1.0);
}
)";

static const char* kLightFragmentShader = R"(#version 410 core
in vec2 v_uv;

// 0 = final lighting (HDR color), 1 = bright parts for bloom
layout(location = 0) out vec4 FragColor;
layout(location = 1) out vec4 BrightColor;

// G-buffers
uniform sampler2D gPosition;
uniform sampler2D gNormal;
uniform sampler2D gAmbient;
uniform sampler2D gDiffuse;
uniform sampler2D gSpecular;

// Camera & light
uniform vec3 u_eye;
uniform vec3 u_lightPos;   // point light position
uniform vec3 u_Ia;
uniform vec3 u_Id;
uniform vec3 u_Is;

// Point light attenuation (constant, linear, quadratic)
uniform float u_attConst;
uniform float u_attLinear;
uniform float u_attQuadratic;

// ===== Point shadow cube =====
uniform samplerCube u_shadowCube;
uniform float       u_far;

// ===== Directional shadow map =====
uniform sampler2D u_dirShadowMap;
uniform mat4      u_lightVP;       // light view-proj
uniform vec3      u_lightDir;      // direction FROM surface TO light (world)
uniform float     u_shadowStrength;
uniform bool      u_enableShadows;

// Bloom
uniform bool  u_enableBloom;
uniform float u_bloomThreshold;

// SSAO
uniform bool      u_enableSSAO;
uniform sampler2D u_ssaoTex;

// ===== SSR =====
uniform bool  u_enableSSR;
uniform mat4  u_view;
uniform mat4  u_proj;
uniform float u_ssrMaxDistance;
uniform int   u_ssrMaxSteps;
uniform float u_ssrStep;
uniform float u_ssrThickness;
uniform float u_ssrIntensity;

// Rectangular area light
uniform vec3 u_rectCenter;
uniform vec3 u_rectNormal;
uniform vec3 u_rectTangent;
uniform vec3 u_rectBitangent;
uniform vec2 u_rectSize;       // full width/height
uniform vec3 u_rectColor;
uniform int  u_rectSamples;
uniform bool u_enableAreaLight;

// Debug view mode:
// 0 = lighting, 1 = pos, 2 = normal, 3 = ambient,
// 4 = diffuse, 5 = specular, 6 = SSAO
uniform int u_viewMode;

// ===== Toon / NPR =====
uniform bool u_enableToon;
uniform int  u_toonSteps;

uniform bool  u_enableEdges;
uniform float u_edgeDepthThreshold;
uniform float u_edgeNormalThreshold;
uniform float u_edgeStrength;

// ------------------------------------------------------------
// Helpers
// ------------------------------------------------------------
float toonQuantize(float x, int steps)
{
    steps = max(steps, 1);
    x = clamp(x, 0.0, 1.0);
    float s = float(max(steps - 1, 1));
    return floor(x * s + 0.5) / s;
}

// For toon highlight (optional “banded” spec)
float toonSpec(float specPow)
{
    // specPow is already in [0,1] usually, but clamp anyway
    float s = clamp(specPow, 0.0, 1.0);

    // 2-band highlight: adjust thresholds to taste
    // You can also do multiple bands with toonQuantize().
    float hi = step(0.65, s);       // hard highlight
    float mid = step(0.35, s) * (1.0 - hi); // mid highlight
    return 0.0 + 0.5 * mid + 1.0 * hi;
}

// Sobel edge detection using depth + normal discontinuities
float sobelEdge(vec2 uv)
{
    vec2 texel = 1.0 / vec2(textureSize(gNormal, 0));

    vec2 o[9] = vec2[](
        vec2(-1,-1), vec2(0,-1), vec2(1,-1),
        vec2(-1, 0), vec2(0, 0), vec2(1, 0),
        vec2(-1, 1), vec2(0, 1), vec2(1, 1)
    );

    float kx[9] = float[](
        -1, 0, 1,
        -2, 0, 2,
        -1, 0, 1
    );
    float ky[9] = float[](
        -1,-2,-1,
         0, 0, 0,
         1, 2, 1
    );

    float gxD = 0.0, gyD = 0.0;
    vec3  gxN = vec3(0.0);
    vec3  gyN = vec3(0.0);

    for (int i = 0; i < 9; ++i)
    {
        vec2 suv = uv + o[i] * texel;

        vec3 p = texture(gPosition, suv).xyz;
        vec3 n = texture(gNormal,   suv).xyz;

        float d = (p == vec3(0.0)) ? 1e6 : length(p - u_eye);

        gxD += d * kx[i];
        gyD += d * ky[i];

        gxN += n * kx[i];
        gyN += n * ky[i];
    }

    float magD = length(vec2(gxD, gyD));
    float magN = length(gxN) + length(gyN);

    float eD = smoothstep(u_edgeDepthThreshold,  u_edgeDepthThreshold * 2.0, magD);
    float eN = smoothstep(u_edgeNormalThreshold, u_edgeNormalThreshold * 2.0, magN);

    return clamp(max(eD, eN), 0.0, 1.0);
}

// ---------- Directional shadow (2D) ----------
float calcDirShadow(vec3 worldPos, vec3 N)
{
    vec4 lp = u_lightVP * vec4(worldPos, 1.0);
    vec3 ndc = lp.xyz / lp.w;
    vec3 sc  = ndc * 0.5 + 0.5;

    if (sc.x < 0.0 || sc.x > 1.0 || sc.y < 0.0 || sc.y > 1.0 || sc.z < 0.0 || sc.z > 1.0)
        return 0.0;

    float NdotL = max(dot(N, u_lightDir), 0.0);
    float bias  = max(0.0025 * (1.0 - NdotL), 0.0005);

    vec2 texelSize = 1.0 / vec2(textureSize(u_dirShadowMap, 0));
    float shadow = 0.0;

    for (int y = -1; y <= 1; ++y)
    for (int x = -1; x <= 1; ++x)
    {
        float pcfDepth = texture(u_dirShadowMap, sc.xy + vec2(x,y) * texelSize).r;
        shadow += (sc.z - bias > pcfDepth) ? 1.0 : 0.0;
    }
    shadow /= 9.0;
    return shadow;
}

// ---------- Point shadow (cube) ----------
float calcPointShadow(vec3 worldPos)
{
    vec3 L = worldPos - u_lightPos;
    float currentDepth = length(L);
    float closestDepth = texture(u_shadowCube, L).r * u_far;

    float bias = 0.03;
    return (currentDepth - bias > closestDepth) ? 1.0 : 0.0;
}

// ---------- Area light (diffuse only, stable in HDR) ----------
vec3 evalRectAreaLightDiffuse(vec3 pos, vec3 N, vec3 Kd)
{
    const int MAX_SAMPLES = 32;
    int S = clamp(u_rectSamples, 1, MAX_SAMPLES);

    float halfW = 0.5 * u_rectSize.x;
    float halfH = 0.5 * u_rectSize.y;
    float area  = u_rectSize.x * u_rectSize.y;

    int grid = int(ceil(sqrt(float(S))));
    int used = 0;

    float accumG = 0.0;

    for (int j = 0; j < grid && used < S; ++j)
    for (int i = 0; i < grid && used < S; ++i, ++used)
    {
        float u = (float(i) + 0.5) / float(grid) - 0.5;
        float v = (float(j) + 0.5) / float(grid) - 0.5;

        vec3 samplePos =
            u_rectCenter +
            (u * 2.0 * halfW) * u_rectTangent +
            (v * 2.0 * halfH) * u_rectBitangent;

        vec3 Lvec = samplePos - pos;
        float d2  = dot(Lvec, Lvec);
        if (d2 <= 0.0) continue;

        vec3 wi = normalize(Lvec);

        float NdotL = max(dot(N, wi), 0.0);
        if (NdotL <= 0.0) continue;

        // one-sided emitter: only if sample is visible from front
        float NL_light = max(dot(-u_rectNormal, wi), 0.0);
        if (NL_light <= 0.0) continue;

        float G = (NdotL * NL_light) / d2;

        // Toon: quantize N·L contribution (NOT the final color)
        if (u_enableToon) {
            // Use the NdotL term; keep geometric term continuous
            float q = toonQuantize(NdotL, u_toonSteps);
            G *= (q / max(NdotL, 1e-6));
        }

        accumG += G;
    }

    if (used == 0) return vec3(0.0);

    float factor  = area / float(used);
    vec3 radiance = u_rectColor * accumG * factor;

    return Kd * radiance;
}

// ---------- SSR helpers ----------
bool isFloorPixel(vec3 pos, vec3 normal)
{
    float epsY = 0.2;
    bool nearPlane = (pos.y > -epsY && pos.y < epsY);
    bool normalUp  = normal.y > 0.8;
    return nearPlane && normalUp;
}

vec3 computeSSR(vec3 pos, vec3 normal, vec3 Ka, vec3 Kd, vec3 Ks, float NsRaw)
{
    vec3 N = normalize(normal);
    vec3 V = normalize(u_eye - pos);
    vec3 R = reflect(-V, N);

    vec3 rayOrigin = pos + N * 0.02;

    float t = 0.1;

    for (int i = 0; i < u_ssrMaxSteps; ++i)
    {
        if (t > u_ssrMaxDistance) break;

        vec3 samplePos = rayOrigin + R * t;

        vec4 clip = u_proj * u_view * vec4(samplePos, 1.0);
        if (clip.w <= 0.0) { t += u_ssrStep; continue; }

        vec3 ndc = clip.xyz / clip.w;
        if (ndc.x < -1.0 || ndc.x > 1.0 || ndc.y < -1.0 || ndc.y > 1.0) {
            t += u_ssrStep; continue;
        }

        vec2 uv = ndc.xy * 0.5 + 0.5;

        vec3 scenePos = texture(gPosition, uv).xyz;
        if (scenePos == vec3(0.0)) { t += u_ssrStep; continue; }

        vec3 sceneNormal = texture(gNormal, uv).xyz;
        if (isFloorPixel(scenePos, sceneNormal)) { t += u_ssrStep; continue; }

        float rayDepth   = length(samplePos - u_eye);
        float sceneDepth = length(scenePos  - u_eye);

        if (abs(sceneDepth - rayDepth) < u_ssrThickness)
        {
            vec3 hitKa   = texture(gAmbient, uv).rgb;
            vec3 hitKd   = texture(gDiffuse, uv).rgb;
            vec4 hitSpec = texture(gSpecular, uv);
            vec3 hitKs   = hitSpec.rgb;
            float hitNsRaw = hitSpec.a;

            bool hitEmissive = (hitNsRaw < 0.0);
            float hitNs = max(hitNsRaw, 1.0);

            if (hitEmissive) {
                return hitKd;
            } else {
                vec3 hN = normalize(sceneNormal);
                vec3 Ld = normalize(u_lightDir);
                vec3 V2 = normalize(u_eye - scenePos);
                vec3 H2 = normalize(Ld + V2);

                float NdotL2 = max(dot(hN, Ld), 0.0);
                float NdotH2 = max(dot(hN, H2), 0.0);

                float ao2 = 1.0;
                if (u_enableSSAO) ao2 = texture(u_ssaoTex, uv).r;

                // Keep SSR lighting non-toon to avoid “banding in reflections”
                vec3 ambient2  = (hitKa * hitKd) * u_Ia * ao2;
                vec3 diffuse2  = u_Id * hitKd * NdotL2;
                vec3 specular2 = (NdotL2 > 0.0) ? (u_Is * hitKs * pow(NdotH2, hitNs)) : vec3(0.0);

                return ambient2 + diffuse2 + specular2;
            }
        }

        t += u_ssrStep;
    }

    return vec3(0.0);
}

// ------------------------------------------------------------
// main
// ------------------------------------------------------------
void main()
{
    vec3 pos      = texture(gPosition, v_uv).xyz;
    vec3 normal   = texture(gNormal,   v_uv).xyz;
    vec3 Ka       = texture(gAmbient,  v_uv).rgb;
    vec3 Kd       = texture(gDiffuse,  v_uv).rgb;
    vec4 specData = texture(gSpecular, v_uv);
    vec3 Ks       = specData.rgb;
    float NsRaw   = specData.a;

    bool  isEmissive = (NsRaw < 0.0);
    float Ns = max(NsRaw, 1.0);

    if (pos == vec3(0.0)) {
        FragColor   = vec4(0.0);
        BrightColor = vec4(0.0);
        return;
    }

    // Debug views
    if (u_viewMode == 1) { FragColor = vec4(normalize(pos) * 0.5 + 0.5, 1.0); BrightColor = vec4(0.0); return; }
    if (u_viewMode == 2) { FragColor = vec4(normalize(normal) * 0.5 + 0.5, 1.0); BrightColor = vec4(0.0); return; }
    if (u_viewMode == 3) { FragColor = vec4(Ka, 1.0); BrightColor = vec4(0.0); return; }
    if (u_viewMode == 4) { FragColor = vec4(Kd, 1.0); BrightColor = vec4(0.0); return; }
    if (u_viewMode == 5) { FragColor = vec4(Ks, 1.0); BrightColor = vec4(0.0); return; }
    if (u_viewMode == 6) {
        float aoDbg = u_enableSSAO ? texture(u_ssaoTex, v_uv).r : 1.0;
        FragColor = vec4(vec3(aoDbg), 1.0); BrightColor = vec4(0.0); return;
    }

    // Emissive (area rect / light spheres)
    if (isEmissive) {
        vec3 emissive = Kd;
        FragColor = vec4(emissive, 1.0);

        vec3 bright = vec3(0.0);
        if (u_enableBloom) {
            float b = max(max(emissive.r, emissive.g), emissive.b);
            if (b > u_bloomThreshold) bright = emissive;
        }
        BrightColor = vec4(bright, 1.0);
        return;
    }

    vec3 N = normalize(normal);
    vec3 V = normalize(u_eye - pos);

    float ao = 1.0;
    if (u_enableSSAO) ao = texture(u_ssaoTex, v_uv).r;

    // Ambient (keep smooth even in toon)
    vec3 ambient = (Ka * Kd) * u_Ia * ao;

    float s = clamp(u_shadowStrength, 0.0, 1.0);

    // --------------------------------------------------------
    // Point light
    // --------------------------------------------------------
    vec3 Lp = normalize(u_lightPos - pos);
    vec3 Hp = normalize(Lp + V);

    float NdotLp = max(dot(N, Lp), 0.0);
    float qLp    = u_enableToon ? toonQuantize(NdotLp, u_toonSteps) : NdotLp;

    vec3 pointDiffuse = u_Id * Kd * qLp;

    // Spec: normal or toon-banded
    vec3 pointSpecular = vec3(0.0);
    if (NdotLp > 0.0) {
        float specPow = pow(max(dot(N, Hp), 0.0), Ns);
        float specAmt = u_enableToon ? toonSpec(specPow) : specPow;
        pointSpecular = u_Is * Ks * specAmt;
    }

    vec3 directPoint = pointDiffuse + pointSpecular;

    float dist = length(u_lightPos - pos);
    float attenuation = 1.0 / (u_attConst + u_attLinear * dist + u_attQuadratic * dist * dist);
    directPoint *= attenuation;

    float pointShadow = 0.0;
    if (u_enableShadows) pointShadow = calcPointShadow(pos);
    vec3 pointWithShadow = directPoint * mix(s, 1.0, 1.0 - pointShadow);

    // --------------------------------------------------------
    // Directional light
    // --------------------------------------------------------
    vec3 Ld = normalize(u_lightDir);
    vec3 Hd = normalize(Ld + V);

    float NdotLd = max(dot(N, Ld), 0.0);
    float qLd    = u_enableToon ? toonQuantize(NdotLd, u_toonSteps) : NdotLd;

    vec3 dirDiffuse = u_Id * Kd * qLd;

    vec3 dirSpecular = vec3(0.0);
    if (NdotLd > 0.0) {
        float specPow = pow(max(dot(N, Hd), 0.0), Ns);
        float specAmt = u_enableToon ? toonSpec(specPow) : specPow;
        dirSpecular = u_Is * Ks * specAmt;
    }

    vec3 directDir = dirDiffuse + dirSpecular;

    float dirShadow = 0.0;
    if (u_enableShadows) dirShadow = calcDirShadow(pos, N);
    vec3 dirWithShadow = directDir * mix(s, 1.0, 1.0 - dirShadow);

    // --------------------------------------------------------
    // Rect area light (diffuse only)
    // --------------------------------------------------------
    vec3 directArea = vec3(0.0);
    if (u_enableAreaLight) {
        directArea = evalRectAreaLightDiffuse(pos, N, Kd) * u_Id;
    }

    // Final lighting (HDR)
    vec3 lighting = ambient + pointWithShadow + dirWithShadow + directArea;

    // Edge darkening (post-light)
    if (u_enableEdges) {
        float e = sobelEdge(v_uv);
        lighting *= (1.0 - e * u_edgeStrength);
    }

    // SSR (apply after edges; don’t toon-quantize SSR itself)
    if (u_enableSSR && isFloorPixel(pos, normal)) {
        vec3 refl = computeSSR(pos, normal, Ka, Kd, Ks, NsRaw);
        lighting = mix(lighting, refl, u_ssrIntensity);
    }

    // Bloom bright-pass (use HDR lighting)
    vec3 bright = vec3(0.0);
    if (u_enableBloom) {
        float b = max(max(lighting.r, lighting.g), lighting.b);
        if (b > u_bloomThreshold) bright = lighting;
    }

    FragColor   = vec4(lighting, 1.0);
    BrightColor = vec4(bright, 1.0);
}
)";




// Gaussian blur for bloom (ping-pong)
static const char* kBlurFragmentShader = R"(#version 410 core
in vec2 v_uv;
out vec4 FragColor;

uniform sampler2D u_image;
uniform bool      u_horizontal;

void main()
{
    float weight[5] = float[](0.227027, 0.1945946, 0.1216216, 0.054054, 0.016216);
    vec2 texelSize = 1.0 / vec2(textureSize(u_image, 0));

    vec3 result = texture(u_image, v_uv).rgb * weight[0];
    for (int i = 1; i < 5; ++i) {
        vec2 offset = u_horizontal
            ? vec2(texelSize.x * float(i), 0.0)
            : vec2(0.0, texelSize.y * float(i));

        result += texture(u_image, v_uv + offset).rgb * weight[i];
        result += texture(u_image, v_uv - offset).rgb * weight[i];
    }
    FragColor = vec4(result, 1.0);
}
)";

// Final combine: HDR scene + blurred bloom, tone mapping & gamma
static const char* kFinalFragmentShader = R"(#version 410 core
in vec2 v_uv;
out vec4 FragColor;

uniform sampler2D u_scene;
uniform sampler2D u_bloomBlur;
uniform sampler2D u_volumetric;

uniform bool  u_enableBloom;
uniform bool  u_enableVolumetric;
uniform float u_bloomIntensity;
uniform float u_volumetricIntensity;
uniform float u_exposure;


void main()
{
    vec3 hdrColor   = texture(u_scene,     v_uv).rgb;
    vec3 bloomColor = texture(u_bloomBlur, v_uv).rgb;
    vec3 volColor   = vec3(0.0);

    if (u_enableVolumetric) {
        volColor = texture(u_volumetric, v_uv).rgb * u_volumetricIntensity;
    }

    // Treat volumetric as extra HDR light before tone mapping
    hdrColor += volColor;

    if (u_enableBloom) {
        hdrColor += bloomColor * u_bloomIntensity;
    }

    // simple exponential tone mapping
    vec3 mapped = vec3(1.0) - exp(-hdrColor * u_exposure);

    // gamma correction
    FragColor = vec4(mapped, 1.0);
}
)";



// Depth-only pass for shadow map
static const char* kDepthVertexShader = R"(#version 410 core
layout(location = 0) in vec3 a_pos;

out vec3 v_worldPos;

uniform mat4 u_model;
uniform mat4 u_lightVP;   // per-face view-proj matrix

void main()
{
    vec4 wp = u_model * vec4(a_pos, 1.0);
    v_worldPos = wp.xyz;
    gl_Position = u_lightVP * wp;
}
)";


static const char* kDepthFragmentShader = R"(#version 410 core
in vec3 v_worldPos;

uniform vec3  u_lightPos;
uniform float u_far;

void main()
{
    // store radial distance / far in the cube depth
    float dist = length(v_worldPos - u_lightPos);
    dist = dist / u_far;          // 0..1
    gl_FragDepth = dist;
}
)";

// SSAO pass: compute ambient occlusion factor into a single-channel texture
static const char* kSSAOFragmentShader = R"(#version 410 core
in vec2 v_uv;
out float FragColor;

uniform sampler2D gPosition;  // world-space position
uniform sampler2D gNormal;    // world-space normal
uniform sampler2D texNoise;

uniform vec3  u_samples[64];
uniform mat4  u_view;
uniform mat4  u_proj;
uniform float u_radius;
uniform float u_bias;
uniform vec2  u_noiseScale;

void main()
{
    vec3 fragPos = texture(gPosition, v_uv).xyz;
    vec3 normal  = normalize(texture(gNormal,   v_uv).xyz);

    // Skip empty pixels (no geometry)
    if (fragPos == vec3(0.0)) {
        FragColor = 1.0;
        return;
    }

    // TBN from random rotation + normal
    vec3 randomVec = normalize(texture(texNoise, v_uv * u_noiseScale).xyz);
    vec3 tangent   = normalize(randomVec - normal * dot(randomVec, normal));
    vec3 bitangent = cross(normal, tangent);
    mat3 TBN       = mat3(tangent, bitangent, normal);

    float occlusion = 0.0;

    for (int i = 0; i < 64; ++i) {
        // sample position in world space
        vec3 samplePos = TBN * u_samples[i];
        samplePos = fragPos + samplePos * u_radius;

        // project sample position into screen space
        vec4 offset = u_proj * u_view * vec4(samplePos, 1.0);
        offset.xyz /= offset.w;
        offset.xyz = offset.xyz * 0.5 + 0.5;

        // outside screen?
        if (offset.x < 0.0 || offset.x > 1.0 ||
            offset.y < 0.0 || offset.y > 1.0)
            continue;

        // position at that screen sample
        vec3 sampleFragPos = texture(gPosition, offset.xy).xyz;

        // depth along the view direction, approximated by distances
        float sampleDist   = length(samplePos      - fragPos);
        float realDist     = length(sampleFragPos  - fragPos);

        float rangeCheck = smoothstep(0.0, 1.0, u_radius / abs(realDist - sampleDist));

        if (realDist < sampleDist - u_bias)
            occlusion += rangeCheck;
    }

    float ao = 1.0 - (occlusion / 64.0);
    ao = clamp(ao, 0.0, 1.0);

    // boost contrast so contact areas get much darker
    ao = pow(ao, 2.5);

    FragColor = ao;
}
)";


// Volumetric light scattering ("god rays") post-process
static const char* kVolumetricFragmentShader = R"(#version 410 core
in vec2 v_uv;
out vec4 FragColor;

// This is now the *bright-pass / occlusion* texture
uniform sampler2D u_scene;          // g_brightColorTex

uniform vec2  u_lightScreenPos;     // [0,1] screen-space
uniform int   u_numSamples;         // e.g. 100
uniform float u_exposure;           // 0.0 ~ 1.0
uniform float u_decay;              // ~0.96815
uniform float u_density;           // ~0.926
uniform float u_weight;             // ~0.58767

// Optional: threshold on the occlusion buffer to ignore tiny noise
uniform float u_threshold;          // you already have this uniform

void main()
{
    // If light is off-screen, no rays
    if (u_lightScreenPos.x < 0.0 || u_lightScreenPos.x > 1.0 ||
        u_lightScreenPos.y < 0.0 || u_lightScreenPos.y > 1.0)
    {
        FragColor = vec4(0.0);
        return;
    }

    vec2 texCoord      = v_uv;
    vec2 delta         = (texCoord - u_lightScreenPos);
    delta             *= (u_density / float(u_numSamples));

    vec2  sampleCoord        = texCoord;
    float illuminationDecay  = 1.0;
    float accumulatedLumin   = 0.0;

    for (int i = 0; i < u_numSamples; ++i)
    {
        sampleCoord -= delta;

        // Sample the bright-pass/occlusion buffer
        vec3 c = texture(u_scene, sampleCoord).rgb;

        // Convert to a single scalar (luminance or just max component)
        float lum = max(max(c.r, c.g), c.b);

        // Optional threshold to kill tiny noise
        if (lum > u_threshold)
        {
            lum *= illuminationDecay * u_weight;
            accumulatedLumin += lum;
        }

        illuminationDecay *= u_decay;
    }

    float intensity = accumulatedLumin * u_exposure;
    FragColor = vec4(vec3(intensity), 1.0);
}
)";







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
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
        GL_TEXTURE_2D, g_hdrColorTex, 0);

    // Bright color (HDR)
    glGenTextures(1, &g_brightColorTex);
    glBindTexture(GL_TEXTURE_2D, g_brightColorTex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, width, height, 0,
        GL_RGBA, GL_FLOAT, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
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
static void renderShadowPass(const glm::vec3& lightPos)
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
            g_shadowTex, 0);
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
    int display_w, display_h;
    glfwGetFramebufferSize(window, &display_w, &display_h);

    // Resize G-buffer if needed
    if (display_w != g_gbufferWidth || display_h != g_gbufferHeight) {
        initGBuffer(display_w, display_h);
    }

    float aspect = (display_h > 0) ? (float)display_w / (float)display_h : 1.0f;
    glm::mat4 proj = glm::perspective(glm::radians(g_fov), aspect, 0.1f, 100.0f);
    glm::mat4 view = glm::lookAt(g_eye, g_center, g_up);

    glm::vec3 lightPos = g_pointLightPos; // point light position follows sphere

    // 1) Shadow pass (point light cube)
    renderDirectionalShadowPass();

    renderShadowPass(lightPos);

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
    glBindTexture(GL_TEXTURE_CUBE_MAP, g_shadowTex);
    glUniform1i(glGetUniformLocation(g_lightProgram, "u_shadowCube"), 5);
    glUniform1f(glGetUniformLocation(g_lightProgram, "u_far"), g_pointShadowFar);

    // SSAO texture
    glActiveTexture(GL_TEXTURE6);
    glBindTexture(GL_TEXTURE_2D, g_enableSSAO ? g_ssaoTex : 0);
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

    glm::vec3 volLightPos = g_useDemoVolLight ? g_volDemoLightPos : lightPos;


    if (doVol)
    {
        // Use the demo sun when the checkbox is on,
        // otherwise fall back to the point light sphere
        glm::vec3 volLightWorld = g_useDemoVolLight ? g_volDemoLightPos : lightPos;

        glm::vec4 clip = proj * view * glm::vec4(volLightWorld, 1.0f);
        if (clip.w <= 0.0f) {
            doVol = false;
        }
        else {
            glm::vec3 ndc = glm::vec3(clip) / clip.w;
            lightScreenPos = glm::vec2(
                ndc.x * 0.5f + 0.5f,
                ndc.y * 0.5f + 0.5f
            );
            if (lightScreenPos.x < 0.0f || lightScreenPos.x > 1.0f ||
                lightScreenPos.y < 0.0f || lightScreenPos.y > 1.0f)
            {
                doVol = false;
            }
        }
    }

    if (doVol) {
        glBindFramebuffer(GL_FRAMEBUFFER, g_volumetricFBO);
        glViewport(0, 0, display_w, display_h);
        glDisable(GL_DEPTH_TEST);
        glClear(GL_COLOR_BUFFER_BIT);

        glUseProgram(g_volumetricProgram);
                                                               
        // IMPORTANT: now the scene buffer is from *this* frame
        glActiveTexture(GL_TEXTURE0);
        //glBindTexture(GL_TEXTURE_2D, g_brightColorTex);
        glBindTexture(GL_TEXTURE_2D, g_hdrColorTex);

        glUniform1i(glGetUniformLocation(g_volumetricProgram, "u_scene"), 0);

        glUniform2fv(glGetUniformLocation(g_volumetricProgram, "u_lightScreenPos"),
            1, glm::value_ptr(lightScreenPos));
        glUniform1i(glGetUniformLocation(g_volumetricProgram, "u_numSamples"), g_volNumSamples);
        glUniform1f(glGetUniformLocation(g_volumetricProgram, "u_exposure"), g_volExposure);
        glUniform1f(glGetUniformLocation(g_volumetricProgram, "u_decay"), g_volDecay);
        glUniform1f(glGetUniformLocation(g_volumetricProgram, "u_density"), g_volDensity);
        glUniform1f(glGetUniformLocation(g_volumetricProgram, "u_weight"), g_volWeight);

        //glUniform1f(glGetUniformLocation(g_volumetricProgram, "u_threshold"), 0.6f);

        // Background key color
        glm::vec3 bg(0.19f, 0.19f, 0.19f);  // from the spec
        glUniform3fv(glGetUniformLocation(g_volumetricProgram, "u_bgColor"),
            1, glm::value_ptr(bg));


        glUniform1f(glGetUniformLocation(g_volumetricProgram, "u_threshold"), g_volThreshold);
        /*glUniform1f(glGetUniformLocation(g_volumetricProgram, "u_sourceRadius"), g_volSourceRadius);*/

        glBindVertexArray(g_quadVAO);
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
        glBindVertexArray(0);

        glBindFramebuffer(GL_FRAMEBUFFER, 0);
    }

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
        g_enableVolumetric ? 1 : 0);


    glUniform1f(glGetUniformLocation(g_finalProgram, "u_bloomIntensity"),
        g_bloomIntensity);

    glUniform1f(glGetUniformLocation(g_finalProgram, "u_volumetricIntensity"),
        g_volIntensity);

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
    ImGui::Text("Point Light");
    ImGui::DragFloat3("Point Pos", &g_pointLightPos.x, 0.05f);
    ImGui::DragFloat3("Light Center", glm::value_ptr(g_lightCenter), 0.05f);
    ImGui::SliderFloat("Light Range", &g_lightRange, 1.0f, 10.0f);

    ImGui::Separator();
    ImGui::Text("Directional Light Camera (2D shadow)");
    ImGui::DragFloat3("Dir Eye", &g_dirLightEye.x, 0.05f);
    ImGui::DragFloat3("Dir Center", &g_dirLightCenter.x, 0.05f);
    ImGui::DragFloat3("Dir Up", &g_dirLightUp.x, 0.05f);

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
    ImGui::DragFloat3("Center", &g_areaCenter.x, 0.01f);
    ImGui::DragFloat2("Size (W,H)", &g_areaSize.x, 0.01f, 0.01f, 5.0f);
    ImGui::DragFloat3("Euler (pitch,yaw,roll)", &g_areaEuler.x, 0.5f, -180.0f, 180.0f);
    ImGui::ColorEdit3("Area Color", &g_areaColor.x);
    ImGui::SliderInt("Samples", &g_areaSamples, 1, 32);

    ImGui::Separator();
    ImGui::Text("Volumetric Light (God Rays)");
    ImGui::Checkbox("Enable Volumetric", &g_enableVolumetric);
    ImGui::SliderFloat("Vol Intensity", &g_volIntensity, 0.0f, 5.0f);
    ImGui::SliderInt("Vol Samples", &g_volNumSamples, 10, 200);
    ImGui::SliderFloat("Vol Exposure", &g_volExposure, 0.0f, 1.0f);
    ImGui::SliderFloat("Vol Decay", &g_volDecay, 0.90f, 1.0f);
    ImGui::SliderFloat("Vol Density", &g_volDensity, 0.1f, 2.0f);
    ImGui::SliderFloat("Vol Weight", &g_volWeight, 0.0f, 2.0f);
    // NEW: demo sun light
    ImGui::Separator();
    ImGui::Text("Volumetric Demo Light");
    ImGui::Checkbox("Use demo light pos", &g_useDemoVolLight);
    ImGui::Checkbox("Show demo light sphere", &g_showVolDemoSphere);
    ImGui::DragFloat3("Demo light world pos", &g_volDemoLightPos.x, 0.05f);
    ImGui::SliderFloat("Vol Source radius", &g_volSourceRadius, 0.0f, 0.3f);
    ImGui::SliderFloat("Vol Threshold", &g_volThreshold, 0.0f, 2.0f);

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
    g_geomProgram = createProgram(kGeomVertexShader, kGeomFragmentShader);
    g_lightProgram = createProgram(kLightVertexShader, kLightFragmentShader);
    g_blurProgram = createProgram(kLightVertexShader, kBlurFragmentShader);
    g_finalProgram = createProgram(kLightVertexShader, kFinalFragmentShader);
    g_depthProgram = createProgram(kDepthVertexShader, kDepthFragmentShader);

    g_dirDepthProgram = createProgram(kDirDepthVS, kDirDepthFS);

    g_ssaoProgram = createProgram(kLightVertexShader, kSSAOFragmentShader);
    g_volumetricProgram = createProgram(kLightVertexShader, kVolumetricFragmentShader);


    initSSAOKernelAndNoise();
    initShadowMap();
    initDirectionalShadowMap();
    initFullscreenQuad();
    initLightSphere();
    initAreaRectMesh();

    int fbw, fbh;
    glfwGetFramebufferSize(window, &fbw, &fbh);
    initGBuffer(fbw, fbh);

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
