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

// Shadow mapping globals
GLuint g_shadowFBO = 0;
GLuint g_shadowTex = 0;
const int SHADOW_MAP_SIZE = 1024;

const float g_pointShadowNear = 0.22f;
const float g_pointShadowFar = 10.0f;

glm::mat4 g_pointShadowMatrices[6];

// Directional light camera (from spec)
glm::vec3 g_lightEye = glm::vec3(1.87659f, 0.4625f, 0.103928f);
glm::vec3 g_lightCenter = glm::vec3(0.0f, 0.5f, 0.0f);
glm::vec3 g_lightUp = glm::vec3(0.0f, 1.0f, 0.0f);
float     g_lightNear = 0.1f;
float     g_lightFar = 10.0f;
float     g_lightRange = 5.0f;  // ortho box half-size

GLuint g_lightSphereVAO = 0;
GLuint g_lightSphereVBO = 0;
GLuint g_lightSphereEBO = 0;
GLsizei g_lightSphereIndexCount = 0;

const float g_lightSphereRadius = 0.22f;  // from assignment

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
float  g_bloomThreshold = 1.0f;
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
uniform vec3 u_lightPos;
uniform vec3 u_Ia;
uniform vec3 u_Id;
uniform vec3 u_Is;

// Point light attenuation (constant, linear, quadratic)
uniform float u_attConst;
uniform float u_attLinear;
uniform float u_attQuadratic;

// Point light shadow cube
uniform samplerCube u_shadowCube;
uniform float       u_far;
uniform float       u_shadowStrength;
uniform bool        u_enableShadows;

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

// Debug view mode:
// 0 = lighting, 1 = pos, 2 = normal, 3 = ambient,
// 4 = diffuse, 5 = specular
uniform int u_viewMode;

bool isFloorPixel(vec3 pos, vec3 normal)
{
    // Your floor is basically at y = 0 with upward normal
    // Adjust eps if needed.
    float epsY = 0.2;
    bool nearPlane = (pos.y > -epsY && pos.y < epsY);
    bool normalUp  = normal.y > 0.8;
    return nearPlane && normalUp;
}

vec3 computeSSR(vec3 pos, vec3 normal,
                vec3 Ka, vec3 Kd, vec3 Ks, float NsRaw)
{
    // Ray origin & direction in world space
    vec3 N = normalize(normal);
    vec3 V = normalize(u_eye - pos);      // view direction (towards eye)
    vec3 R = reflect(-V, N);              // reflection direction (into scene)

    vec3 rayOrigin = pos + N * 0.02;      // small bias

    float t = 0.1;
    vec3 hitColor = vec3(0.0);
    bool hit = false;

    for (int i = 0; i < u_ssrMaxSteps; ++i) {
        if (t > u_ssrMaxDistance)
            break;

        vec3 samplePos = rayOrigin + R * t;

        // project samplePos to screen to get UV
        vec4 clip = u_proj * u_view * vec4(samplePos, 1.0);
        if (clip.w <= 0.0) {
            t += u_ssrStep;
            continue;
        }

        vec3 ndc = clip.xyz / clip.w;
        // outside screen?
        if (ndc.x < -1.0 || ndc.x > 1.0 ||
            ndc.y < -1.0 || ndc.y > 1.0) {
            t += u_ssrStep;
            continue;
        }

        vec2 uv = ndc.xy * 0.5 + 0.5;

        vec3 scenePos = texture(gPosition, uv).xyz;
        if (scenePos == vec3(0.0)) {
            t += u_ssrStep;
            continue;
        }

        // Reject self-hits on the floor itself (prevents banding)
        vec3 sceneNormal = texture(gNormal, uv).xyz;
        if (isFloorPixel(scenePos, sceneNormal)) {
            t += u_ssrStep;
            continue;
        }

        // Compare depths from the camera instead of from the ray origin
        float rayDepth   = length(samplePos - u_eye);
        float sceneDepth = length(scenePos - u_eye);

        // depth intersection test (screen-space ray vs scene)
        if (abs(sceneDepth - rayDepth) < u_ssrThickness) {
            // Hit: fetch G-buffer data at that point and re-compute lighting
            vec3 hitNormal = sceneNormal;
            vec3 hitKa     = texture(gAmbient, uv).rgb;
            vec3 hitKd     = texture(gDiffuse, uv).rgb;
            vec4 hitSpec   = texture(gSpecular, uv);
            vec3 hitKs     = hitSpec.rgb;
            float hitNsRaw = hitSpec.a;
            bool  hitEmissive = (hitNsRaw < 0.0);
            float hitNs   = max(hitNsRaw, 1.0);

            if (hitEmissive) {
                hitColor = hitKd;   // emissive: just use color
            } else {
                vec3 hN = normalize(hitNormal);
                vec3 L  = normalize(u_lightPos - scenePos);
                vec3 V2 = normalize(u_eye      - scenePos);
                vec3 H2 = normalize(L + V2);

                float NdotL2 = max(dot(hN, L), 0.0);
                float NdotH2 = max(dot(hN, H2), 0.0);

                float ao2 = 1.0;
                if (u_enableSSAO) {
                    ao2 = texture(u_ssaoTex, uv).r;
                }

                vec3 ambient2  = (hitKa * hitKd) * u_Ia * ao2;
                vec3 diffuse2  = u_Id * hitKd * NdotL2;
                vec3 specular2 = (NdotL2 > 0.0)
                    ? (u_Is * hitKs * pow(NdotH2, hitNs))
                    : vec3(0.0);
                vec3 direct2   = diffuse2 + specular2;

                float dist2 = length(u_lightPos - scenePos);
                float atten2 = 1.0 /
                    (u_attConst + u_attLinear * dist2 + u_attQuadratic * dist2 * dist2);

                direct2 *= atten2;
                hitColor = ambient2 + direct2;
            }

            hit = true;
            break;
        }

        t += u_ssrStep;
    }

    if (!hit)
        return vec3(0.0);

    return hitColor;
}


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
    float Ns      = max(NsRaw, 1.0);

    // Skip pixels with no geometry
    if (pos == vec3(0.0)) {
        FragColor   = vec4(0.0);
        BrightColor = vec4(0.0);
        return;
    }

    // ---------- DEBUG VIEWS (no bloom) ----------
    if (u_viewMode == 1) {
        vec3 p = normalize(pos) * 0.5 + 0.5;
        FragColor   = vec4(p, 1.0);
        BrightColor = vec4(0.0);
        return;
    }
    if (u_viewMode == 2) {
        vec3 n = normalize(normal) * 0.5 + 0.5;
        FragColor   = vec4(n, 1.0);
        BrightColor = vec4(0.0);
        return;
    }
    if (u_viewMode == 3) {
        FragColor   = vec4(Ka, 1.0);
        BrightColor = vec4(0.0);
        return;
    }
    if (u_viewMode == 4) {
        FragColor   = vec4(Kd, 1.0);
        BrightColor = vec4(0.0);
        return;
    }
    if (u_viewMode == 5) {
        FragColor   = vec4(Ks, 1.0);
        BrightColor = vec4(0.0);
        return;
    }

    // NEW: SSAO debug view
    if (u_viewMode == 6) {
        float ao = 1.0;
        if (u_enableSSAO) {
            ao = texture(u_ssaoTex, v_uv).r;
        }
        FragColor   = vec4(vec3(ao), 1.0); // grayscale AO
        BrightColor = vec4(0.0);
        return;
    }
    // ---------- END DEBUG VIEWS ----------

    // Emissive objects: encoded by negative shininess (NsRaw < 0.0)
    if (isEmissive) {
        vec3 emissive = Kd;

        FragColor = vec4(emissive, 1.0);

        vec3 bright = vec3(0.0);
        if (u_enableBloom) {
            float brightness = max(max(emissive.r, emissive.g), emissive.b);
            if (brightness > u_bloomThreshold)
                bright = emissive;
        }
        BrightColor = vec4(bright, 1.0);
        return;
    }

    // Normal lighting path
    vec3 N = normalize(normal);
    vec3 L = normalize(u_lightPos - pos);
    vec3 V = normalize(u_eye      - pos);
    vec3 H = normalize(L + V);

    float NdotL = max(dot(N, L), 0.0);
    float NdotH = max(dot(N, H), 0.0);

    float ao = 1.0;
    if (u_enableSSAO) {
        ao = texture(u_ssaoTex, v_uv).r;
    }

    // Hint: multiply ambient by diffuse color for better look
    vec3 ambient  = (Ka * Kd) * u_Ia * ao;
    vec3 diffuse  = u_Id * Kd * NdotL * ao;
    vec3 specular = (NdotL > 0.0) ? (u_Is * Ks * pow(NdotH, Ns)) : vec3(0.0);
    vec3 direct   = diffuse + specular;

    // ---- point light attenuation ----
    float dist = length(u_lightPos - pos);
    float attenuation = 1.0 /
        (u_attConst + u_attLinear * dist + u_attQuadratic * dist * dist);

    direct *= attenuation;

    // ---- point light shadow from cube ----
    float shadow = 0.0;
    if (u_enableShadows) {
        vec3  lightToFrag = pos - u_lightPos;
        float currentDepth = length(lightToFrag);
        float closestDepth = texture(u_shadowCube, lightToFrag).r * u_far;

        float bias = 0.03;
        shadow = (currentDepth - bias > closestDepth) ? 1.0 : 0.0;
    }

    float s = clamp(u_shadowStrength, 0.0, 1.0);
    vec3 lighting = ambient + mix(direct * s, direct, 1.0 - shadow);

    // ===== Screen-Space Reflection on floor =====
    if (u_enableSSR && isFloorPixel(pos, normal)) {
        vec3 refl = computeSSR(pos, normal, Ka, Kd, Ks, NsRaw);
        // if refl is nonzero, blend it in
        lighting = mix(lighting, refl, u_ssrIntensity);
    }


    // Bloom bright-pass
    vec3 bright = vec3(0.0);
    if (u_enableBloom) {
        float brightness = max(max(lighting.r, lighting.g), lighting.b);
        if (brightness > u_bloomThreshold)
            bright = lighting;
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
uniform bool  u_enableBloom;
uniform float u_bloomIntensity;
uniform float u_exposure;

void main()
{
    vec3 hdrColor   = texture(u_scene,     v_uv).rgb;
    vec3 bloomColor = texture(u_bloomBlur, v_uv).rgb;

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
    // tweak the exponent: 2.0 ~ subtle, 3–4 = stronger effect
    ao = pow(ao, 2.5);

    FragColor = ao;
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


// Helper to get current trice model matrix
static glm::mat4 getTriceModel()
{
    glm::mat4 m(1.0f);
    m = glm::translate(m, g_tricePos);
    m = glm::rotate(m, glm::radians(g_triceYaw), glm::vec3(0.0f, 1.0f, 0.0f));
    m = glm::scale(m, glm::vec3(g_triceScale));
    return m;
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

    glm::vec3 lightPos = g_lightEye; // point light position follows sphere

    // 1) Shadow pass (point light cube)
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

    // ---- Emissive Point Light Sphere ----
    if (g_lightSphereVAO != 0)
    {
        glm::mat4 model(1.0f);
        model = glm::translate(model, g_lightEye);     // same position as point light
        model = glm::scale(model, glm::vec3(g_lightSphereRadius));

        glBindVertexArray(g_lightSphereVAO);

        glUniformMatrix4fv(glGetUniformLocation(g_geomProgram, "u_model"),
            1, GL_FALSE, glm::value_ptr(model));

        // emissive color stored in Kd
        glm::vec3 Kd = glm::vec3(50.0f);  // bright white so bloom works
        glm::vec3 Ka = glm::vec3(0.0f);
        glm::vec3 Ks = glm::vec3(0.0f);
        float Ns = -1.0f;    // <---- negative shininess marks emissive

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

    glUniform1i(glGetUniformLocation(g_finalProgram, "u_enableBloom"), g_enableBloom ? 1 : 0);
    glUniform1f(glGetUniformLocation(g_finalProgram, "u_bloomIntensity"), g_bloomIntensity);
    glUniform1f(glGetUniformLocation(g_finalProgram, "u_exposure"), g_exposure);

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
    ImGui::Text("Directional Light");
    ImGui::DragFloat3("Light Eye", glm::value_ptr(g_lightEye), 0.05f);
    ImGui::DragFloat3("Light Center", glm::value_ptr(g_lightCenter), 0.05f);
    ImGui::SliderFloat("Light Range", &g_lightRange, 1.0f, 10.0f);

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
    g_ssaoProgram = createProgram(kLightVertexShader, kSSAOFragmentShader);
    initSSAOKernelAndNoise();
    initShadowMap();
    initFullscreenQuad();
    initLightSphere();

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


    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}