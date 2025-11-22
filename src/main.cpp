#include <glad/glad.h>
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#define TINYOBJLOADER_IMPLEMENTATION
#include <tiny_obj_loader.h>

#include <iostream>
#include <vector>
#include <string>

// =============== GLOBALS ===============

ImVec4 g_clearColor = ImVec4(0.1f, 0.1f, 0.12f, 1.0f);

// camera
glm::vec3 g_eye(4.0f, 1.0f, -1.5f);
glm::vec3 g_center(3.0f, 1.0f, -1.5f);
glm::vec3 g_up(0.0f, 1.0f, 0.0f);
float     g_fov = 45.0f;

// input
bool g_mouseRotating = false;
double g_lastX = 0.0, g_lastY = 0.0;

// shader program
GLuint g_program = 0;

// simple mesh & material
struct Mesh {
    GLuint vao = 0;
    GLuint vbo = 0;
    GLuint ebo = 0;
    GLsizei indexCount = 0;
    GLuint diffuseTex = 0;   // 0 = no texture
    glm::vec3 diffuseColor = glm::vec3(1.0f);
};
std::vector<Mesh> g_meshes;

// =============== UTILS ===============

static void glfw_error_callback(int error, const char* description)
{
    fprintf(stderr, "GLFW Error %d: %s\n", error, description);
}

static GLuint compileShader(GLenum type, const char* src)
{
    GLuint s = glCreateShader(type);
    glShaderSource(s, 1, &src, nullptr);
    glCompileShader(s);
    GLint ok;
    glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        GLint len = 0;
        glGetShaderiv(s, GL_INFO_LOG_LENGTH, &len);
        std::string log(len, '\0');
        glGetShaderInfoLog(s, len, nullptr, log.data());
        std::cerr << "Shader compile error:\n" << log << std::endl;
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
        GLint len = 0;
        glGetProgramiv(prog, GL_INFO_LOG_LENGTH, &len);
        std::string log(len, '\0');
        glGetProgramInfoLog(prog, len, nullptr, log.data());
        std::cerr << "Program link error:\n" << log << std::endl;
    }
    glDeleteShader(vs);
    glDeleteShader(fs);
    return prog;
}

static GLuint loadTexture2D(const std::string& path, bool srgb = false)
{
    int w, h, n;
    stbi_set_flip_vertically_on_load(true);
    unsigned char* data = stbi_load(path.c_str(), &w, &h, &n, 0);
    if (!data) {
        std::cerr << "Failed to load texture: " << path << std::endl;
        return 0;
    }

    GLenum format = GL_RGB;
    if (n == 1) format = GL_RED;
    else if (n == 3) format = GL_RGB;
    else if (n == 4) format = GL_RGBA;

    GLuint tex;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexImage2D(GL_TEXTURE_2D, 0, (format == GL_RGBA ? GL_RGBA8 : GL_RGB8),
        w, h, 0, format, GL_UNSIGNED_BYTE, data);
    glGenerateMipmap(GL_TEXTURE_2D);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);

    stbi_image_free(data);
    return tex;
}

// =============== LOAD SCENE ===============

struct Vertex {
    glm::vec3 pos;
    glm::vec3 normal;
    glm::vec2 uv;
};

static void loadScene_GreyWhiteRoom(const std::string& objPath)
{
    tinyobj::ObjReaderConfig config;
    config.triangulate = true;

    tinyobj::ObjReader reader;
    if (!reader.ParseFromFile(objPath, config)) {
        std::cerr << "TinyObj error: " << reader.Error() << std::endl;
        return;
    }
    if (!reader.Warning().empty())
        std::cout << "TinyObj warning: " << reader.Warning() << std::endl;

    const auto& attrib = reader.GetAttrib();
    const auto& shapes = reader.GetShapes();
    const auto& materials = reader.GetMaterials();

    // Base directory for textures
    size_t slashPos = objPath.find_last_of("/\\");
    std::string baseDir = (slashPos == std::string::npos) ? "" : objPath.substr(0, slashPos + 1);

    // Preload textures for materials
    std::vector<GLuint> matTexID(materials.size(), 0);
    std::vector<glm::vec3> matDiffuse(materials.size(), glm::vec3(1.0f));
    for (size_t m = 0; m < materials.size(); ++m) {
        matDiffuse[m] = glm::vec3(materials[m].diffuse[0],
            materials[m].diffuse[1],
            materials[m].diffuse[2]);
        if (!materials[m].diffuse_texname.empty()) {
            std::string texPath = baseDir + materials[m].diffuse_texname;
            matTexID[m] = loadTexture2D(texPath);
            std::cout << "Loaded texture " << texPath << " (mat " << m << ")\n";
        }
    }

    // Build mesh per shape (assuming each shape has one material – usually true)
    for (const auto& shape : shapes) {
        if (shape.mesh.indices.empty()) continue;

        int matID = -1;
        if (!shape.mesh.material_ids.empty())
            matID = shape.mesh.material_ids[0];

        std::vector<Vertex> vertices;
        std::vector<unsigned int> indices;

        vertices.reserve(shape.mesh.indices.size());
        indices.reserve(shape.mesh.indices.size());

        for (size_t i = 0; i < shape.mesh.indices.size(); ++i) {
            const tinyobj::index_t& idx = shape.mesh.indices[i];

            Vertex v{};
            v.pos = glm::vec3(
                attrib.vertices[3 * idx.vertex_index + 0],
                attrib.vertices[3 * idx.vertex_index + 1],
                attrib.vertices[3 * idx.vertex_index + 2]);

            if (idx.normal_index >= 0) {
                v.normal = glm::vec3(
                    attrib.normals[3 * idx.normal_index + 0],
                    attrib.normals[3 * idx.normal_index + 1],
                    attrib.normals[3 * idx.normal_index + 2]);
            }
            else {
                v.normal = glm::vec3(0, 1, 0);
            }

            if (idx.texcoord_index >= 0) {
                v.uv = glm::vec2(
                    attrib.texcoords[2 * idx.texcoord_index + 0],
                    attrib.texcoords[2 * idx.texcoord_index + 1]);
            }
            else {
                v.uv = glm::vec2(0.0f);
            }

            vertices.push_back(v);
            indices.push_back(static_cast<unsigned int>(i));
        }

        Mesh mesh;
        glGenVertexArrays(1, &mesh.vao);
        glGenBuffers(1, &mesh.vbo);
        glGenBuffers(1, &mesh.ebo);

        glBindVertexArray(mesh.vao);
        glBindBuffer(GL_ARRAY_BUFFER, mesh.vbo);
        glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(Vertex),
            vertices.data(), GL_STATIC_DRAW);

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, mesh.ebo);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(unsigned int),
            indices.data(), GL_STATIC_DRAW);

        // layout:
        // 0: position
        // 1: normal
        // 2: uv
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex),
            (void*)offsetof(Vertex, pos));
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex),
            (void*)offsetof(Vertex, normal));
        glEnableVertexAttribArray(2);
        glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(Vertex),
            (void*)offsetof(Vertex, uv));

        glBindVertexArray(0);

        mesh.indexCount = static_cast<GLsizei>(indices.size());
        if (matID >= 0 && matID < (int)materials.size()) {
            mesh.diffuseTex = matTexID[matID];
            mesh.diffuseColor = matDiffuse[matID];
        }

        g_meshes.push_back(mesh);
    }

    std::cout << "Loaded " << g_meshes.size() << " meshes\n";
}

// =============== SHADERS ===============

static const char* kVertexShader = R"(#version 410 core
layout(location = 0) in vec3 a_pos;
layout(location = 1) in vec3 a_normal;
layout(location = 2) in vec2 a_uv;

out vec3 v_worldPos;
out vec3 v_normal;
out vec2 v_uv;

uniform mat4 u_model;
uniform mat4 u_view;
uniform mat4 u_proj;

void main()
{
    vec4 wp = u_model * vec4(a_pos, 1.0);
    v_worldPos = wp.xyz;
    v_normal   = mat3(transpose(inverse(u_model))) * a_normal;
    v_uv       = a_uv;
    gl_Position = u_proj * u_view * wp;
}
)";

static const char* kFragmentShader = R"(#version 410 core
in vec3 v_worldPos;
in vec3 v_normal;
in vec2 v_uv;

out vec4 FragColor;

uniform vec3 u_eye;

uniform vec3 u_lightPos;
uniform vec3 u_lightColor;

uniform vec3 u_diffuseColor;
uniform sampler2D u_diffuseTex;
uniform bool u_useTex;

void main()
{
    vec3 N = normalize(v_normal);
    vec3 L = normalize(u_lightPos - v_worldPos);
    vec3 V = normalize(u_eye - v_worldPos);
    vec3 H = normalize(L + V);

    float NdotL = max(dot(N, L), 0.0);
    float NdotH = max(dot(N, H), 0.0);

    vec3 baseColor = u_diffuseColor;
    if (u_useTex) {
        vec4 tex = texture(u_diffuseTex, v_uv);
        if (tex.a < 0.5)
            discard;                    // *** alpha test for leaves ***
        baseColor *= tex.rgb;
    }

    vec3 ambient  = 0.1 * baseColor;
    vec3 diffuse  = NdotL * baseColor;
    vec3 specular = pow(NdotH, 32.0) * u_lightColor;

    FragColor = vec4(ambient + diffuse + specular, 1.0);
}
)";

// =============== RENDERING ===============

static void on_display(GLFWwindow* window)
{
    int display_w, display_h;
    glfwGetFramebufferSize(window, &display_w, &display_h);

    glViewport(0, 0, display_w, display_h);
    glEnable(GL_DEPTH_TEST);

    glClearColor(g_clearColor.x * g_clearColor.w,
        g_clearColor.y * g_clearColor.w,
        g_clearColor.z * g_clearColor.w,
        g_clearColor.w);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    float aspect = (display_h > 0) ? (float)display_w / (float)display_h : 1.0f;
    glm::mat4 proj = glm::perspective(glm::radians(g_fov), aspect, 0.1f, 100.0f);
    glm::mat4 view = glm::lookAt(g_eye, g_center, g_up);
    glm::mat4 model = glm::mat4(1.0f);

    glUseProgram(g_program);
    glUniformMatrix4fv(glGetUniformLocation(g_program, "u_model"), 1, GL_FALSE, glm::value_ptr(model));
    glUniformMatrix4fv(glGetUniformLocation(g_program, "u_view"), 1, GL_FALSE, glm::value_ptr(view));
    glUniformMatrix4fv(glGetUniformLocation(g_program, "u_proj"), 1, GL_FALSE, glm::value_ptr(proj));

    glUniform3fv(glGetUniformLocation(g_program, "u_eye"), 1, glm::value_ptr(g_eye));

    // simple light slightly above the table
    glm::vec3 lightPos(3.0f, 2.0f, -1.5f);
    glm::vec3 lightColor(1.0f, 1.0f, 1.0f);
    glUniform3fv(glGetUniformLocation(g_program, "u_lightPos"), 1, glm::value_ptr(lightPos));
    glUniform3fv(glGetUniformLocation(g_program, "u_lightColor"), 1, glm::value_ptr(lightColor));

    for (const auto& mesh : g_meshes) {
        glBindVertexArray(mesh.vao);

        glUniform3fv(glGetUniformLocation(g_program, "u_diffuseColor"), 1,
            glm::value_ptr(mesh.diffuseColor));

        bool useTex = (mesh.diffuseTex != 0);
        glUniform1i(glGetUniformLocation(g_program, "u_useTex"), useTex);
        if (useTex) {
            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D, mesh.diffuseTex);
            glUniform1i(glGetUniformLocation(g_program, "u_diffuseTex"), 0);
        }

        glDrawElements(GL_TRIANGLES, mesh.indexCount, GL_UNSIGNED_INT, 0);
    }

    glBindVertexArray(0);
    glUseProgram(0);
}

// =============== GUI ===============

static void on_gui()
{
    ImGuiIO& io = ImGui::GetIO(); (void)io;

    ImGui::Begin("Camera");

    ImGui::Text("Eye");
    ImGui::DragFloat3("eye", glm::value_ptr(g_eye), 0.05f);
    ImGui::Text("Center");
    ImGui::DragFloat3("center", glm::value_ptr(g_center), 0.05f);
    ImGui::SliderFloat("FOV", &g_fov, 20.0f, 80.0f);

    ImGui::ColorEdit3("Clear color", (float*)&g_clearColor);

    ImGui::Text("Use WASD = move on plane");
    ImGui::Text("Q/E = move down/up");

    ImGui::End();
}

// =============== INPUT HANDLERS ===============

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
    if (button == GLFW_MOUSE_BUTTON_LEFT) {
        if (action == GLFW_PRESS) {
            g_mouseRotating = true;
            glfwGetCursorPos(window, &g_lastX, &g_lastY);
        }
        else {
            g_mouseRotating = false;
        }
    }
}

static void cursor_pos_callback(GLFWwindow* window, double xpos, double ypos)
{
    if (!g_mouseRotating) return;

    float dx = float(xpos - g_lastX);
    float dy = float(ypos - g_lastY);
    g_lastX = xpos;
    g_lastY = ypos;

    // orbit the camera around center
    glm::vec3 dir = g_eye - g_center;
    float radius = glm::length(dir);
    if (radius < 1e-3f) return;

    float yaw = atan2(dir.z, dir.x);
    float pitch = asin(dir.y / radius);

    const float sensitivity = 0.005f;
    yaw -= dx * sensitivity;
    pitch -= dy * sensitivity;
    pitch = glm::clamp(pitch, -1.4f, 1.4f);

    dir.x = radius * cos(pitch) * cos(yaw);
    dir.y = radius * sin(pitch);
    dir.z = radius * cos(pitch) * sin(yaw);

    g_eye = g_center + dir;
}

// =============== MAIN ===============

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
    GLFWwindow* window = glfwCreateWindow((int)(1280 * main_scale), (int)(800 * main_scale),
        "Grey White Room (ImGui)", nullptr, nullptr);
    if (window == nullptr)
        return 1;

    glfwMakeContextCurrent(window);
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        std::cerr << "Failed to initialize GLAD\n";
        return -1;
    }
    glfwSwapInterval(1);

    // callbacks
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

    // === our GL resources ===
    g_program = createProgram(kVertexShader, kFragmentShader);
    loadScene_GreyWhiteRoom("./assets/indoor_model/Grey_White_Room.obj"); // path relative to exe
    //comment

    // Main loop
    while (!glfwWindowShouldClose(window))
    {
        glfwPollEvents();
        if (glfwGetWindowAttrib(window, GLFW_ICONIFIED) != 0)
        {
            ImGui_ImplGlfw_Sleep(10);
            continue;
        }

        // Start ImGui frame
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        on_gui();
        on_display(window);

        // Render ImGui on top
        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        glfwSwapBuffers(window);
    }

    // Cleanup
    for (auto& m : g_meshes) {
        if (m.diffuseTex) glDeleteTextures(1, &m.diffuseTex);
        if (m.ebo) glDeleteBuffers(1, &m.ebo);
        if (m.vbo) glDeleteBuffers(1, &m.vbo);
        if (m.vao) glDeleteVertexArrays(1, &m.vao);
    }

    if (g_program) glDeleteProgram(g_program);

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
