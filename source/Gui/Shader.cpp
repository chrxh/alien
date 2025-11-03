#include "Shader.h"

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

Shader _Shader::create(std::filesystem::path const& vertexPath, std::filesystem::path const& fragmentPath, std::filesystem::path const& geometryPath)
{
    return Shader(new _Shader(vertexPath, fragmentPath, geometryPath));
}

Shader _Shader::createFromSource(std::string_view vertexSource, std::string_view fragmentSource, std::string_view geometrySource)
{
    return Shader(new _Shader(vertexSource, fragmentSource, geometrySource));
}

void _Shader::use()
{
    glUseProgram(_id);
}
void _Shader::setBool(std::string const& name, bool value) const
{
    glUniform1i(glGetUniformLocation(_id, name.c_str()), (int)value);
}
void _Shader::setInt(std::string const& name, int value) const
{
    glUniform1i(glGetUniformLocation(_id, name.c_str()), value);
}
void _Shader::setFloat(std::string const& name, float value) const
{
    glUniform1f(glGetUniformLocation(_id, name.c_str()), value);
}

void _Shader::setVec2(std::string const& name, RealVector2D const& value) const
{
    glUniform2f(glGetUniformLocation(_id, name.c_str()), value.x, value.y);
}

void _Shader::setVec3(std::string const& name, FloatColorRGB const& value) const
{
    glUniform3f(glGetUniformLocation(_id, name.c_str()), value.r, value.g, value.b);
}

void _Shader::checkCompileErrors(GLuint shader, std::string type, std::string const& identifier)
{
    GLint success;
    GLchar infoLog[1024];
    if (type != "PROGRAM") {
        glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
        if (!success) {
            glGetShaderInfoLog(shader, 1024, NULL, infoLog);
            std::cout << "ERROR::SHADER_COMPILATION_ERROR of type: " << type << "\n"
                      << "IDENTIFIER: " << identifier << "\n"
                      << infoLog << "\n -- --------------------------------------------------- -- " << std::endl;
        }
    } else {
        glGetProgramiv(shader, GL_LINK_STATUS, &success);
        if (!success) {
            glGetProgramInfoLog(shader, 1024, NULL, infoLog);
            std::cout << "ERROR::PROGRAM_LINKING_ERROR of type: " << type << "\n"
                      << "IDENTIFIER: " << identifier << "\n"
                      << infoLog << "\n -- --------------------------------------------------- -- " << std::endl;
        }
    }
}

_Shader::_Shader(
    std::filesystem::path const& vertexPath,
    std::filesystem::path const& fragmentPath,
    std::filesystem::path const& geometryPath /*, std::optional<unsigned int> sharedVbo*/)
{
    // 1. retrieve the vertex/fragment source code from filePath
    std::string vertexCode;
    std::string fragmentCode;
    std::string geometryCode;
    std::ifstream vShaderFile;
    std::ifstream fShaderFile;
    std::ifstream gShaderFile;
    // ensure ifstream objects can throw exceptions:
    vShaderFile.exceptions(std::ifstream::failbit | std::ifstream::badbit);
    fShaderFile.exceptions(std::ifstream::failbit | std::ifstream::badbit);
    gShaderFile.exceptions(std::ifstream::failbit | std::ifstream::badbit);
    try {
        // open files
        vShaderFile.open(vertexPath);
        fShaderFile.open(fragmentPath);
        std::stringstream vShaderStream, fShaderStream;
        // read file's buffer contents into streams
        vShaderStream << vShaderFile.rdbuf();
        fShaderStream << fShaderFile.rdbuf();
        // close file handlers
        vShaderFile.close();
        fShaderFile.close();
        // convert stream into string
        vertexCode = vShaderStream.str();
        fragmentCode = fShaderStream.str();
        // if geometry shader path is present, also load a geometry shader
        if (!geometryPath.empty()) {
            gShaderFile.open(geometryPath);
            std::stringstream gShaderStream;
            gShaderStream << gShaderFile.rdbuf();
            gShaderFile.close();
            geometryCode = gShaderStream.str();
        }
    } catch (std::ifstream::failure&) {
        std::cout << "ERROR::SHADER::FILE_NOT_SUCCESFULLY_READ" << std::endl;
    }
    char const* vShaderCode = vertexCode.c_str();
    char const* fShaderCode = fragmentCode.c_str();

    // 2. compile shaders
    unsigned int vertex, fragment;

    // Vertex shader
    vertex = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertex, 1, &vShaderCode, NULL);
    glCompileShader(vertex);
    checkCompileErrors(vertex, "VERTEX", vertexPath.string());

    // Fragment Shader
    fragment = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragment, 1, &fShaderCode, NULL);
    glCompileShader(fragment);
    checkCompileErrors(fragment, "FRAGMENT", fragmentPath.string());

    // If geometry shader is given, compile geometry shader
    unsigned int geometry;
    if (!geometryPath.empty()) {
        const char* gShaderCode = geometryCode.c_str();
        geometry = glCreateShader(GL_GEOMETRY_SHADER);
        glShaderSource(geometry, 1, &gShaderCode, NULL);
        glCompileShader(geometry);
        checkCompileErrors(geometry, "GEOMETRY", geometryPath.string());
    }
    // Shader Program
    _id = glCreateProgram();
    glAttachShader(_id, vertex);
    glAttachShader(_id, fragment);
    if (!geometryPath.empty()) {
        glAttachShader(_id, geometry);
    }
    glLinkProgram(_id);
    checkCompileErrors(_id, "PROGRAM", "shader_program");

    // Delete the shaders as they're linked into our program now and no longer necessary
    glDeleteShader(vertex);
    glDeleteShader(fragment);
    if (!geometryPath.empty()) {
        glDeleteShader(geometry);
    }
}

_Shader::_Shader(std::string_view vertexSource, std::string_view fragmentSource, std::string_view geometrySource)
{
    // Convert string_view to C strings
    std::string vertexCode(vertexSource);
    std::string fragmentCode(fragmentSource);
    std::string geometryCode(geometrySource);

    char const* vShaderCode = vertexCode.c_str();
    char const* fShaderCode = fragmentCode.c_str();

    // Compile shaders
    unsigned int vertex, fragment;

    // Vertex shader
    vertex = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertex, 1, &vShaderCode, NULL);
    glCompileShader(vertex);
    checkCompileErrors(vertex, "VERTEX", "embedded_vertex_shader");

    // Fragment Shader
    fragment = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragment, 1, &fShaderCode, NULL);
    glCompileShader(fragment);
    checkCompileErrors(fragment, "FRAGMENT", "embedded_fragment_shader");

    // If geometry shader source is given, compile geometry shader
    unsigned int geometry;
    bool hasGeometry = !geometrySource.empty();
    if (hasGeometry) {
        const char* gShaderCode = geometryCode.c_str();
        geometry = glCreateShader(GL_GEOMETRY_SHADER);
        glShaderSource(geometry, 1, &gShaderCode, NULL);
        glCompileShader(geometry);
        checkCompileErrors(geometry, "GEOMETRY", "embedded_geometry_shader");
    }

    // Shader Program
    _id = glCreateProgram();
    glAttachShader(_id, vertex);
    glAttachShader(_id, fragment);
    if (hasGeometry) {
        glAttachShader(_id, geometry);
    }
    glLinkProgram(_id);
    checkCompileErrors(_id, "PROGRAM", "shader_program");

    // Delete the shaders as they're linked into our program now and no longer necessary
    glDeleteShader(vertex);
    glDeleteShader(fragment);
    if (hasGeometry) {
        glDeleteShader(geometry);
    }
}
