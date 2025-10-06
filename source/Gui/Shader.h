#pragma once

#include <filesystem>
#include <string>

#include "glad/glad.h"

#include "Definitions.h"

class _Shader
{
public:
    _Shader(
        std::filesystem::path const& vertexPath,
        std::filesystem::path const& fragmentPath,
        std::filesystem::path const& geometryPath = std::filesystem::path());
    
    void use();
    void setBool(const std::string& name, bool value) const;
    void setInt(const std::string& name, int value) const;
    void setFloat(const std::string& name, float value) const;
    void setVec2(const std::string& name, float x, float y) const;

    unsigned int getVao() const { return _vao; }
    unsigned int getVbo() const { return _vbo; }
    unsigned int getEbo() const { return _ebo; }

private:
    void checkCompileErrors(GLuint shader, std::string type);

    unsigned int _id = 0;
    unsigned int _vao = 0;
    unsigned int _vbo = 0;
    unsigned int _ebo = 0;
};
