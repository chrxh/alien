#pragma once

#include <filesystem>
#include <string>

#include "glad/glad.h"

#include "Definitions.h"

class _Shader
{
public:
    unsigned int ID;

    _Shader(
        std::filesystem::path const& vertexPath,
        std::filesystem::path const& fragmentPath,
        std::filesystem::path const& geometryPath = std::filesystem::path());
    
    void use();
    void setBool(const std::string& name, bool value) const;
    void setInt(const std::string& name, int value) const;
    void setFloat(const std::string& name, float value) const;

private:
    void checkCompileErrors(GLuint shader, std::string type);
};
