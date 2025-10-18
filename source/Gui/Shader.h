#pragma once

#include <filesystem>
#include <string>

#include <glad/glad.h>

#include "Definitions.h"

class _Shader
{
public:
    static Shader create(
        std::filesystem::path const& vertexPath,
        std::filesystem::path const& fragmentPath,
        std::filesystem::path const& geometryPath = std::filesystem::path());

    void use();
    void setBool(const std::string& name, bool value) const;
    void setInt(const std::string& name, int value) const;
    void setFloat(const std::string& name, float value) const;
    void setVec2(const std::string& name, float x, float y) const;

private:
    _Shader(
        std::filesystem::path const& vertexPath,
        std::filesystem::path const& fragmentPath,
        std::filesystem::path const& geometryPath);

    void checkCompileErrors(GLuint shader, std::string type, std::filesystem::path const& path);

    unsigned int _id = 0;
};
