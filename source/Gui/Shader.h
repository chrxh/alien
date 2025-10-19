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
    void setBool(std::string const& name, bool value) const;
    void setInt(std::string const& name, int value) const;
    void setFloat(std::string const& name, float value) const;
    void setVec2(std::string const& name, RealVector2D const& value) const;
    void setVec3(std::string const& name, RealVector3D const& value) const;

private:
    _Shader(
        std::filesystem::path const& vertexPath,
        std::filesystem::path const& fragmentPath,
        std::filesystem::path const& geometryPath);

    void checkCompileErrors(GLuint shader, std::string type, std::filesystem::path const& path);

    unsigned int _id = 0;
};
