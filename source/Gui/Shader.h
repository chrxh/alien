#pragma once

#include <filesystem>
#include <string>
#include <string_view>

#include <glad/glad.h>

#include "Definitions.h"

class _Shader
{
public:
    static Shader create(
        std::filesystem::path const& vertexPath,
        std::filesystem::path const& fragmentPath,
        std::filesystem::path const& geometryPath = std::filesystem::path());

    static Shader createFromSource(std::string_view vertexSource, std::string_view fragmentSource, std::string_view geometrySource = "");

    void use();
    void setBool(std::string const& name, bool value) const;
    void setInt(std::string const& name, int value) const;
    void setFloat(std::string const& name, float value) const;
    void setVec2(std::string const& name, RealVector2D const& value) const;
    void setVec3(std::string const& name, FloatColorRGB const& value) const;

private:
    _Shader(std::filesystem::path const& vertexPath, std::filesystem::path const& fragmentPath, std::filesystem::path const& geometryPath);

    _Shader(std::string_view vertexSource, std::string_view fragmentSource, std::string_view geometrySource);

    void checkCompileErrors(GLuint shader, std::string type, std::string const& identifier);

    unsigned int _id = 0;
};
