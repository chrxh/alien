#pragma once

#include <string>

#include "glad/glad.h"

#include "Definitions.h"

class _Shader
{
public:
    unsigned int ID;

    _Shader(const char* vertexPath, const char* fragmentPath, const char* geometryPath = nullptr);
    
    void use();
    void setBool(const std::string& name, bool value) const;
    void setInt(const std::string& name, int value) const;
    void setFloat(const std::string& name, float value) const;

private:
    void checkCompileErrors(GLuint shader, std::string type);
};
