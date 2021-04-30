#pragma once

#include <QGraphicsScene>
#include <QOpenGLExtraFunctions>

#include "Definitions.h"

class QOpenGLShaderProgram;

class OpenGLUniverseScene
    : public QGraphicsScene
    , protected QOpenGLExtraFunctions
{
public:
    const char* vertexShaderSource = "#version 330 core\n"
                                     "layout(location = 0) in vec3 aPos;\n"
                                     "layout(location = 1) in vec3 aColor;\n"
                                     "layout(location = 2) in vec2 aTexCoord;\n"
                                     "out vec3 ourColor;\n"
                                     "out vec2 TexCoord;\n"
                                     "void main()\n"
                                     "\n{"
                                     "  gl_Position = vec4(aPos, 1.0);\n"
                                     "  ourColor = aColor;\n"
                                     "  TexCoord = vec2(aTexCoord.x, aTexCoord.y);\n"
                                     "}\0";
    const char* fragmentShaderSource = "#version 330 core\n"
                                       "out vec4 FragColor;\n"
                                       "in vec3 ourColor;\n"
                                       "in vec2 TexCoord;\n"
                                       "uniform sampler2D texture1;\n"
                                       "void main()\n"
                                       "{\n"
                                       "    FragColor = texture(texture1, TexCoord);\n"
                                       "}\0";

    unsigned int shaderProgram;
    unsigned int VBO, VAO, EBO;
    unsigned int texture;

    OpenGLUniverseScene(
        SimulationAccess* access,
        IntVector2D const& viewSize,
        std::mutex& mutex,
        QObject* parent = nullptr);

    ImageResource getImageResource() const;

    void updateTexture();

    void drawBackground(QPainter* painter, const QRectF& rect) override;

private:
    SimulationAccess* _access;
    std::mutex& _mutex;
    ImageResource _imageResource;

    QOpenGLContext* m_context;
    GLint m_posAttr = 0;
    GLint m_colAttr = 0;
    GLint m_matrixUniform = 0;

    QOpenGLShaderProgram* m_program = nullptr;
};
