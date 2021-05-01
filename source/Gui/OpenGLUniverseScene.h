#pragma once

#include <QGraphicsScene>
#include <QOpenGLExtraFunctions>
#include <QOpenGLBuffer>
#include <QOpenGLVertexArrayObject>
#include <QOpenGLTexture>

#include "Definitions.h"

class QOpenGLShaderProgram;

class OpenGLUniverseScene
    : public QGraphicsScene
    , protected QOpenGLExtraFunctions
{
public:
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

    GLint m_posAttr = 0;
    GLint m_colAttr = 0;
    GLint m_matrixUniform = 0;

    QOpenGLBuffer m_vertex;
    QOpenGLVertexArrayObject m_object;
    QOpenGLShaderProgram* m_program = nullptr;
    QOpenGLTexture* m_texture = nullptr;
};
