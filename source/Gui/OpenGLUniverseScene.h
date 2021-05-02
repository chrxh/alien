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
/*
        SimulationAccess* access,
        IntVector2D const& viewSize,
*/
        QOpenGLContext* context,
        QObject* parent = nullptr);

    void init(SimulationAccess* access, std::mutex& mutex);


    ImageResource getImageResource() const;

    void resize(IntVector2D const& size);

    void drawBackground(QPainter* painter, const QRectF& rect) override;

private:
    void updateTexture(IntVector2D const& size);

    SimulationAccess* _access;
    boost::optional<std::mutex&> _mutex;
    ImageResource _imageResource;

    GLint m_posAttr = 0;
    GLint m_colAttr = 0;
    GLint m_matrixUniform = 0;

    QOpenGLBuffer m_vertex;
    QOpenGLVertexArrayObject m_vertexArrayObject;
    QOpenGLShaderProgram* m_program = nullptr;
    QOpenGLTexture* m_texture = nullptr;
    QOpenGLFramebufferObject* m_frameBufferObject = nullptr;
    bool _first = true;
};
