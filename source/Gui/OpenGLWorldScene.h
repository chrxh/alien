#pragma once

#include <QGraphicsScene>
#include <QOpenGLExtraFunctions>
#include <QOpenGLBuffer>
#include <QOpenGLVertexArrayObject>
#include <QOpenGLTexture>

#include "Definitions.h"
#include "SimulationViewSettings.h"

class QOpenGLShaderProgram;
class QOpenGLFramebufferObject;

class OpenGLWorldScene
    : public QGraphicsScene
    , protected QOpenGLExtraFunctions
{
public:
    OpenGLWorldScene(
        QOpenGLContext* context,
        QObject* parent = nullptr);

    void init(SimulationAccess* access, std::mutex& mutex);

    void setSettings(SimulationViewSettings const& settings);

    enum class MotionBlurFactor
    {
        Default, High
    };
    void setMotionBlurFactor(MotionBlurFactor factor);

    ImageResource getImageResource() const;

    void resize(IntVector2D const& size);

    void drawBackground(QPainter* painter, const QRectF& rect) override;

private:
    void updateTexture(IntVector2D const& size);

    SimulationAccess* _access;
    SimulationViewSettings _settings;
    boost::optional<std::mutex&> _mutex;
    ImageResource _imageResource;

    QSurface* _surface = nullptr;
    QOpenGLContext* _context = nullptr;

    QOpenGLBuffer m_vertex;
    QOpenGLVertexArrayObject m_vertexArrayObject;
    QOpenGLShaderProgram* m_program = nullptr;
    QOpenGLTexture* m_texture = nullptr;
    QOpenGLFramebufferObject* m_frameBufferObject = nullptr;
};
