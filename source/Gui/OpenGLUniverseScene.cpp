#include "OpenGLUniverseScene.h"

#include <QOpenGLShader>
#include <QFile>

#include "EngineInterface/SimulationAccess.h"

float vertices[] = {
    // positions        // texture coords
    1.0f,  1.0f,  0.0f, 1.0f, 0.0f,  // top right
    1.0f,  -1.0f, 0.0f, 1.0f, 1.0f,  // bottom right
    -1.0f, -1.0f, 0.0f, 0.0f, 1.0f,  // bottom left
    -1.0f, 1.0f,  0.0f, 0.0f, 0.0f   // top left
};

namespace
{
    QString readFile(QString const& filename)
    {
        QFile file("://" + filename);
        if (!file.open(QIODevice::ReadOnly)) {
            throw std::runtime_error(filename.toStdString() + " not found");
        }
        QTextStream in(&file);
        return in.readAll();
    }
}

OpenGLUniverseScene::OpenGLUniverseScene(
    SimulationAccess* access,
    IntVector2D const& viewSize,
    std::mutex& mutex,
    QObject* parent /*= nullptr*/)
    : QGraphicsScene(parent)
    , _access(access)
    , _mutex(mutex)
{
    setSceneRect(0, 0, viewSize.x - 3, viewSize.y - 3);

    auto context = new QOpenGLContext(parent);
    context->create();
    initializeOpenGLFunctions();

    //create shaders
    auto vertex_shader = new QOpenGLShader(QOpenGLShader::Vertex, this);
    if (!vertex_shader->compileSourceCode(readFile("VertexShader.txt"))) {
        throw std::runtime_error("vertex shader compilation failed");
    }
    auto fragment_shader = new QOpenGLShader(QOpenGLShader::Fragment, this);
    if (!fragment_shader->compileSourceCode(readFile("FragmentShader.txt"))) {
        throw std::runtime_error("fragment shader compilation failed");
    }
    m_program = new QOpenGLShaderProgram(this);
    m_program->addShader(vertex_shader);
    m_program->addShader(fragment_shader);
    m_program->link();
    m_program->bind();

    //create buffer
    m_vertex.create();
    m_vertex.bind();
    m_vertex.setUsagePattern(QOpenGLBuffer::StaticDraw);
    m_vertex.allocate(vertices, sizeof(vertices));

    //create Vertex Array Object
    m_object.create();
    m_object.bind();
    m_program->enableAttributeArray(0);
    m_program->enableAttributeArray(1);
    m_program->setAttributeBuffer(0, GL_FLOAT, 0, 3, 5 * sizeof(float));
    m_program->setAttributeBuffer(1, GL_FLOAT, 3 * sizeof(float), 2, 5 * sizeof(float));
    m_program->setUniformValue("texture1", 0);

        //texture
    m_texture = new QOpenGLTexture(QOpenGLTexture::Target2D);
    m_texture->setMinificationFilter(QOpenGLTexture::Nearest);
    m_texture->setMagnificationFilter(QOpenGLTexture::Nearest);
    m_texture->setWrapMode(QOpenGLTexture::Repeat);
    m_texture->setFormat(QOpenGLTexture::RGBA8_UNorm);

    //release (unbind) all
    m_object.release();
    m_vertex.release();
    m_program->release();

    updateTexture();

    _imageResource = _access->registerImageResource(m_texture->textureId());
}

ImageResource OpenGLUniverseScene::getImageResource() const
{
    return _imageResource;
}

void OpenGLUniverseScene::updateTexture()
{
    m_texture->bind();

    m_texture->setSize(sceneRect().width(), sceneRect().height());
    m_texture->allocateStorage(QOpenGLTexture::RGBA, QOpenGLTexture::UInt8);

    m_texture->release();
}

void OpenGLUniverseScene::drawBackground(QPainter* painter, const QRectF& rect)
{
    std::lock_guard<std::mutex> lock(_mutex);
    m_program->bind();
    m_texture->bind();

    m_object.bind();
    glDrawArrays(GL_QUADS, 0, 4);
    //        glDrawElements(GL_QUADS, 1, GL_UNSIGNED_INT, 0);
    m_object.release();
    m_texture->release();
    m_program->release();
}
