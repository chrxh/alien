#include "OpenGLWorldScene.h"

#include <QOpenGLShader>
#include <QFile>
#include <QOpenGLFramebufferObject>

#include "Base/Exceptions.h"
#include "EngineInterface/SimulationAccess.h"

namespace
{
    float vertices[] = {
        // positions        // texture coords
        1.0f,  1.0f,  0.0f, 1.0f, 1.0f,  // top right
        1.0f,  -1.0f, 0.0f, 1.0f, 0.0f,  // bottom right
        -1.0f, -1.0f, 0.0f, 0.0f, 0.0f,  // bottom left
        -1.0f, 1.0f,  0.0f, 0.0f, 1.0f   // top left
    };

    QString readFile(QString const& filename)
    {
        QFile file("://" + filename);
        if (!file.open(QIODevice::ReadOnly)) {
            throw BugReportException(filename.toStdString() + " not found");
        }
        QTextStream in(&file);
        return in.readAll();
    }
}

OpenGLWorldScene::OpenGLWorldScene(QOpenGLContext* context, QObject* parent /*= nullptr*/)
    : QGraphicsScene(parent)
    , _surface(context->surface())
    , _context(context)
{
    initializeOpenGLFunctions();

    //create shaders
    auto vertex_shader = new QOpenGLShader(QOpenGLShader::Vertex, this);
    if (!vertex_shader->compileSourceCode(readFile("Shader/VertexShader.glsl"))) {
        throw BugReportException("vertex shader compilation failed");
    }
    auto fragment_shader = new QOpenGLShader(QOpenGLShader::Fragment, this);
    if (!fragment_shader->compileSourceCode(readFile("Shader/FragmentShader.glsl"))) {
        throw BugReportException("fragment shader compilation failed");
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
    m_vertexArrayObject.create();
    m_vertexArrayObject.bind();
    m_program->enableAttributeArray(0);
    m_program->enableAttributeArray(1);
    m_program->setAttributeBuffer(0, GL_FLOAT, 0, 3, 5 * sizeof(float));
    m_program->setAttributeBuffer(1, GL_FLOAT, 3 * sizeof(float), 2, 5 * sizeof(float));
    m_program->setUniformValue("texture1", 0);
    m_program->setUniformValue("texture2", 1);
    m_program->setUniformValue("glowEffect", _settings.glowEffect);
    m_program->setUniformValue("motionEffect", _settings.motionEffect);
    m_program->setUniformValue("motionBlurFactor", 0.60f);

    //release (unbind) all
    m_vertexArrayObject.release();
    m_vertex.release();
    m_program->release();
}

void OpenGLWorldScene::init(SimulationAccess* access, std::mutex& mutex)
{
    _access = access;
    _mutex = mutex;
}

void OpenGLWorldScene::setSettings(SimulationViewSettings const& settings)
{
    _settings = settings;
    _context->makeCurrent(_surface);
    m_program->bind();
    m_program->setUniformValue("glowEffect", settings.glowEffect);
    m_program->setUniformValue("motionEffect", settings.motionEffect);
    m_program->release();
}

void OpenGLWorldScene::setMotionBlurFactor(MotionBlurFactor factor)
{
    _context->makeCurrent(_surface);
    m_program->bind();
    if (MotionBlurFactor::Default == factor) {
        m_program->setUniformValue("motionBlurFactor", 0.60f);
    } else {
        m_program->setUniformValue("motionBlurFactor", 0.45f);
    }
    m_program->release();
}

ImageResource OpenGLWorldScene::getImageResource() const
{
    return _imageResource;
}

void OpenGLWorldScene::resize(IntVector2D const& size)
{
    setSceneRect(0, 0, size.x, size.y);
    updateTexture(size);

    _context->makeCurrent(_surface);

    delete m_frameBufferObject;
    m_frameBufferObject =
        new QOpenGLFramebufferObject(QSize(size.x, size.y), QOpenGLFramebufferObject::CombinedDepthStencil);

    glViewport(0, 0, size.x, size.y);
}

void OpenGLWorldScene::drawBackground(QPainter* painter, const QRectF& rect)
{
    std::lock_guard<std::mutex> lock(*_mutex);
    m_program->bind();
    m_vertexArrayObject.bind();

    m_texture->bind();
    m_frameBufferObject->bind();
    m_program->setUniformValue("phase", 0);


    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, m_texture->textureId());

    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, m_frameBufferObject->texture());

    glDrawArrays(GL_QUADS, 0, 4);
    m_texture->release();

    m_frameBufferObject->bindDefault();
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, m_frameBufferObject->texture());
    m_program->setUniformValue("phase", 1);
    glDrawArrays(GL_QUADS, 0, 4);
    glBindTexture(GL_TEXTURE_2D, 0);

    m_vertexArrayObject.release();
    m_program->release();
}

void OpenGLWorldScene::updateTexture(IntVector2D const& size)
{
    delete m_texture;

    m_texture = new QOpenGLTexture(QOpenGLTexture::Target2D);
    m_texture->setMinificationFilter(QOpenGLTexture::Linear);
    m_texture->setMagnificationFilter(QOpenGLTexture::Linear);
    m_texture->setWrapMode(QOpenGLTexture::ClampToBorder);
    m_texture->setFormat(QOpenGLTexture::RGBA16_UNorm);

    m_texture->bind();
    m_texture->setSize(size.x, size.y);
    m_texture->allocateStorage(QOpenGLTexture::RGBA, QOpenGLTexture::UInt16);
    m_texture->release();

    _imageResource = _access->registerImageResource(m_texture->textureId());
}
