#include "VectorUniverseView.h"

#include <QGraphicsPixmapItem>
#include <QGraphicsSceneMouseEvent>
#include <QGraphicsView>
#include <QScrollBar>
#include <QResizeEvent>
#include <QtGui>
#include <QOpenGLExtraFunctions>
#include <QMatrix4x4>
#include <QOpenGLWidget>
#include <QOpenGLShaderProgram>

#include <QtCore/qmath.h>

#include "Base/ServiceLocator.h"
#include "CoordinateSystem.h"
#include "DataRepository.h"
#include "EngineInterface/EngineInterfaceBuilderFacade.h"
#include "EngineInterface/PhysicalActions.h"
#include "EngineInterface/SimulationAccess.h"
#include "EngineInterface/SimulationContext.h"
#include "EngineInterface/SimulationController.h"
#include "EngineInterface/SpaceProperties.h"
#include "Gui/Notifier.h"
#include "Gui/Settings.h"
#include "Gui/ViewportInterface.h"
#include "VectorImageSectionItem.h"
#include "VectorViewport.h"

            float vertices[] = {
    0.5f,
    0.5f,
    0.0f,  // top right
    0.5f,
    -0.5f,
    0.0f,  // bottom right
    -0.5f,
    -0.5f,
    0.0f,  // bottom left
    -0.5f,
    0.5f,
    0.0f  // top left
};
unsigned int indices[] = {
    // note that we start from 0!
    0,
    1,
    3,  // first Triangle
    1,
    2,
    3  // second Triangle
};

#include <iostream>

class VectorViewGraphicsScene
    : public QGraphicsScene
    , protected QOpenGLExtraFunctions
{
public:
/*
    const char* vertexShaderSource = "attribute highp vec4 posAttr;\n"
                                            "attribute lowp vec4 colAttr;\n"
                                            "varying lowp vec4 col;\n"
                                            "uniform highp mat4 matrix;\n"
                                            "void main() {\n"
                                            "   col = colAttr;\n"
                                            "   gl_Position = matrix * posAttr;\n"
                                            "}\n";

    const char* fragmentShaderSource = "varying lowp vec4 col;\n"
                                              "void main() {\n"
                                              "   gl_FragColor = col;\n"
                                              "}\n";
*/
    const char* vertexShaderSource = "#version 330 core\n"
                                     "layout (location = 0) in vec3 aPos;\n"
                                     "void main()\n"
                                     "{\n"
                                     "   gl_Position = vec4(aPos.x, aPos.y, aPos.z, 1.0);\n"
                                     "}\0";
    const char* fragmentShaderSource = "#version 330 core\n"
                                       "out vec4 FragColor;\n"
                                       "void main()\n"
                                       "{\n"
                                       "   FragColor = vec4(1.0f, 0.5f, 0.2f, 1.0f);\n"
                                       "}\n\0";

    unsigned int shaderProgram;
    unsigned int VBO, VAO, EBO;

    VectorViewGraphicsScene(IntVector2D const& displaySize, std::mutex& mutex, QObject* parent = nullptr)
        : QGraphicsScene(parent)
        , _displaySize(displaySize)
        , _mutex(mutex)
    {
        m_context = new QOpenGLContext;
        m_context->create();
        initializeOpenGLFunctions();

                        unsigned int vertexShader = glCreateShader(GL_VERTEX_SHADER);
        glShaderSource(vertexShader, 1, &vertexShaderSource, NULL);
        glCompileShader(vertexShader);
        // check for shader compile errors
        int success;
        char infoLog[512];
        glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &success);
        if (!success) {
            glGetShaderInfoLog(vertexShader, 512, NULL, infoLog);
            std::cout << "ERROR::SHADER::VERTEX::COMPILATION_FAILED\n" << infoLog << std::endl;
        }
        // fragment shader
        unsigned int fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
        glShaderSource(fragmentShader, 1, &fragmentShaderSource, NULL);
        glCompileShader(fragmentShader);
        // check for shader compile errors
        glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &success);
        if (!success) {
            glGetShaderInfoLog(fragmentShader, 512, NULL, infoLog);
            std::cout << "ERROR::SHADER::FRAGMENT::COMPILATION_FAILED\n" << infoLog << std::endl;
        }

        // link shaders
        shaderProgram = glCreateProgram();
        glAttachShader(shaderProgram, vertexShader);
        glAttachShader(shaderProgram, fragmentShader);
        glLinkProgram(shaderProgram);
        // check for linking errors
        glGetProgramiv(shaderProgram, GL_LINK_STATUS, &success);
        if (!success) {
            glGetProgramInfoLog(shaderProgram, 512, NULL, infoLog);
            std::cout << "ERROR::SHADER::PROGRAM::LINKING_FAILED\n" << infoLog << std::endl;
        }
        glDeleteShader(vertexShader);
        glDeleteShader(fragmentShader);

        // set up vertex data (and buffer(s)) and configure vertex attributes
        // ------------------------------------------------------------------
        glGenVertexArrays(1, &VAO);
        glGenBuffers(1, &VBO);
        glGenBuffers(1, &EBO);
        // bind the Vertex Array Object first, then bind and set vertex buffer(s), and then configure vertex attributes(s).
        glBindVertexArray(VAO);

        /*
        m_program = new QOpenGLShaderProgram(this);
        m_program->addShaderFromSourceCode(QOpenGLShader::Vertex, vertexShaderSource);
        m_program->addShaderFromSourceCode(QOpenGLShader::Fragment, fragmentShaderSource);
        m_program->link();
        m_posAttr = m_program->attributeLocation("posAttr");
        Q_ASSERT(m_posAttr != -1);
        m_colAttr = m_program->attributeLocation("colAttr");
        Q_ASSERT(m_colAttr != -1);
        m_matrixUniform = m_program->uniformLocation("matrix");
        Q_ASSERT(m_matrixUniform != -1);

        glGenBuffers(1, &VBO);
        glBindBuffer(GL_ARRAY_BUFFER, VBO);

        float vertices[] = {-0.5f, -0.5f, 0.0f, 0.5f, -0.5f, 0.0f, 0.0f, 0.5f, 0.0f};
        glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
*/
        _image = boost::make_shared<QImage>(_displaySize.x, _displaySize.y, QImage::Format_ARGB32);

    }

    void resize(IntVector2D const& size) { _image = boost::make_shared<QImage>(size.x, size.y, QImage::Format_ARGB32); }

    QImagePtr getImageOfVisibleRect()
    {
        return _image;
    }

    void drawBackground(QPainter* painter, const QRectF& rect) override
    {
        std::lock_guard<std::mutex> lock(_mutex);

        glBindBuffer(GL_ARRAY_BUFFER, VBO);
        glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
        glEnableVertexAttribArray(0);

        // note that this is allowed, the call to glVertexAttribPointer registered VBO as the vertex attribute's bound vertex buffer object so afterwards we can safely unbind
        glBindBuffer(GL_ARRAY_BUFFER, 0);

        // remember: do NOT unbind the EBO while a VAO is active as the bound element buffer object IS stored in the VAO; keep the EBO bound.
        //glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

        // You can unbind the VAO afterwards so other VAO calls won't accidentally modify this VAO, but this rarely happens. Modifying other
        // VAOs requires a call to glBindVertexArray anyways so we generally don't unbind VAOs (nor VBOs) when it's not directly necessary.
        glBindVertexArray(0);
//--------
        glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        // draw our first triangle
        glUseProgram(shaderProgram);
        glBindVertexArray(
            VAO);  // seeing as we only have a single VAO there's no need to bind it every time, but we'll do so to keep things a bit more organized
        //glDrawArrays(GL_TRIANGLES, 0, 6);
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);

        /*
        glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        m_program->bind();

        QMatrix4x4 matrix;
//        matrix.perspective(00.0f, 4.0f / 3.0f, 0.1f, 100.0f);
        matrix.translate(0, 0, -1);
//        matrix.rotate(100.0f, 0, 1, 0);

        m_program->setUniformValue(m_matrixUniform, matrix);

        static const GLfloat vertices[] = {-1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f};
        static const GLfloat colors[] = {1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f};

        glVertexAttribPointer(m_posAttr, 2, GL_FLOAT, GL_FALSE, 0, vertices);
        glVertexAttribPointer(m_colAttr, 3, GL_FLOAT, GL_FALSE, 0, colors);

        glEnableVertexAttribArray(m_posAttr);
        glEnableVertexAttribArray(m_colAttr);

        glDrawArrays(GL_QUADS, 0, 4);

        glDisableVertexAttribArray(m_colAttr);
        glDisableVertexAttribArray(m_posAttr);

        m_program->release();
*/

        //        painter->drawImage(0, 0, *_image);
    }

private:
    QImagePtr _image = nullptr;
    IntVector2D _displaySize;
    std::mutex& _mutex;

    QOpenGLContext* m_context;
    GLint m_posAttr = 0;
    GLint m_colAttr = 0;
    GLint m_matrixUniform = 0;

    QOpenGLShaderProgram* m_program = nullptr;
};

VectorUniverseView::VectorUniverseView(QGraphicsView* graphicsView, QObject* parent)
    : UniverseView(graphicsView, parent)
{
}

void VectorUniverseView::init(
    Notifier* notifier,
    SimulationController* controller,
    SimulationAccess* access,
    DataRepository* repository)
{
    _graphicsView->setViewport(new QOpenGLWidget());
    _graphicsView->setViewportUpdateMode(QGraphicsView::FullViewportUpdate);

    disconnectView();
    _controller = controller;
    _repository = repository;
    _notifier = notifier;

    SET_CHILD(_access, access);

    auto width = _graphicsView->width();
    auto height = _graphicsView->height();

    delete _scene;
    _scene = new VectorViewGraphicsScene(IntVector2D{width, height}, repository->getImageMutex(), this);
    //    _scene->setBackgroundBrush(QBrush(Const::BackgroundColor));
    _scene->update();
    _scene->installEventFilter(this);
    _graphicsView->installEventFilter(this);
    _scene->setSceneRect(0, 0, width - 3, height - 3);
}

void VectorUniverseView::connectView()
{
    disconnectView();
    _connections.push_back(
        connect(_controller, &SimulationController::nextFrameCalculated, this, &VectorUniverseView::requestImage));
    _connections.push_back(
        connect(_notifier, &Notifier::notifyDataRepositoryChanged, this, &VectorUniverseView::receivedNotifications));
    _connections.push_back(
        connect(_repository, &DataRepository::imageReady, this, &VectorUniverseView::imageReady, Qt::QueuedConnection));
    _connections.push_back(
        connect(_graphicsView->horizontalScrollBar(), &QScrollBar::valueChanged, this, &VectorUniverseView::scrolled));
    _connections.push_back(
        connect(_graphicsView->verticalScrollBar(), &QScrollBar::valueChanged, this, &VectorUniverseView::scrolled));
}

void VectorUniverseView::disconnectView()
{
    for (auto const& connection : _connections) {
        disconnect(connection);
    }
    _connections.clear();
}

void VectorUniverseView::refresh()
{
    requestImage();
}

bool VectorUniverseView::isActivated() const
{
    return _graphicsView->scene() == _scene;
}

void VectorUniverseView::activate(double zoomFactor)
{
    _graphicsView->setViewportUpdateMode(QGraphicsView::NoViewportUpdate);
    _graphicsView->setScene(_scene);
    _graphicsView->resetTransform();

    setZoomFactor(zoomFactor);
}

double VectorUniverseView::getZoomFactor() const
{
    return _zoomFactor;
}

void VectorUniverseView::setZoomFactor(double zoomFactor)
{
    _zoomFactor = zoomFactor;
}

void VectorUniverseView::setZoomFactor(double zoomFactor, QVector2D const& fixedPos)
{
    auto worldPosOfScreenCenter = getCenterPositionOfScreen();
    auto origZoomFactor = _zoomFactor;

    _zoomFactor = zoomFactor;
    QVector2D mu(
        worldPosOfScreenCenter.x() * (zoomFactor / origZoomFactor - 1.0),
        worldPosOfScreenCenter.y() * (zoomFactor / origZoomFactor - 1.0));
    QVector2D correction(
        mu.x() * (worldPosOfScreenCenter.x() - fixedPos.x()) / worldPosOfScreenCenter.x(),
        mu.y() * (worldPosOfScreenCenter.y() - fixedPos.y()) / worldPosOfScreenCenter.y());
    centerTo(worldPosOfScreenCenter - correction);
}

QVector2D VectorUniverseView::getCenterPositionOfScreen() const
{
    return _center;
}

void VectorUniverseView::centerTo(QVector2D const& position)
{
    _center = position;
}

bool VectorUniverseView::eventFilter(QObject* object, QEvent* event)
{
    if (object == _scene) {
        if (event->type() == QEvent::GraphicsSceneMousePress) {
            mousePressEvent(static_cast<QGraphicsSceneMouseEvent*>(event));
        }

        if (event->type() == QEvent::GraphicsSceneMouseMove) {
            mouseMoveEvent(static_cast<QGraphicsSceneMouseEvent*>(event));
        }

        if (event->type() == QEvent::GraphicsSceneMouseRelease) {
            mouseReleaseEvent(static_cast<QGraphicsSceneMouseEvent*>(event));
        }
    }

    if (object = _graphicsView) {
        if (event->type() == QEvent::Resize) {
            resize(static_cast<QResizeEvent*>(event));
        }
    }
    return false;
}

void VectorUniverseView::mousePressEvent(QGraphicsSceneMouseEvent* event)
{
    auto pos = mapViewToWorldPosition(QVector2D(event->scenePos().x(), event->scenePos().y()));

    if (event->buttons() == Qt::MouseButton::LeftButton) {
        Q_EMIT startContinuousZoomIn(pos);
    }
    if (event->buttons() == Qt::MouseButton::RightButton) {
        Q_EMIT startContinuousZoomOut(pos);
    }
    /*
    if (!_controller->getRun()) {
        QVector2D pos(event->scenePos().x() / _zoomFactor, event->scenePos().y() / _zoomFactor);
        _access->selectEntities(pos);
        requestImage();
    }
*/
}

void VectorUniverseView::mouseMoveEvent(QGraphicsSceneMouseEvent* e)
{
    /*
    auto const pos = QVector2D(e->scenePos().x() / _zoomFactor, e->scenePos().y() / _zoomFactor);
    auto const lastPos = QVector2D(e->lastScenePos().x() / _zoomFactor, e->lastScenePos().y() / _zoomFactor);

    if (_controller->getRun()) {
        if (e->buttons() == Qt::MouseButton::LeftButton) {
            auto const force = (pos - lastPos) / 10;
            _access->applyAction(boost::make_shared<_ApplyForceAction>(lastPos, pos, force));
        }
        if (e->buttons() == Qt::MouseButton::RightButton) {
            auto const force = (pos - lastPos) / 10;
            _access->applyAction(boost::make_shared<_ApplyRotationAction>(lastPos, pos, force));
        }
    }
    else {
        if (e->buttons() == Qt::MouseButton::LeftButton) {
            auto const displacement = pos - lastPos;
            _access->applyAction(boost::make_shared<_MoveSelectionAction>(displacement));
            requestImage();
        }
    }
*/
}

void VectorUniverseView::mouseReleaseEvent(QGraphicsSceneMouseEvent* event)
{
    Q_EMIT endContinuousZoom();

/*
    if (!_controller->getRun()) {
        _access->deselectAll();
        requestImage();
    }
*/
}

void VectorUniverseView::resize(QResizeEvent* event)
{
    auto size = event->size();
    _scene->resize({size.width(), size.height()});
    _scene->setSceneRect(QRect(QPoint(0, 0), QPoint(size.width() - 3, size.height() - 3)));
}

void VectorUniverseView::receivedNotifications(set<Receiver> const& targets)
{
    if (targets.find(Receiver::VisualEditor) == targets.end()) {
        return;
    }

    requestImage();
}

void VectorUniverseView::requestImage()
{
    if (!_connections.empty()) {
        auto topLeft = mapViewToWorldPosition(QVector2D(0, 0));
        auto bottomRight = mapViewToWorldPosition(QVector2D(_graphicsView->width() - 1, _graphicsView->height() - 1));
        RealRect rect{RealVector2D(topLeft), RealVector2D(bottomRight)};
        _repository->requireVectorImageFromSimulation(rect, _zoomFactor, _scene->getImageOfVisibleRect());
    }
}

void VectorUniverseView::imageReady()
{
    _scene->update();
}

void VectorUniverseView::scrolled()
{
    requestImage();
}

QVector2D VectorUniverseView::mapViewToWorldPosition(QVector2D const& viewPos) const
{
    QVector2D relCenter(
        static_cast<float>(_graphicsView->width() / (2.0 * _zoomFactor)),
        static_cast<float>(_graphicsView->height() / (2.0 * _zoomFactor)));
    QVector2D relWorldPos(viewPos.x() / _zoomFactor, viewPos.y() / _zoomFactor);
    return _center - relCenter + relWorldPos;
}
