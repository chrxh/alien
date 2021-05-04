#pragma once

#include <QGraphicsScene>
#include <QVector2D>
#include <QTimer>

#include "EngineInterface/Definitions.h"
#include "EngineInterface/Descriptions.h"
#include "UniverseView.h"
#include "Definitions.h"

class OpenGLUniverseScene;
class QResizeEvent;

class OpenGLUniverseView : public UniverseView
{
    Q_OBJECT
public:
    OpenGLUniverseView(QGraphicsView* graphicsView, QObject* parent = nullptr);
    virtual ~OpenGLUniverseView() = default;

    virtual void init(
        Notifier* notifier,
        SimulationController* controller,
        SimulationAccess* access,
        DataRepository* repository);

    void connectView() override;
    void disconnectView() override;
    void refresh() override;

    bool isActivated() const override;
    void activate(double zoomFactor) override;

    double getZoomFactor() const override;
    void setZoomFactor(double zoomFactor) override;
    void setZoomFactor(double zoomFactor, IntVector2D const& viewPos) override;

    QVector2D getCenterPositionOfScreen() const override;

    void centerTo(QVector2D const& position) override;

protected:
    bool eventFilter(QObject* object, QEvent* event) override;
    void mousePressEvent(QGraphicsSceneMouseEvent *event);
    void mouseMoveEvent(QGraphicsSceneMouseEvent* event);
    void mouseReleaseEvent(QGraphicsSceneMouseEvent *event);
    void resize(QResizeEvent* event);

private:
    void centerTo(QVector2D const& worldPosition, IntVector2D const& viewPos);

    Q_SLOT void receivedNotifications(set<Receiver> const& targets);
    Q_SLOT void requestImage();
    Q_SLOT void imageReady();
    Q_SLOT void scrolled();
    Q_SLOT void updateViewTimeout();

    QVector2D mapViewToWorldPosition(QVector2D const& viewPos) const;
    QVector2D mapDeltaViewToDeltaWorldPosition(QVector2D const& viewPos) const;

    list<QMetaObject::Connection> _connections;

    OpenGLUniverseScene* _scene = nullptr;

    SimulationAccess* _access = nullptr;
    DataRepository* _repository = nullptr;
    SimulationController* _controller = nullptr;
    QOpenGLWidget* _viewport = nullptr;

    Notifier* _notifier = nullptr;
    QTimer _updateViewTimer;
    int _scheduledViewUpdates = 0;

    double _zoomFactor = 0.0;
    QVector2D _center;

    //navigation
    boost::optional<QVector2D> _worldPosForMovement;
};

