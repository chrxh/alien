#pragma once

#include <QGraphicsScene>
#include <QVector2D>
#include <QTimer>

#include "EngineInterface/Definitions.h"
#include "EngineInterface/Descriptions.h"
#include "UniverseView.h"
#include "Definitions.h"

class VectorViewGraphicsScene;
class QResizeEvent;

class VectorUniverseView : public UniverseView
{
    Q_OBJECT
public:
    VectorUniverseView(QGraphicsView* graphicsView, QObject* parent = nullptr);
    virtual ~VectorUniverseView() = default;

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
    void setZoomFactor(double zoomFactor, QVector2D const& fixedPos) override;

    QVector2D getCenterPositionOfScreen() const override;

    void centerTo(QVector2D const& position) override;

protected:
    bool eventFilter(QObject* object, QEvent* event) override;
    void mousePressEvent(QGraphicsSceneMouseEvent *event);
    void mouseMoveEvent(QGraphicsSceneMouseEvent* e);
    void mouseReleaseEvent(QGraphicsSceneMouseEvent *event);
    void resize(QResizeEvent* event);

private:
    Q_SLOT void receivedNotifications(set<Receiver> const& targets);
    Q_SLOT void requestImage();
    Q_SLOT void imageReady();
    Q_SLOT void scrolled();

    QVector2D mapViewToWorldPosition(QVector2D const& viewPos) const;

    list<QMetaObject::Connection> _connections;

    VectorViewGraphicsScene* _scene = nullptr;

    SimulationAccess* _access = nullptr;
    DataRepository* _repository = nullptr;
    SimulationController* _controller = nullptr;

    Notifier* _notifier = nullptr;
    double _zoomFactor = 0.0;
    QVector2D _center;
};

