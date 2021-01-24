#pragma once

#include <QGraphicsScene>
#include <QVector2D>
#include <QTimer>

#include "ModelBasic/Definitions.h"
#include "ModelBasic/Descriptions.h"
#include "Definitions.h"

class VectorUniverseView : public QGraphicsScene
{
    Q_OBJECT
public:
    VectorUniverseView(QObject* parent = nullptr);
    virtual ~VectorUniverseView();

    virtual void init(
        Notifier* notifier,
        SimulationController* controller,
        SimulationAccess* access,
        DataRepository* manipulator,
        ViewportInterface* viewport);
    virtual void activate();
    virtual void deactivate();

    virtual void refresh();

protected:
    void mousePressEvent(QGraphicsSceneMouseEvent *event) override;
    void mouseMoveEvent(QGraphicsSceneMouseEvent* e) override;
    void mouseReleaseEvent(QGraphicsSceneMouseEvent *event) override;

private:
    Q_SLOT void receivedNotifications(set<Receiver> const& targets);
    Q_SLOT void requestImage();
    Q_SLOT void imageReady();
    Q_SLOT void scrolled();

    list<QMetaObject::Connection> _connections;

    SimulationAccess* _access = nullptr;
    DataRepository* _repository = nullptr;
    SimulationController* _controller = nullptr;
    ViewportInterface* _viewport = nullptr;

    VectorImageSectionItem* _imageSectionItem = nullptr;
    bool _isActivated = false;

    Notifier* _notifier = nullptr;
};

