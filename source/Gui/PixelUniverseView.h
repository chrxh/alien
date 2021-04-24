#pragma once

#include <QGraphicsScene>
#include <QVector2D>
#include <QTimer>

#include "EngineInterface/Definitions.h"
#include "EngineInterface/Descriptions.h"

#include "Gui/Definitions.h"
#include "UniverseView.h"

class PixelUniverseView : public UniverseView
{
    Q_OBJECT
public:
    PixelUniverseView(QGraphicsView* graphicsView, QObject* parent = nullptr);
    virtual ~PixelUniverseView() = default;

    void init(
        Notifier* notifier,
        SimulationController* controller,
        SimulationAccess* access,
        DataRepository* manipulator);

    void connectView() override;
    void disconnectView() override;
    void refresh() override;

    bool isActivated() const override;
    void activate(double zoomFactor) override;

    double getZoomFactor() const override;
    void setZoomFactor(double zoomFactor) override;
    void setZoomFactor(double zoomFactor, QVector2D const& fixedPos) {}

    QVector2D getCenterPositionOfScreen() const override;

    void centerTo(QVector2D const& position) override;

protected:
    bool eventFilter(QObject* object, QEvent* event) override;
    void mousePressEvent(QGraphicsSceneMouseEvent *event);
	void mouseMoveEvent(QGraphicsSceneMouseEvent* e);
    void mouseReleaseEvent(QGraphicsSceneMouseEvent *event) ;

private:
	Q_SLOT void receivedNotifications(set<Receiver> const& targets);
	Q_SLOT void requestImage();
	Q_SLOT void imageReady();
	Q_SLOT void scrolled();

	list<QMetaObject::Connection> _connections;

    QGraphicsScene* _scene = nullptr;

    SimulationAccess* _access = nullptr;
	DataRepository* _repository = nullptr;
	SimulationController* _controller = nullptr;
    PixelViewport* _viewport = nullptr;

    PixelImageSectionItem* _imageSectionItem = nullptr;

	Notifier* _notifier = nullptr;
    double _zoomFactor = 0.0;
};

