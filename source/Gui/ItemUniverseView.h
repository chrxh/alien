#pragma once

#include <QGraphicsScene>
#include <QGraphicsEllipseItem>
#include <QGraphicsLineItem>
#include <QMap>
#include <QVector2D>

#include "EngineInterface/Definitions.h"
#include "EngineInterface/Descriptions.h"
#include "Definitions.h"
#include "DataRepository.h"
#include "UniverseView.h"

class ItemUniverseView : public UniverseView
{
    Q_OBJECT
public:
    ItemUniverseView (QGraphicsView* graphicsView, QObject *parent = nullptr);
	virtual ~ItemUniverseView() = default;

    void init(Notifier* notifier, SimulationController* controller, DataRepository* manipulator);

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

	virtual void toggleCenterSelection(bool value);

protected:
    bool eventFilter(QObject* object, QEvent* event) override;
    void mousePressEvent(QGraphicsSceneMouseEvent* e);
	void mouseMoveEvent(QGraphicsSceneMouseEvent* e);
	void mouseReleaseEvent(QGraphicsSceneMouseEvent* e);

private:
	void requestData();
	boost::optional<QVector2D> getCenterPosOfSelection() const;
	void centerSelectionIfEnabled();
    void updateItems();

	Q_SLOT void receivedNotifications(set<Receiver> const& targets);
	Q_SLOT void cellInfoToggled(bool showInfo);
	Q_SLOT void scrolled();

	struct Selection
	{
		list<uint64_t> cellIds;
		list<uint64_t> particleIds;
	};
	Selection getSelectionFromItems(QList<QGraphicsItem*> const &items) const;
	void delegateSelection(Selection const& selection);
	void startMarking(QPointF const& scenePos);

    QGraphicsScene* _scene = nullptr;

	list<QMetaObject::Connection> _connections;

	SimulationController* _controller = nullptr;
	ItemViewport* _viewport = nullptr;

	ItemManager* _itemManager = nullptr;
	DataRepository* _repository = nullptr;
	Notifier* _notifier = nullptr;

    double _zoomFactor = 0.0;
    bool _mouseButtonPressed = true;
	bool _centerSelection = false;
};
