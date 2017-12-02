#pragma once

#include <QMatrix>
#include "ViewportInterface.h"

class ViewportController
	: public ViewportInterface
{
	Q_OBJECT
public:
	ViewportController(QObject* parent = nullptr) : ViewportInterface(parent) {}
	virtual ~ViewportController() = default;

	virtual void init(QGraphicsView* view, QGraphicsScene* pixelScene, QGraphicsScene* shapeScene, ActiveScene activeScene);

	virtual void setModeToUpdate() override;
	virtual void setModeToNoUpdate() override;

	virtual void setActiveScene(ActiveScene activeScene);
	virtual ActiveScene getActiveScene() const;

	virtual QRectF getRect() const override;
	virtual QVector2D getCenter() const override;

	virtual void zoomIn();
	virtual void zoomOut();
	virtual qreal getZoomFactor() const;

	virtual void saveScrollPos();
	virtual void restoreScrollPos();

private:
	void setSceneToView(optional<ActiveScene> oldActiveScene, ActiveScene activeScene);


	QGraphicsView* _view = nullptr;
	QGraphicsScene* _pixelScene = nullptr;
	QGraphicsScene* _itemScene = nullptr;

	IntVector2D _sceneScrollbarPos;

	optional<ActiveScene> _activeScene;
};
