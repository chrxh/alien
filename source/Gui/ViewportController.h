/*
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

	virtual void init(QGraphicsView* view, QGraphicsScene* pixelScene, QGraphicsScene* vectorScene, 
        QGraphicsScene* itemScene, ActiveView activeScene);

	virtual void setModeToUpdate() override;
	virtual void setModeToNoUpdate() override;

	virtual void setActiveScene(ActiveView activeScene);
	virtual ActiveView getActiveScene() const;

	virtual QRectF getRect() const override;
	virtual QVector2D getCenter() const override;

    virtual void zoom(double factor, bool notify = true);
	virtual qreal getZoomFactor() const override;

	virtual void scrollToPos(QVector2D pos, NotifyScrollChanged notify) override;
	virtual void saveScrollPos();
	virtual void restoreScrollPos();

private:
	void connectAll();
	void disconnectAll();

	void setSceneToView(optional<ActiveView> oldActiveScene, ActiveView activeScene);

	list<QMetaObject::Connection> _connections;

	QGraphicsView* _view = nullptr;
	QGraphicsScene* _pixelScene = nullptr;
    QGraphicsScene* _vectorScene = nullptr;
    QGraphicsScene* _itemScene = nullptr;

	double _sceneScrollbarPosX;
    double _sceneScrollbarPosY;
    optional<ActiveView> _activeScene;

    double _zoom = 1.0;
};
*/
