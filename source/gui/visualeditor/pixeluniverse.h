#ifndef PIXELUNIVERSE_H
#define PIXELUNIVERSE_H

#include <QGraphicsScene>
#include <QVector2D>
#include <QTimer>

#include "gui/Definitions.h"
#include "model/Definitions.h"

class PixelUniverse : public QGraphicsScene
{
    Q_OBJECT
public:
    PixelUniverse(QObject* parent = nullptr);
    virtual ~PixelUniverse();

	virtual void init(SimulationController* controller, ViewportInfo* viewport);
	virtual void setActive();
	virtual void setInactive();

private:
    Q_SLOT void requestData();
	Q_SLOT void retrieveAndDisplayData();

	void displayClusters(DataDescription const& data) const;
	void displayParticles(DataDescription const& data) const;

	SimulationAccess* _simAccess = nullptr;
	SimulationController* _controller = nullptr;
	ViewportInfo* _viewport = nullptr;
    QGraphicsPixmapItem* _pixmap = nullptr;
    QImage* _image = nullptr;

};

#endif // PIXELUNIVERSE_H
