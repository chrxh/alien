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
    PixelUniverse(QObject* parent=0);
    virtual ~PixelUniverse();

	virtual void init(SimulationController* controller, ViewportInfo* viewport);

    void reset ();

private:
	void requestAllData();
    Q_SLOT void requestData();
	Q_SLOT void retrieveAndDisplayData();

	void displayClusters(DataDescription const& data) const;
	void displayparticles(DataDescription const& data) const;

	SimulationAccess* _simAccess = nullptr;
	ViewportInfo* _viewport = nullptr;
    QGraphicsPixmapItem* _pixmap = nullptr;
    QImage* _image = nullptr;

};

#endif // PIXELUNIVERSE_H
