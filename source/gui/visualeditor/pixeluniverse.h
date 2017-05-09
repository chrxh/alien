#ifndef PIXELUNIVERSE_H
#define PIXELUNIVERSE_H

#include <QGraphicsScene>
#include <QVector2D>
#include <QTimer>

#include "model/Definitions.h"

class PixelUniverse : public QGraphicsScene
{
    Q_OBJECT
public:
    PixelUniverse(QObject* parent=0);
    ~PixelUniverse();

	virtual void init(SimulationController* controller);

    void reset ();

private:
    Q_SLOT void requestData();
	Q_SLOT void retrieveAndDisplayData();

	void displayClusters(DataDescription const& data) const;

	SimulationAccess* _simAccess = nullptr;
    QGraphicsPixmapItem* _pixmap = nullptr;
    QImage* _image = nullptr;

};

#endif // PIXELUNIVERSE_H
