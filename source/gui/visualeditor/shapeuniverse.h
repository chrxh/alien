 #ifndef SHAPEUNIVERSE_H
#define SHAPEUNIVERSE_H

#include <QGraphicsScene>
#include <QGraphicsEllipseItem>
#include <QGraphicsLineItem>
#include <QMap>
#include <QVector2D>

#include "model/Definitions.h"
#include "model/Entities/CellTO.h"
#include "gui/Definitions.h"


class ShapeUniverse : public QGraphicsScene
{
    Q_OBJECT
public:
    ShapeUniverse (QObject *parent = 0);
	virtual ~ShapeUniverse();

};

#endif // SHAPEUNIVERSE_H
