#ifndef MAPMANIPULATOR_H
#define MAPMANIPULATOR_H

#include "model/Definitions.h"
#include "CellDescription.h"

class MapManipulator
	: public QObject
{
	Q_OBJECT
public:
	MapManipulator(QObject* parent = nullptr) : QObject(parent) {}
	virtual ~MapManipulator() = default;

	virtual void addCell(CellDescription desc) = 0;
};

#endif // MAPMANIPULATOR_H
