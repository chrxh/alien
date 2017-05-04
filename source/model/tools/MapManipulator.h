#ifndef MAPMANIPULATOR_H
#define MAPMANIPULATOR_H

#include "model/Definitions.h"

class MapManipulator
	: public QObject
{
	Q_OBJECT
public:
	MapManipulator(QObject* parent = nullptr) : QObject(parent) {}
	virtual ~MapManipulator() = default;
};

#endif // MAPMANIPULATOR_H
