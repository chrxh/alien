#ifndef MAPMANIPULATORIMPL_H
#define MAPMANIPULATORIMPL_H

#include "model/tools/MapManipulator.h"

class MapManipulatorImpl
	: public MapManipulator
{
	Q_OBJECT
public:
	MapManipulatorImpl(QObject* parent = nullptr) : MapManipulator(parent) {}
	virtual ~MapManipulatorImpl() = default;
};

#endif // MAPMANIPULATORIMPL_H
