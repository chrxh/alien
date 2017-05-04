#ifndef MAPMANIPULATORAPI_H
#define MAPMANIPULATORAPI_H

#include "model/Definitions.h"
#include "model/tools/CellDescription.h"

class SimulationManipulatorApi
	: public QObject
{
	Q_OBJECT
public:
	SimulationManipulatorApi(QObject* parent = nullptr) : QObject(parent) {}
	virtual ~SimulationManipulatorApi() = default;

	virtual void init(SimulationContextApi* context) = 0;

	virtual void addCell(CellDescription desc) = 0;
};

#endif // MAPMANIPULATORAPI_H
