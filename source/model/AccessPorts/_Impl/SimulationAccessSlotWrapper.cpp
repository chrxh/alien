#include "SimulationAccessSlotWrapper.h"


void SimulationAccessSlotWrapper::init(SimulationContext* context)
{
	connect(context->getUnitThreadController(), &UnitThreadController::timestepCalculated
		, this, &SimulationAccessSlotWrapper::timestepCalculated);
}

void SimulationAccessSlotWrapper::timestepCalculated()
{
	accessToSimulation();
}
