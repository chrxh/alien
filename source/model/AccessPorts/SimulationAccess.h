#ifndef SIMULATIONACCESS_H
#define SIMULATIONACCESS_H

#include "model/Definitions.h"

template<typename DataDescriptionType>
class SimulationAccess
{
public:
	virtual ~SimulationAccess() = default;

	virtual void init(SimulationContextApi* context) = 0;

	virtual void addData(DataDescriptionType const &desc) = 0;
	virtual void removeData(DataDescriptionType const &desc) = 0;
	virtual void updateData(DataDescriptionType const &desc) = 0;
	virtual void getData(IntRect rect, DataDescriptionType& result) = 0;
};

#endif // SIMULATIONACCESS_H
