#ifndef SIMULATIONACCESS_H
#define SIMULATIONACCESS_H

#include "model/Definitions.h"

class SimulationAccessSignalWrapper
	: public QObject
{
	Q_OBJECT
public:
	SimulationAccessSignalWrapper(QObject* parent = nullptr) : QObject(parent) {}
	virtual ~SimulationAccessSignalWrapper() = default;

	Q_SIGNAL void dataReadyToRetrieve();
};

template<typename DataDescriptionType>
class SimulationAccess
	: public SimulationAccessSignalWrapper
{
public:
	SimulationAccess(QObject* parent = nullptr) : SimulationAccessSignalWrapper(parent) {}
	virtual ~SimulationAccess() = default;

	virtual void init(SimulationContextApi* context) = 0;

	virtual void addData(DataDescriptionType const &desc) = 0;
	virtual void removeData(DataDescriptionType const &desc) = 0;
	virtual void updateData(DataDescriptionType const &desc) = 0;

	virtual void requireData(IntRect rect) = 0;
	virtual DataDescriptionType const& retrieveData() = 0;
};

#endif // SIMULATIONACCESS_H
