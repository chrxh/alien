#ifndef UNITCONTEXTAPI_H
#define UNITCONTEXTAPI_H

#include "model/Definitions.h"

class UnitContextApi
	: public QObject
{
	Q_OBJECT
public:
	UnitContextApi(QObject* parent = nullptr) : QObject(parent) {}
	virtual ~UnitContextApi() = default;

	virtual CellMap* getCellMap() const = 0;
	virtual EnergyParticleMap* getEnergyParticleMap() const = 0;
	virtual MapCompartment* getMapCompartment() const = 0;

	virtual QList<CellCluster*>& getClustersRef() = 0;
	virtual QList<EnergyParticle*>& getEnergyParticlesRef() = 0;
};

#endif // UNITCONTEXTAPI_H
