#include <QFile>

#include "global/NumberGenerator.h"

#include "model/physics/Physics.h"
#include "model/entities/Cell.h"
#include "model/entities/CellCluster.h"
#include "model/entities/EnergyParticle.h"
#include "model/entities/Token.h"
#include "model/context/UnitContext.h"

#include "UnitImpl.h"

UnitImpl::UnitImpl(QObject* parent)
	: Unit(parent)
{
}

void UnitImpl::init(UnitContext* context)
{
	_context = context;
}

qreal UnitImpl::calcTransEnergy() const
{

	qreal transEnergy(0.0);
	foreach(CellCluster* cluster, _context->getClustersRef()) {
		if (!cluster->isEmpty()) {
			transEnergy += Physics::kineticEnergy(cluster->getCellsRef().size(), cluster->getVelocity(), 0.0, 0.0);
		}
	}
	return transEnergy;
}

qreal UnitImpl::calcRotEnergy() const
{
	qreal rotEnergy(0.0);
	foreach(CellCluster* cluster, _context->getClustersRef()) {
		if (cluster->getMass() > 1.0) {
			rotEnergy += Physics::kineticEnergy(0.0, QVector3D(), cluster->getAngularMass(), cluster->getAngularVel());
		}
	}
	return rotEnergy;
}

qreal UnitImpl::calcInternalEnergy() const
{
	qreal internalEnergy(0.0);
	foreach(CellCluster* cluster, _context->getClustersRef()) {
		if (!cluster->isEmpty()) {
			foreach(Cell* cell, cluster->getCellsRef()) {
				internalEnergy += cell->getEnergyIncludingTokens();
			}
		}
	}
	foreach(EnergyParticle* energyParticle, _context->getEnergyParticlesRef()) {
		internalEnergy += energyParticle->getEnergy();
	}
	return internalEnergy;
}

void UnitImpl::calcNextTimestep()
{
	processingClusterInit();
	processingClusterDissipation();
	processingClusterMutationByChance();
	processingClusterMovement();
	processingClusterToken();
	processingClusterCompletion();

	processingEnergyParticles();

	emit nextTimestepCalculated();
}

UnitContext * UnitImpl::getContext() const
{
	return _context;
}

void UnitImpl::processingEnergyParticles()
{
	QMutableListIterator<EnergyParticle*> p(_context->getEnergyParticlesRef());
	while (p.hasNext()) {
		EnergyParticle* e(p.next());
		CellCluster* cluster(0);
		if (!e->processingMovement(cluster)) {

			//transform into cell?
			if (cluster) {
				_context->getClustersRef() << cluster;
			}
			delete e;
			p.remove();
		}
	}
}

void UnitImpl::processingClusterCompletion()
{
	foreach(CellCluster* cluster, _context->getClustersRef()) {
		cluster->processingCompletion();
	}
}

void UnitImpl::processingClusterToken()
{
	QMutableListIterator<CellCluster*> j(_context->getClustersRef());
	QList< EnergyParticle* > energyParticles;
	while (j.hasNext()) {
		CellCluster* cluster(j.next());
		if (cluster->isEmpty()) {
			delete cluster;
			j.remove();
		}
		else {
			energyParticles.clear();
			bool decompose = false;
			cluster->processingToken(energyParticles, decompose);
			_context->getEnergyParticlesRef() << energyParticles;

			//decompose cluster?
			if (decompose) {
				j.remove();
				QList< CellCluster* > newClusters = cluster->decompose();
				foreach(CellCluster* newCluster, newClusters) {
					j.insert(newCluster);
				}
			}
		}
	}
}

void UnitImpl::processingClusterMovement()
{
	foreach(CellCluster* cluster, _context->getClustersRef()) {
		cluster->processingMovement();
	}
}

void UnitImpl::processingClusterMutationByChance()
{
	foreach(CellCluster* cluster, _context->getClustersRef()) {
		cluster->processingMutationByChance();
	}
}

void UnitImpl::processingClusterDissipation()
{
	QMutableListIterator<CellCluster*> i(_context->getClustersRef());
	QList< EnergyParticle* > energyParticles;
	while (i.hasNext()) {

		QList< CellCluster* > fragments;
		CellCluster* cluster(i.next());
		energyParticles.clear();
		cluster->processingDissipation(fragments, energyParticles);
		_context->getEnergyParticlesRef() << energyParticles;

		//new cell cluster fragments?
		//        bool delCluster = false;
		if ((!fragments.empty()) || (cluster->isEmpty())) {
			//            delCluster = true;
			delete cluster;
			i.remove();
			foreach(CellCluster* cluster2, fragments) {
				i.insert(cluster2);
			}
		}
	}
}

void UnitImpl::processingClusterInit()
{
	foreach(CellCluster* cluster, _context->getClustersRef()) {
		cluster->processingInit();
	}
}



