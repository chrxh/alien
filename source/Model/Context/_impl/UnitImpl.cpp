#include "Model/Physics/Physics.h"
#include "Model/Entities/Cell.h"
#include "Model/Entities/Cluster.h"
#include "Model/Entities/Particle.h"
#include "Model/Entities/Token.h"
#include "Model/Context/UnitContext.h"
#include "Model/Context/SpaceMetric.h"
#include "Model/Context/MapCompartment.h"

#include "UnitImpl.h"

UnitImpl::UnitImpl(QObject* parent)
	: Unit(parent)
{
}

UnitImpl::~UnitImpl()
{
}

void UnitImpl::init(UnitContext* context)
{
	SET_CHILD(_context, context);
}

qreal UnitImpl::calcTransEnergy() const
{

	qreal transEnergy(0.0);
	foreach(Cluster* cluster, _context->getClustersRef()) {
		if (!cluster->isEmpty()) {
			transEnergy += Physics::kineticEnergy(cluster->getCellsRef().size(), cluster->getVelocity(), 0.0, 0.0);
		}
	}
	return transEnergy;
}

qreal UnitImpl::calcRotEnergy() const
{
	qreal rotEnergy(0.0);
	foreach(Cluster* cluster, _context->getClustersRef()) {
		if (cluster->getMass() > 1.0) {
			rotEnergy += Physics::kineticEnergy(0.0, QVector2D(), cluster->getAngularMass(), cluster->getAngularVel());
		}
	}
	return rotEnergy;
}

qreal UnitImpl::calcInternalEnergy() const
{
	qreal internalEnergy(0.0);
	foreach(Cluster* cluster, _context->getClustersRef()) {
		if (!cluster->isEmpty()) {
			foreach(Cell* cell, cluster->getCellsRef()) {
				internalEnergy += cell->getEnergyIncludingTokens();
			}
		}
	}
	foreach(Particle* energyParticle, _context->getEnergyParticlesRef()) {
		internalEnergy += energyParticle->getEnergy();
	}
	return internalEnergy;
}

void UnitImpl::calculateTimestep()
{

	processingClustersInit();
	processingClustersDissipation();
	processingClustersMutationByChance();
	processingClustersMovement();
	processingClustersToken();
	processingClustersCompletion();
	processingParticlesMovement();

	incClustersTimestamp();
	incParticlesTimestamp();
	_context->incTimestamp();

	processingParticlesCompartmentAllocation();
	processingClustersCompartmentAllocation();

	Q_EMIT timestepCalculated();
}

UnitContext * UnitImpl::getContext() const
{
	return _context;
}

void UnitImpl::processingClustersCompletion()
{
	foreach(Cluster* cluster, _context->getClustersRef()) {
		cluster->processingCompletion();
	}
}

void UnitImpl::processingClustersToken()
{
	QMutableListIterator<Cluster*> j(_context->getClustersRef());
	QList< Particle* > energyParticles;
	while (j.hasNext()) {
		Cluster* cluster(j.next());
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
				QList< Cluster* > newClusters = cluster->decompose();
				foreach(Cluster* newCluster, newClusters) {
					j.insert(newCluster);
				}
			}
		}
	}
}

void UnitImpl::processingClustersMovement()
{
	foreach(Cluster* cluster, _context->getClustersRef()) {
		cluster->processingMovement();
	}
}

void UnitImpl::processingClustersMutationByChance()
{
	foreach(Cluster* cluster, _context->getClustersRef()) {
		cluster->processingMutationByChance();
	}
}

void UnitImpl::processingClustersDissipation()
{
	QMutableListIterator<Cluster*> i(_context->getClustersRef());
	QList< Particle* > energyParticles;
	while (i.hasNext()) {

		QList< Cluster* > fragments;
		Cluster* cluster(i.next());
		energyParticles.clear();
		cluster->processingDissipation(fragments, energyParticles);
		_context->getEnergyParticlesRef() << energyParticles;

		//new cell cluster fragments?
		//        bool delCluster = false;
		if ((!fragments.empty()) || (cluster->isEmpty())) {
			//            delCluster = true;
			delete cluster;
			i.remove();
			foreach(Cluster* cluster2, fragments) {
				i.insert(cluster2);
			}
		}
	}
}

void UnitImpl::processingClustersInit()
{
	foreach(Cluster* cluster, _context->getClustersRef()) {
		cluster->processingInit();
	}
}

void UnitImpl::processingClustersCompartmentAllocation()
{
	auto compartment = _context->getMapCompartment();
	auto spaceMetric = _context->getSpaceMetric();

	QMutableListIterator<Cluster*> clusterIter(_context->getClustersRef());
	while (clusterIter.hasNext()) {
		Cluster* cluster = clusterIter.next();
		IntVector2D intPos = spaceMetric->correctPositionAndConvertToIntVector(cluster->getPosition());
		if (!compartment->isPointInCompartment(intPos)) {
			clusterIter.remove();
			auto otherContext = compartment->getNeighborContext(intPos);
			otherContext->getClustersRef().push_back(cluster);
			cluster->setContext(otherContext);
		}
	}
}

void UnitImpl::processingParticlesMovement()
{
	QMutableListIterator<Particle*> p(_context->getEnergyParticlesRef());
	while (p.hasNext()) {
		Particle* e(p.next());
		Cluster* cluster(0);
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

void UnitImpl::incClustersTimestamp()
{
	for(auto const& cluster : _context->getClustersRef()) {
		cluster->incTimestampIfFit();
	}
}

void UnitImpl::incParticlesTimestamp()
{
	for (auto const& particle : _context->getEnergyParticlesRef()) {
		particle->incTimestampIfFit();
	}
}

void UnitImpl::processingParticlesCompartmentAllocation()
{
	auto compartment = _context->getMapCompartment();
	auto spaceMetric = _context->getSpaceMetric();

	QMutableListIterator<Particle*> particleIter(_context->getEnergyParticlesRef());
	while (particleIter.hasNext()) {
		Particle* particle = particleIter.next();
		IntVector2D intPos = spaceMetric->correctPositionAndConvertToIntVector(particle->getPosition());
		if (!compartment->isPointInCompartment(intPos)) {
			particleIter.remove();
			auto otherContext = compartment->getNeighborContext(intPos);
			otherContext->getEnergyParticlesRef().push_back(particle);
			particle->setContext(otherContext);
		}
	}
}
