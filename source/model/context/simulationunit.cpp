#include <QFile>

#include "global/numbergenerator.h"

#include "model/physics/physics.h"
#include "model/entities/cell.h"
#include "model/entities/cellcluster.h"
#include "model/entities/energyparticle.h"
#include "model/entities/token.h"
#include "simulationunitcontext.h"

#include "simulationunit.h"

SimulationUnit::SimulationUnit (QObject* parent)
    : QObject(parent)
{
}

SimulationUnit::~SimulationUnit ()
{
}

void SimulationUnit::init(SimulationUnitContext* context)
{
	_context = context;
}

qreal SimulationUnit::calcTransEnergy ()
{

    qreal transEnergy(0.0);
    foreach(CellCluster* cluster, _context->getClustersRef()) {
		if (!cluster->isEmpty()) {
			transEnergy += Physics::kineticEnergy(cluster->getCellsRef().size(), cluster->getVelocity(), 0.0, 0.0);
		}
    }
    return transEnergy;
}

qreal SimulationUnit::calcRotEnergy ()
{
    qreal rotEnergy(0.0);
	foreach(CellCluster* cluster, _context->getClustersRef()) {
		if (cluster->getMass() > 1.0) {
			rotEnergy += Physics::kineticEnergy(0.0, QVector3D(), cluster->getAngularMass(), cluster->getAngularVel());
		}
    }
    return rotEnergy;
}

qreal SimulationUnit::calcInternalEnergy ()
{
    qreal internalEnergy(0.0);
    foreach( CellCluster* cluster, _context->getClustersRef() ) {
        if( !cluster->isEmpty() ) {
            foreach( Cell* cell, cluster->getCellsRef() ) {
                internalEnergy += cell->getEnergyIncludingTokens();
            }
        }
    }
    foreach( EnergyParticle* energyParticle, _context->getEnergyParticlesRef() ) {
        internalEnergy += energyParticle->getEnergy();
    }
    return internalEnergy;
}


void SimulationUnit::calcNextTimestep ()
{
	
	_context->lock();

	processingClusterInit();
	processingClusterDissipation();
	processingClusterMutationByChance();
	processingClusterMovement();
	processingClusterToken();
	processingClusterCompletion();

	processingEnergyParticles();

    _context->unlock();

    emit nextTimestepCalculated();
}

void SimulationUnit::processingEnergyParticles()
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

void SimulationUnit::processingClusterCompletion()
{
	foreach(CellCluster* cluster, _context->getClustersRef()) {
		cluster->processingCompletion();
		debugCluster(cluster, 5);
	}
}

void SimulationUnit::processingClusterToken()
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
			debugCluster(cluster, 4);

			//decompose cluster?
			if (decompose) {
				j.remove();
				QList< CellCluster* > newClusters = cluster->decompose();
				foreach(CellCluster* newCluster, newClusters) {
					debugCluster(newCluster, 4);
					j.insert(newCluster);
				}
			}
		}
	}
}

void SimulationUnit::processingClusterMovement()
{
	foreach(CellCluster* cluster, _context->getClustersRef()) {
		cluster->processingMovement();
		debugCluster(cluster, 3);
	}
}

void SimulationUnit::processingClusterMutationByChance()
{
	foreach(CellCluster* cluster, _context->getClustersRef()) {
		cluster->processingMutationByChance();
		debugCluster(cluster, 3);
	}
}

void SimulationUnit::processingClusterDissipation()
{
	QMutableListIterator<CellCluster*> i(_context->getClustersRef());
	QList< EnergyParticle* > energyParticles;
	while (i.hasNext()) {

		QList< CellCluster* > fragments;
		CellCluster* cluster(i.next());
		energyParticles.clear();
		cluster->processingDissipation(fragments, energyParticles);
		_context->getEnergyParticlesRef() << energyParticles;

		debugCluster(cluster, 2);
		//new cell cluster fragments?
//        bool delCluster = false;
		if ((!fragments.empty()) || (cluster->isEmpty())) {
			//            delCluster = true;
			delete cluster;
			i.remove();
			foreach(CellCluster* cluster2, fragments) {
				debugCluster(cluster2, 2);
				i.insert(cluster2);
			}
		}
	}
}

void SimulationUnit::processingClusterInit()
{
	foreach(CellCluster* cluster, _context->getClustersRef()) {
		cluster->processingInit();
	}
}

void SimulationUnit::debugCluster (CellCluster* c, int s)
{
    /*foreach(Cell* cell, c->getCellsRef()) {
        for(int i = 0; i < cell->getNumConnections(); ++i) {
            QVector3D d = cell->getRelPosition()-cell->getConnection(i)->getRelPosition();
            if( d.length() > (CRIT_CELL_DIST_MAX+0.1) )
                qDebug("%d: %f", s, d.length());
        }
    }*/
}




