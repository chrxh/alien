#include <QFile>

#include "global/global.h"

#include "physics/physics.h"
#include "entities/cell.h"
#include "entities/cellcluster.h"
#include "entities/energyparticle.h"
#include "entities/token.h"

#include "simulationunit.h"
#include "simulationcontext.h"

SimulationUnit::SimulationUnit (SimulationContext* context, QObject* parent)
    : QObject(parent)
	, _context(context)
{
}

SimulationUnit::~SimulationUnit ()
{
}

void SimulationUnit::setRandomSeed(uint seed)
{
	qsrand(seed);
	qrand();
}

qreal SimulationUnit::calcTransEnergy ()
{
    qreal transEnergy(0.0);
    foreach(CellCluster* cluster, _context->getClustersRef()) {
        if( !cluster->isEmpty() )
            transEnergy += Physics::kineticEnergy(cluster->getCellsRef().size(),
                                                  cluster->getVel(),
                                                  0.0,
                                                  0.0);
    }
    return transEnergy;
}

qreal SimulationUnit::calcRotEnergy ()
{
    qreal rotEnergy(0.0);
    foreach( CellCluster* cluster, _context->getClustersRef() ) {
        if( cluster->getMass() > 1.0 )
            rotEnergy += Physics::kineticEnergy(0.0,
                                                QVector3D(0.0, 0.0, 0.0),
                                                cluster->getAngularMass(),
                                                cluster->getAngularVel());
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
        internalEnergy += energyParticle->amount;
    }
    return internalEnergy;
}


void SimulationUnit::calcNextTimestep ()
{
	
	_context->lock();

    //cell movement: step 1
    foreach( CellCluster* cluster, _context->getClustersRef()) {
        cluster->processingInit();
    }

    //cell movement: step 2
    //----
//    qreal eOld(calcInternalEnergy()+(calcTransEnergy()+calcRotEnergy())/INTERNAL_TO_KINETIC_ENERGY);
    //----
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
        if( (!fragments.empty()) || (cluster->isEmpty()) ) {
//            delCluster = true;
            delete cluster;
            i.remove();
            foreach( CellCluster* cluster2, fragments ) {
                debugCluster(cluster2, 2);
                i.insert(cluster2);
            }
        }
    }
    //----
//    qreal eNew(calcInternalEnergy()+(calcTransEnergy()+calcRotEnergy())/INTERNAL_TO_KINETIC_ENERGY);
//    if( ( qAbs(eOld-eNew) > 0.1 ) )
        //qDebug("step 2: old: %f new: %f, internal: %f, trans: %f, rot: %f", eOld, eNew, calcInternalEnergy(), calcTransEnergy(), calcRotEnergy());
    //----

    //cell movement: step 3
    foreach( CellCluster* cluster, _context->getClustersRef()) {
        cluster->processingMovement();
        debugCluster(cluster, 3);
    }


    //cell movement: step 4
    QMutableListIterator<CellCluster*> j(_context->getClustersRef());
    while (j.hasNext()) {
        CellCluster* cluster(j.next());
        if( cluster->isEmpty()) {
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
            if( decompose ) {
                j.remove();
                QList< CellCluster* > newClusters = cluster->decompose();
                foreach( CellCluster* newCluster, newClusters) {
                    debugCluster(newCluster, 4);
                    j.insert(newCluster);
                }
            }
        }
    }

    //cell movement: step 5
    foreach( CellCluster* cluster, _context->getClustersRef()) {
        cluster->processingFinish();
        debugCluster(cluster, 5);
    }

    //energy particle movement
    QMutableListIterator<EnergyParticle*> p(_context->getEnergyParticlesRef());
    while (p.hasNext()) {
        EnergyParticle* e(p.next());
        CellCluster* cluster(0);
        if( e->movement(cluster) ) {

            //transform into cell?
            if( cluster ) {
                _context->getClustersRef() << cluster;
            }
            delete e;
            p.remove();
        }
    }

    _context->unlock();

    emit nextTimestepCalculated();
}

void SimulationUnit::debugCluster (CellCluster* c, int s)
{
    /*foreach(Cell* cell, c->getCellsRef()) {
        for(int i = 0; i < cell->getNumConnections(); ++i) {
            QVector3D d = cell->getRelPos()-cell->getConnection(i)->getRelPos();
            if( d.length() > (CRIT_CELL_DIST_MAX+0.1) )
                qDebug("%d: %f", s, d.length());
        }
    }*/
}




//    msleep(100);
    //----------------
/*    foreach( CellCluster* cluster, _clusters) {
        QList< Cell* > component;
        quint64 tag(GlobalFunctions::getTag());
        cluster->getConnectedComponent(cluster->_cells[0], tag, component);
        if( component.size() != cluster->_cells.size() ) {
            qDebug("ALARM4");
        }
    }*/
    //----------------
