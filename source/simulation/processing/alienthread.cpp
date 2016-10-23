#include "alienthread.h"
#include "../entities/aliencell.h"
#include "../entities/aliencellcluster.h"
#include "../entities/aliengrid.h"
#include "../entities/alientoken.h"
#include "../../global/globalfunctions.h"
#include "../physics/physics.h"

#include<QThread>


AlienThread::AlienThread (QObject* parent)
    : QThread(parent), _space(0)
{

}

AlienThread::~AlienThread ()
{
}

void AlienThread::init (AlienGrid* space)
{
    _space = space;
}

QList< AlienCellCluster* >& AlienThread::getClusters ()
{
    return _space->getClusters();
}

QList< AlienEnergy* >& AlienThread::getEnergyParticles ()
{
    return _space->getEnergyParticles();
}

qreal AlienThread::calcTransEnergy ()
{
    qreal transEnergy(0.0);
    foreach( AlienCellCluster* cluster, _space->getClusters() ) {
        if( !cluster->isEmpty() )
            transEnergy += Physics::kineticEnergy(cluster->getCells().size(),
                                                  cluster->getVel(),
                                                  0.0,
                                                  0.0);
    }
    return transEnergy;
}

qreal AlienThread::calcRotEnergy ()
{
    qreal rotEnergy(0.0);
    foreach( AlienCellCluster* cluster, _space->getClusters() ) {
        if( cluster->getMass() > 1.0 )
            rotEnergy += Physics::kineticEnergy(0.0,
                                                QVector3D(0.0, 0.0, 0.0),
                                                cluster->getAngularMass(),
                                                cluster->getAngularVel());
    }
    return rotEnergy;
}

qreal AlienThread::calcInternalEnergy ()
{
    qreal internalEnergy(0.0);
    foreach( AlienCellCluster* cluster, _space->getClusters() ) {
        if( !cluster->isEmpty() ) {
            foreach( AlienCell* cell, cluster->getCells() ) {
                internalEnergy += cell->getEnergyIncludingTokens();
            }
        }
    }
    foreach( AlienEnergy* energyParticle, _space->getEnergyParticles() ) {
        internalEnergy += energyParticle->amount;
    }
    return internalEnergy;
}

void AlienThread::setRandomSeed (uint seed)
{
    qsrand(seed);
    qrand();
}

void AlienThread::calcNextTimestep ()
{
    _space->lockData();

    //cell movement: step 1
    foreach( AlienCellCluster* cluster, _space->getClusters()) {
        cluster->movementProcessingStep1();
    }

    //cell movement: step 2
    //----
//    qreal eOld(calcInternalEnergy()+(calcTransEnergy()+calcRotEnergy())/INTERNAL_TO_KINETIC_ENERGY);
    //----
    QMutableListIterator<AlienCellCluster*> i(_space->getClusters());
    QList< AlienEnergy* > energyParticles;
    while (i.hasNext()) {

        QList< AlienCellCluster* > fragments;
        AlienCellCluster* cluster(i.next());
        energyParticles.clear();
        cluster->movementProcessingStep2(fragments, energyParticles);
        _space->getEnergyParticles() << energyParticles;

        debugCluster(cluster, 2);
        //new cell cluster fragments?
//        bool delCluster = false;
        if( (!fragments.empty()) || (cluster->isEmpty()) ) {
//            delCluster = true;
            delete cluster;
            i.remove();
            foreach( AlienCellCluster* cluster2, fragments ) {
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
    foreach( AlienCellCluster* cluster, _space->getClusters()) {
        cluster->movementProcessingStep3();
        debugCluster(cluster, 3);
    }

    //cell movement: step 4
    QMutableListIterator<AlienCellCluster*> j(_space->getClusters());
    while (j.hasNext()) {
        AlienCellCluster* cluster(j.next());
        if( cluster->isEmpty()) {
            delete cluster;
            j.remove();
        }
        else {
            energyParticles.clear();
            bool decompose = false;
            cluster->movementProcessingStep4(energyParticles, decompose);
            _space->getEnergyParticles() << energyParticles;
            debugCluster(cluster, 4);

            //decompose cluster?
            if( decompose ) {
                j.remove();
                QList< AlienCellCluster* > newClusters = cluster->decompose();
                foreach( AlienCellCluster* newCluster, newClusters) {
                    debugCluster(newCluster, 4);
                    j.insert(newCluster);
                }
            }
        }
    }

    //cell movement: step 5
    foreach( AlienCellCluster* cluster, _space->getClusters()) {
        cluster->movementProcessingStep5();
        debugCluster(cluster, 5);
    }

    //energy particle movement
    QMutableListIterator<AlienEnergy*> p(_space->getEnergyParticles());
    while (p.hasNext()) {
        AlienEnergy* e(p.next());
        AlienCellCluster* cluster(0);
        if( e->movement(cluster) ) {

            //transform into cell?
            if( cluster ) {
                _space->getClusters() << cluster;
            }
            delete e;
            p.remove();
        }
    }

    _space->unlockData();

    emit nextTimestepCalculated();
}

void AlienThread::debugCluster (AlienCellCluster* c, int s)
{
    /*foreach(AlienCell* cell, c->getCells()) {
        for(int i = 0; i < cell->getNumConnections(); ++i) {
            QVector3D d = cell->getRelPos()-cell->getConnection(i)->getRelPos();
            if( d.length() > (CRIT_CELL_DIST_MAX+0.1) )
                qDebug("%d: %f", s, d.length());
        }
    }*/
}




//    msleep(100);
    //----------------
/*    foreach( AlienCellCluster* cluster, _clusters) {
        QList< AlienCell* > component;
        quint64 tag(GlobalFunctions::getTag());
        cluster->getConnectedComponent(cluster->_cells[0], tag, component);
        if( component.size() != cluster->_cells.size() ) {
            qDebug("ALARM4");
        }
    }*/
    //----------------
