#include "simulationunit.h"

#include "entities/cell.h"
#include "entities/cellcluster.h"
#include "entities/energyparticle.h"
#include "entities/grid.h"
#include "entities/token.h"
#include "physics/physics.h"

#include "global/global.h"

#include <QFile>

SimulationUnit::SimulationUnit (QObject* parent)
    : QObject(parent), _grid(0)
{

}

SimulationUnit::~SimulationUnit ()
{
}

void SimulationUnit::init (Grid* grid)
{
    _grid = grid;
}

QList< CellCluster* >& SimulationUnit::getClusters ()
{
    return _grid->getClusters();
}

QList< EnergyParticle* >& SimulationUnit::getEnergyParticles ()
{
    return _grid->getEnergyParticles();
}

qreal SimulationUnit::calcTransEnergy ()
{
    qreal transEnergy(0.0);
    foreach( CellCluster* cluster, _grid->getClusters() ) {
        if( !cluster->isEmpty() )
            transEnergy += Physics::kineticEnergy(cluster->getCells().size(),
                                                  cluster->getVel(),
                                                  0.0,
                                                  0.0);
    }
    return transEnergy;
}

qreal SimulationUnit::calcRotEnergy ()
{
    qreal rotEnergy(0.0);
    foreach( CellCluster* cluster, _grid->getClusters() ) {
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
    foreach( CellCluster* cluster, _grid->getClusters() ) {
        if( !cluster->isEmpty() ) {
            foreach( Cell* cell, cluster->getCells() ) {
                internalEnergy += cell->getEnergyIncludingTokens();
            }
        }
    }
    foreach( EnergyParticle* energyParticle, _grid->getEnergyParticles() ) {
        internalEnergy += energyParticle->amount;
    }
    return internalEnergy;
}

void SimulationUnit::setRandomSeed (uint seed)
{
    qsrand(seed);
    qrand();
}

void SimulationUnit::calcNextTimestep ()
{
    _grid->lockData();

    //<TEMP>
    static int a = 0;
    if( ++a == 999 ) {
        qDebug() << _grid->getClusters().first()->getPosition();//->getCells().first()->calcPosition();
    }

    if( a < 300 ) {

        QString fileName = QString("../source/testdata/test%1.dat").arg(a);
        QFile file(fileName);
        if( file.open(QIODevice::ReadOnly) ) {
            QDataStream in(&file);
            quint32 numCluster;
            in >> numCluster;
            QList<QList<QVector3D>> clusterCellPosList;
            QList<QList<QVector3D>> clusterCellVelList;
            QList<QVector3D> clusterPosList;
            QList<qreal> clusterAngleList;
            QList<QVector3D> clusterVelList;
            QList<qreal> clusterAnglularVelList;
            QList<qreal> clusterAnglularMassList;
            for(int i = 0; i < numCluster; ++i) {
                QList<QVector3D> cellPosList;
                QList<QVector3D> cellVelList;
                QVector3D pos;
                qreal angle;
                QVector3D vel;
                qreal angularVel;
                qreal angularMass;
                quint32 numCell;
                in >> pos;
                in >> angle;
                in >> vel;
                in >> angularVel;
                in >> angularMass;
                clusterPosList << pos;
                clusterAngleList << angle;
                clusterVelList << vel;
                clusterAnglularVelList << angularVel;
                clusterAnglularMassList << angularMass;
                in >> numCell;
                for(int i = 0; i < numCell; ++i) {
                    QVector3D pos;
                    QVector3D vel;
                    in >> pos;
                    in >> vel;
                    cellPosList << pos;
                    cellVelList << vel;
                }
                clusterCellPosList << cellPosList;
                clusterCellVelList << cellVelList;
            }
            file.close();

            //checking
            int minCluster = qMin(_grid->getClusters().size(), (int)numCluster);
            bool breaking = false;
            qDebug() <<  "Timestep: " << a;
            for(int i = 0; i < minCluster; ++i) {
                QList<QVector3D> cellPosList = clusterCellPosList.at(i);
                QList<QVector3D> cellVelList = clusterCellVelList.at(i);
                CellCluster* cluster = _grid->getClusters().at(i);
                int minCell = qMin(cluster->getCells().size(), cellPosList.size());

                if(clusterPosList.at(i) != cluster->getPosition())
                    qDebug() << "DEVIATION of cluster " << i << "/" << minCluster << " in pos; OLD: " << clusterPosList.at(i) << ", NEW: " << cluster->getPosition();
                if(clusterAngleList.at(i) != cluster->getAngle())
                    qDebug() << "DEVIATION of cluster " << i << "/" << minCluster << " in angle; OLD: " << clusterAngleList.at(i) << ", NEW: " << cluster->getAngle();
                if(clusterVelList.at(i) != cluster->getVel()) {
                    qDebug() << "DEVIATION of cluster " << i << "/" << minCluster << " in vel; OLD: ("
                             << QString::number(clusterVelList.at(i).x(), 'g', 20) << ", "
                             << QString::number(clusterVelList.at(i).y(), 'g', 20)
                             << "), NEW: ("
                             << QString::number(cluster->getVel().x(), 'g', 20) << ", "
                             << QString::number(cluster->getVel().y(), 'g', 20) << ")";
                }
                if(clusterAnglularVelList.at(i) != cluster->getAngularVel())
                    qDebug() << "DEVIATION of cluster " << i << "/" << minCluster << " in angularVel; OLD: " << QString::number(clusterAnglularVelList.at(i), 'g', 20) << ", NEW: " << QString::number(cluster->getAngularVel(), 'g', 20);
                if(clusterAnglularMassList.at(i) != cluster->getAngularMass())
                    qDebug() << "DEVIATION of cluster " << i << "/" << minCluster << " in angularMass; OLD: " << clusterAnglularMassList.at(i) << ", NEW: " << cluster->getAngularMass();
                for(int j = 0; j < minCell; ++j) {
                    if( cellPosList.at(j) != cluster->getCells().at(j)->getRelPos()) {
                        qDebug() << "DEVIATION of cluster " << i << "/" << minCluster << " and cell " << j << "/" << minCell
                                 << " in pos; OLD: " << cellPosList.at(j) << ", NEW: " << cluster->getCells().at(j)->getRelPos();
                    }
                    if( cellVelList.at(j) != cluster->getCells().at(j)->getVel()) {
                        qDebug() << "DEVIATION of cluster " << i << "/" << minCluster << " and cell " << j << "/" << minCell
                                 << " in vel; OLD: ("
                                 << QString::number(cellVelList.at(j).x(), 'g', 20) << ", "
                                 << QString::number(cellVelList.at(j).y(), 'g', 20)
                                 << "), NEW: ("
                                 << QString::number(cluster->getCells().at(j)->getVel().x(), 'g', 20) << ", "
                                 << QString::number(cluster->getCells().at(j)->getVel().y(), 'g', 20) << ")";
//                        breaking = true;
//                        break;
                    }
                }
                if (breaking)
                    break;
                }
        }

/*        QString fileName = QString("../source/testdata/test%1.dat").arg(a);
        if( !fileName.isEmpty() ) {
            QFile file(fileName);
            if( file.open(QIODevice::WriteOnly) ) {
                QDataStream out(&file);
                quint32 numCluster = _grid->getClusters().size();
                out << numCluster;
                foreach (CellCluster* cluster, _grid->getClusters()) {
                    quint32 numCells = cluster->getCells().size();
                    out << cluster->getPosition();
                    out << cluster->getAngle();
                    out << cluster->getVel();
                    out << cluster->getAngularVel();
                    out << cluster->getAngularMass();
                    out << numCells;
                    foreach (Cell* cell, cluster->getCells()) {
                        out << cell->getRelPos();
                        out << cell->getVel();
                    }
                }
                file.close();
            }
        }*/
    }
    //</TEMP>

    //cell movement: step 1
    foreach( CellCluster* cluster, _grid->getClusters()) {
        cluster->movementProcessingStep1();
    }

    //cell movement: step 2
    //----
//    qreal eOld(calcInternalEnergy()+(calcTransEnergy()+calcRotEnergy())/INTERNAL_TO_KINETIC_ENERGY);
    //----
/*    QMutableListIterator<CellCluster*> i(_grid->getClusters());
    QList< EnergyParticle* > energyParticles;
    while (i.hasNext()) {

        QList< CellCluster* > fragments;
        CellCluster* cluster(i.next());
        energyParticles.clear();
        cluster->movementProcessingStep2(fragments, energyParticles);
        _grid->getEnergyParticles() << energyParticles;

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
*/
    //cell movement: step 3
    foreach( CellCluster* cluster, _grid->getClusters()) {
        cluster->movementProcessingStep3();
        debugCluster(cluster, 3);
    }


    /*int p = 0;  //TEMP
    foreach( CellCluster* cluster, _grid->getClusters()) {
        if( p++ == 3857+2021) {
            qDebug() << "Vel of 3857+2021: ";
        }
    }
    */


    //cell movement: step 4
    QMutableListIterator<CellCluster*> j(_grid->getClusters());
    int z = 0;  //TEMP
    int c = 0;  //TEMP
    while (j.hasNext()) {
        CellCluster* cluster(j.next());
        if( cluster->isEmpty()) {
            delete cluster;
            j.remove();
            if (z <= 3857)   //TEMP
                ++c;
        }
        else
            ++z; //TEMP
/*        else {
            energyParticles.clear();
            bool decompose = false;
            cluster->movementProcessingStep4(energyParticles, decompose);
            _grid->getEnergyParticles() << energyParticles;
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
        }*/
    }
    /*if (a == 1) //TEMP
        qDebug() << "Deleted objects: " << c;*/

    //cell movement: step 5
/*    foreach( CellCluster* cluster, _grid->getClusters()) {
        cluster->movementProcessingStep5();
        debugCluster(cluster, 5);
    }
*/
    //energy particle movement
    /*QMutableListIterator<EnergyParticle*> p(_grid->getEnergyParticles());
    while (p.hasNext()) {
        EnergyParticle* e(p.next());
        CellCluster* cluster(0);
        if( e->movement(cluster) ) {

            //transform into cell?
            if( cluster ) {
                _grid->getClusters() << cluster;
            }
            delete e;
            p.remove();
        }
    }*/

    _grid->unlockData();

    emit nextTimestepCalculated();
}

void SimulationUnit::debugCluster (CellCluster* c, int s)
{
    /*foreach(Cell* cell, c->getCells()) {
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
