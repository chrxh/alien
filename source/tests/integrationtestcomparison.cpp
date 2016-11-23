#include "integrationtestcomparison.h"
#include "testsettings.h"
#include "model/simulationcontroller.h"
#include "model/metadatamanager.h"
#include "model/simulationsettings.h"
#include "model/entities/grid.h"
#include "model/entities/cellcluster.h"
#include "model/entities/cell.h"
#include "global/global.h"

#include <QtTest/QtTest>

namespace {
    bool loadDataAndReturnSuccess(SimulationController* simulationController, QString fileName)
    {
        QFile file(fileName );
        bool fileOpened = file.open(QIODevice::ReadOnly);
        if( fileOpened ) {
            QDataStream in(&file);
            QMap< quint64, quint64 > oldNewCellIdMap;
            QMap< quint64, quint64 > oldNewClusterIdMap;
            simulationController->buildUniverse(in, oldNewClusterIdMap, oldNewCellIdMap);
            simulationParameters.readData(in);
            MetadataManager::getGlobalInstance().readMetadataUniverse(in, oldNewClusterIdMap, oldNewCellIdMap);
            MetadataManager::getGlobalInstance().readSymbolTable(in);
            file.close();
        }
        return fileOpened;
    }

    char const* createValueDeviationMessageForCluster (int time, int clusterId, QString what, qreal ref, qreal comp)
    {
        QString msg = QString("Deviation at time ") + QString::number(time);
        msg += QString(" in cluster ") + QString::number(clusterId) + QString(" ") + what;
        msg += QString(": reference value: ");
        msg += QString::number(ref, 'g', 12);
        msg += QString(" and computation: ");
        msg += QString::number(comp, 'g', 12);
        return msg.toLatin1().data();
    }

    char const* createVectorDeviationMessageForCluster (int time, int clusterId, QString what, QVector3D ref, QVector3D comp)
    {
        QString msg = QString("Deviation at time ") + QString::number(time);
        msg += QString(" in cluster ") + QString::number(clusterId) + QString(" ") + what;
        msg += QString(": reference value: ");
        msg += QString("(") + QString::number(ref.x(), 'g', 12) + QString(", ") + QString::number(ref.y(), 'g', 12) + QString(")");
        msg += QString(" and computation: ");
        msg += QString("(") + QString::number(comp.x(), 'g', 12) + QString(", ") + QString::number(comp.y(), 'g', 12) + QString(")");
        return msg.toLatin1().data();
    }

    char const* createVectorDeviationMessageForCell (int time, int clusterId, int cellId, QString what, QVector3D ref, QVector3D comp)
    {
        QString msg = QString("Deviation at time ") + QString::number(time);
        msg += QString(" in cluster ") + QString::number(clusterId);
        msg += QString(" at cell ") + QString::number(cellId) + QString(" ") + what;
        msg += QString(": reference value: ");
        msg += QString("(") + QString::number(ref.x(), 'g', 12) + QString(", ") + QString::number(ref.y(), 'g', 12) + QString(")");
        msg += QString(" and computation: ");
        msg += QString("(") + QString::number(comp.x(), 'g', 12) + QString(", ") + QString::number(comp.y(), 'g', 12) + QString(")");
        return msg.toLatin1().data();
    }
}

void IntegrationTestComparison::initTestCase()
{
    _simController = new SimulationController(SimulationController::Threading::NO_EXTRA_THREAD, this);
    QString fileName = TESTDATA_COMPARISON_REF_FOLDER + QString("/initial.sim");
    if (!loadDataAndReturnSuccess(_simController, fileName)) {
        QString msg = QString("Could not open file ") + fileName + QString(" in IntegrationTestDeterminism::loadData().");
        QFAIL(msg.toLatin1().data());
    }
}

void IntegrationTestComparison::testRunSimulations ()
{
    Grid* grid = _simController->getGrid();
    for (int time = 0; time < TIMESTEPS; ++time) {
        qsrand(time);
        _simController->requestNextTimestep();
    }

    {
        int time = TIMESTEPS;
        QString fileName = TESTDATA_COMPARISON_REF_FOLDER + QString("/computation.dat");
        QFile file(fileName);
        if (!file.open(QIODevice::ReadOnly)) {
            QWARN("Test data for IntegrationTestComparison do not exist. It will now be created for the next cycle.");
        }
        else {
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
            QVERIFY2(grid->getClusters().size() == static_cast<int>(numCluster), "Deviation in number of clusters.");
            int minNumCluster = qMin(grid->getClusters().size(), static_cast<int>(numCluster));
            for(int i = 0; i < minNumCluster; ++i) {
                QList<QVector3D> cellPosList = clusterCellPosList.at(i);
                QList<QVector3D> cellVelList = clusterCellVelList.at(i);
                CellCluster* cluster = grid->getClusters().at(i);
                int minNumCell = qMin(cluster->getCells().size(), cellPosList.size());
                QVERIFY2(clusterPosList.at(i) == cluster->getPosition(), createVectorDeviationMessageForCluster(time, cluster->getId(), "in pos", clusterPosList.at(i), cluster->getPosition()));
                QVERIFY2(clusterVelList.at(i) == cluster->getVel(), createVectorDeviationMessageForCluster(time, cluster->getId(), "in vel", clusterVelList.at(i), cluster->getVel()));
                QVERIFY2(clusterAngleList.at(i) == cluster->getAngle(), createValueDeviationMessageForCluster(time, cluster->getId(), "in angle", clusterAngleList.at(i), cluster->getAngle()));
                QVERIFY2(clusterAnglularVelList.at(i) == cluster->getAngularVel(), createValueDeviationMessageForCluster(time, cluster->getId(), "in angular vel", clusterAnglularVelList.at(i), cluster->getAngularVel()));
                QVERIFY2(clusterAnglularMassList.at(i) == cluster->getAngularMass(), createValueDeviationMessageForCluster(time, cluster->getId(), "in angular mass", clusterAnglularMassList.at(i), cluster->getAngularMass()));

                for(int j = 0; j < minNumCell; ++j) {
                    QVERIFY2(cellPosList.at(j) == cluster->getCells().at(j)->getRelPos(), createVectorDeviationMessageForCell(time, cluster->getId()
                        , cluster->getCells().at(j)->getId(), "in rel pos", cellPosList.at(j), cluster->getCells().at(j)->getRelPos()));
                    QVERIFY2(cellVelList.at(j) == cluster->getCells().at(j)->getVel(), createVectorDeviationMessageForCell(time, cluster->getId()
                        , cluster->getCells().at(j)->getId(), "in rel pos", cellVelList.at(j), cluster->getCells().at(j)->getVel()));
                }
            }
        }
    }
    {
        QString fileName = TESTDATA_COMPARISON_SIM_FOLDER + QString("/computation.dat");
        QFile file(fileName);
        if( file.open(QIODevice::WriteOnly) ) {
            QDataStream out(&file);
            quint32 numCluster = grid->getClusters().size();
            out << numCluster;
            foreach (CellCluster* cluster, grid->getClusters()) {
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
    }
}


void IntegrationTestComparison::cleanupTestCase()
{

}
