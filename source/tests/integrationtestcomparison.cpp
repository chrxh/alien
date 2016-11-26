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

    bool loadSimulationAndReturnSuccess(SimulationController* simulationController)
    {
        QFile file(INTEGRATIONTEST_COMPARISON_INIT);
        bool fileOpened = file.open(QIODevice::ReadOnly);
        if (fileOpened) {
            QDataStream in(&file);
            simulationController->buildUniverse(in);
            file.close();
        }
        return fileOpened;
    }

    struct LoadedReferenceData {
        bool success = false;
        QList<QList<QVector3D>> clusterCellPosList;
        QList<QList<QVector3D>> clusterCellVelList;
        QList<QVector3D> clusterPosList;
        QList<qreal> clusterAngleList;
        QList<QVector3D> clusterVelList;
        QList<qreal> clusterAnglularVelList;
        QList<qreal> clusterAnglularMassList;
    };

    LoadedReferenceData loadReferenceData()
    {
        LoadedReferenceData ref;
        QFile file(INTEGRATIONTEST_COMPARISON_REF);
        if (!file.open(QIODevice::ReadOnly))
            return ref;

        QDataStream in(&file);
        quint32 numCluster;
        in >> numCluster;
        for(int i = 0; i < numCluster; ++i) {
            QList<QVector3D> cellPosList;
            QList<QVector3D> cellVelList;
            QVector3D pos;
            qreal angle;
            QVector3D vel;
            qreal angularVel;
            qreal angularMass;
            quint32 numCell;
            in >> pos >> angle >> vel >> angularVel >> angularMass;
            ref.clusterPosList << pos;
            ref.clusterAngleList << angle;
            ref.clusterVelList << vel;
            ref.clusterAnglularVelList << angularVel;
            ref.clusterAnglularMassList << angularMass;
            in >> numCell;
            for(int i = 0; i < numCell; ++i) {
                QVector3D pos;
                QVector3D vel;
                in >> pos >> vel;
                cellPosList << pos;
                cellVelList << vel;
            }
            ref.clusterCellPosList << cellPosList;
            ref.clusterCellVelList << cellVelList;
        }
        file.close();
        ref.success = true;
        return ref;
    }

    bool updateReferenceDataAndReturnSuccess(SimulationController* simulationController)
    {
        if (!INTEGRATIONTEST_COMPARISON_UPDATE_REF)
            return false;
        QFile file(INTEGRATIONTEST_COMPARISON_REF);
        Grid* grid = simulationController->getGrid();
        bool fileOpened = file.open(QIODevice::WriteOnly);
        if (fileOpened) {
            QDataStream out(&file);
            quint32 numCluster = grid->getClusters().size();
            out << numCluster;
            foreach (CellCluster* cluster, grid->getClusters()) {
                quint32 numCells = cluster->getCellsRef().size();
                out << cluster->getPosition();
                out << cluster->getAngle();
                out << cluster->getVel();
                out << cluster->getAngularVel();
                out << cluster->getAngularMass();
                out << numCells;
                foreach (Cell* cell, cluster->getCellsRef()) {
                    out << cell->getRelPos();
                    out << cell->getVel();
                }
            }
            file.close();
        }
        return fileOpened;
    }

    void runSimulation(SimulationController* simulationController)
    {
        for (int time = 0; time < INTEGRATIONTEST_COMPARISON_TIMESTEPS; ++time) {
            simulationController->requestNextTimestep();
        }
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

    void compareReferenceWithSimulation(SimulationController* simulationController, LoadedReferenceData const& ref)
    {
        Grid* grid = simulationController->getGrid();
        int refNumCluster = ref.clusterPosList.size();
        QVERIFY2(grid->getClusters().size() == static_cast<int>(refNumCluster), "Deviation in number of clusters.");
        int minNumCluster = qMin(grid->getClusters().size(), static_cast<int>(refNumCluster));
        for(int i = 0; i < minNumCluster; ++i) {
            CellCluster* cluster = grid->getClusters().at(i);
            QVERIFY2(ref.clusterPosList.at(i) == cluster->getPosition(), createVectorDeviationMessageForCluster(INTEGRATIONTEST_COMPARISON_TIMESTEPS, cluster->getId(), "in pos", ref.clusterPosList.at(i), cluster->getPosition()));
            QVERIFY2(ref.clusterVelList.at(i) == cluster->getVel(), createVectorDeviationMessageForCluster(INTEGRATIONTEST_COMPARISON_TIMESTEPS, cluster->getId(), "in vel", ref.clusterVelList.at(i), cluster->getVel()));
            QVERIFY2(ref.clusterAngleList.at(i) == cluster->getAngle(), createValueDeviationMessageForCluster(INTEGRATIONTEST_COMPARISON_TIMESTEPS, cluster->getId(), "in angle", ref.clusterAngleList.at(i), cluster->getAngle()));
            QVERIFY2(ref.clusterAnglularVelList.at(i) == cluster->getAngularVel(), createValueDeviationMessageForCluster(INTEGRATIONTEST_COMPARISON_TIMESTEPS, cluster->getId(), "in angular vel", ref.clusterAnglularVelList.at(i), cluster->getAngularVel()));
            QVERIFY2(ref.clusterAnglularMassList.at(i) == cluster->getAngularMass(), createValueDeviationMessageForCluster(INTEGRATIONTEST_COMPARISON_TIMESTEPS, cluster->getId(), "in angular mass", ref.clusterAnglularMassList.at(i), cluster->getAngularMass()));
            QList<QVector3D> cellPosList = ref.clusterCellPosList.at(i);
            QList<QVector3D> cellVelList = ref.clusterCellVelList.at(i);
            int minNumCell = qMin(cluster->getCellsRef().size(), cellPosList.size());
            for(int j = 0; j < minNumCell; ++j) {
                QVERIFY2(cellPosList.at(j) == cluster->getCellsRef().at(j)->getRelPos(), createVectorDeviationMessageForCell(INTEGRATIONTEST_COMPARISON_TIMESTEPS, cluster->getId()
                    , cluster->getCellsRef().at(j)->getId(), "in rel pos", cellPosList.at(j), cluster->getCellsRef().at(j)->getRelPos()));
                QVERIFY2(cellVelList.at(j) == cluster->getCellsRef().at(j)->getVel(), createVectorDeviationMessageForCell(INTEGRATIONTEST_COMPARISON_TIMESTEPS, cluster->getId()
                    , cluster->getCellsRef().at(j)->getId(), "in vel", cellVelList.at(j), cluster->getCellsRef().at(j)->getVel()));
            }
        }
    }
}

void IntegrationTestComparison::initTestCase()
{
    _simulationController = new SimulationController(SimulationController::Threading::NO_EXTRA_THREAD, this);
    if (!loadSimulationAndReturnSuccess(_simulationController)) {
        QString msg = QString("Could not open file ") + INTEGRATIONTEST_COMPARISON_INIT + QString(" in loadDataAndReturnSuccess(...).");
        QFAIL(msg.toLatin1().data());
    }
}

void IntegrationTestComparison::testRunAndCompareSimulation ()
{
    runSimulation(_simulationController);
    LoadedReferenceData ref = loadReferenceData();
    bool refUpdated = updateReferenceDataAndReturnSuccess(_simulationController);
    if (ref.success)
        compareReferenceWithSimulation(_simulationController, ref);
    else if (refUpdated)
        QFAIL("Reference file does not exist. It has been created for the next cycle.");
    else
        QFAIL("Reference file does not exist. It has not been created.");
}


void IntegrationTestComparison::cleanupTestCase()
{

}
