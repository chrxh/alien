#include "integrationtestdeterminism.h"
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
    void loadData(SimulationController* simulationController)
    {
        QFile file(TESTDATA_FILENAME);
        bool fileOpened = file.open(QIODevice::ReadOnly);
        QString msg = QString("Could not open file %1 in IntegrationTestDeterminism::loadData().").arg(TESTDATA_FILENAME);
        QVERIFY2(fileOpened, msg.toLatin1().data());
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
    }
}

void IntegrationTestDeterminism::initTestCase()
{
    _simController1 = new SimulationController(SimulationController::Threading::NO_EXTRA_THREAD, this);
    _simController2 = new SimulationController(SimulationController::Threading::NO_EXTRA_THREAD, this);
    GlobalFunctions::setTag(_tag1);
    loadData(_simController1);
    _tag1 = GlobalFunctions::getTag();
    GlobalFunctions::setTag(_tag2);
    loadData(_simController2);
    _tag2 = GlobalFunctions::getTag();
}

void IntegrationTestDeterminism::testRunSimulations ()
{
    for (int i = 0; i < TIMESTEPS; ++i) {
        QString msg = QString("Number of clusters do not coincide at timestep %1.").arg(i);
        QVERIFY2(compareClusterSizes(), msg.toLatin1().data());

        QList<int> abnormalClusterNumbers = getAbnormalClusterNumbers(i);
        if (!abnormalClusterNumbers.empty()) {
            QString msg = QString("The following clusters do not coincide at timestep %1: ").arg(i);
            for (int i = 0; i < 10 && i < abnormalClusterNumbers.size(); ++i) {
                msg += QString("%1 ").arg(abnormalClusterNumbers.at(i));
            }
            QFAIL(msg.toLatin1().data());
        }
        qsrand(i);
        GlobalFunctions::setTag(_tag1);
        _simController1->requestNextTimestep();
        _tag1 = GlobalFunctions::getTag();
        qsrand(i);
        GlobalFunctions::setTag(_tag2);
        _simController2->requestNextTimestep();
        _tag2 = GlobalFunctions::getTag();
    }
}

bool IntegrationTestDeterminism::compareClusterSizes ()
{
    Grid* grid1 = _simController1->getGrid();
    Grid* grid2 = _simController2->getGrid();
    return grid1->getClusters().size() == grid2->getClusters().size();
}

QList<int> IntegrationTestDeterminism::getAbnormalClusterNumbers (int timestep)
{
    Grid* grid1 = _simController1->getGrid();
    Grid* grid2 = _simController2->getGrid();
    int minClusters = qMin(grid1->getClusters().size(), grid2->getClusters().size());
    QList<int> abnormalClusterNumbers;
    for (int i = 0; i < minClusters; ++i) {
        CellCluster* cluster1 = grid1->getClusters().at(i);
        CellCluster* cluster2 = grid2->getClusters().at(i);
        if (!cluster1->compareEqual(cluster2)) {
            abnormalClusterNumbers << i;
        }
    }
    return abnormalClusterNumbers;
}

void IntegrationTestDeterminism::cleanupTestCase()
{

}
