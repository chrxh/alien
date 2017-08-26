#include <gtest/gtest.h>
#include <QFile>

#include "Model/SimulationController.h"
#include "Model/Context/UnitContext.h"
#include "Model/Settings.h"
#include "Model/Entities/Cluster.h"
#include "Model/Entities/Cell.h"

#include "tests/TestSettings.h"

class ComparisonTest : public ::testing::Test
{
public:
	ComparisonTest();
	~ComparisonTest();

protected:
	SimulationController* _simulationController = nullptr;
};

/*namespace {

    bool loadSimulationAndReturnSuccess(SimulationController* simulationController)
    {
        QFile file(INTEGRATIONTEST_COMPARISON_INIT);
        bool fileOpened = file.open(QIODevice::ReadOnly);
        if (fileOpened) {
            QDataStream in(&file);
            simulationController->loadUniverse(in);
            file.close();
        }
        return fileOpened;
    }

    struct LoadedReferenceData {
        bool success = false;
        QList<QList<QVector2D>> clusterCellPosList;
        QList<QList<QVector2D>> clusterCellVelList;
        QList<QVector2D> clusterPosList;
        QList<qreal> clusterAngleList;
        QList<QVector2D> clusterVelList;
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
            QList<QVector2D> cellPosList;
            QList<QVector2D> cellVelList;
            QVector2D pos;
            qreal angle;
            QVector2D vel;
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
                QVector2D pos;
                QVector2D vel;
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
        SimulationContext* context = simulationController->getSimulationContext();
        bool fileOpened = file.open(QIODevice::WriteOnly);
        if (fileOpened) {
            QDataStream out(&file);
            quint32 numCluster = context->getClustersRef().size();
            out << numCluster;
            foreach (CellCluster* cluster, context->getClustersRef()) {
                quint32 numCells = cluster->getCellsRef().size();
                out << cluster->getPosition();
                out << cluster->getAngle();
                out << cluster->getVelocity();
                out << cluster->getAngularVel();
                out << cluster->getAngularMass();
                out << numCells;
                foreach (Cell* cell, cluster->getCellsRef()) {
                    out << cell->getRelPosition();
                    out << cell->getVelocity();
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

    char const* createVectorDeviationMessageForCluster (int time, int clusterId, QString what, QVector2D ref, QVector2D comp)
    {
        QString msg = QString("Deviation at time ") + QString::number(time);
        msg += QString(" in cluster ") + QString::number(clusterId) + QString(" ") + what;
        msg += QString(": reference value: ");
        msg += QString("(") + QString::number(ref.x(), 'g', 12) + QString(", ") + QString::number(ref.y(), 'g', 12) + QString(")");
        msg += QString(" and computation: ");
        msg += QString("(") + QString::number(comp.x(), 'g', 12) + QString(", ") + QString::number(comp.y(), 'g', 12) + QString(")");
        return msg.toLatin1().data();
    }

    char const* createVectorDeviationMessageForCell (int time, int clusterId, int cellId, QString what, QVector2D ref, QVector2D comp)
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

    void compareReferenceWithSimulation (SimulationController* simulationController, LoadedReferenceData const& ref)
    {
        UnitContext* context = simulationController->getSimulationContext();
        int refNumCluster = ref.clusterPosList.size();
		ASSERT_EQ(context->getClustersRef().size(), static_cast<int>(refNumCluster))
			<< "Deviation in number of clusters.";
        int minNumCluster = qMin(context->getClustersRef().size(), static_cast<int>(refNumCluster));
        for(int i = 0; i < minNumCluster; ++i) {
            CellCluster* cluster = context->getClustersRef().at(i);
			ASSERT_EQ(ref.clusterPosList.at(i), cluster->getPosition()) 
				<< createVectorDeviationMessageForCluster(INTEGRATIONTEST_COMPARISON_TIMESTEPS, cluster->getId(), "in pos", ref.clusterPosList.at(i), cluster->getPosition());
			ASSERT_EQ(ref.clusterVelList.at(i), cluster->getVelocity()) 
				<< createVectorDeviationMessageForCluster(INTEGRATIONTEST_COMPARISON_TIMESTEPS, cluster->getId(), "in vel", ref.clusterVelList.at(i), cluster->getVelocity());
			ASSERT_EQ(ref.clusterAngleList.at(i), cluster->getAngle()) 
				<< createValueDeviationMessageForCluster(INTEGRATIONTEST_COMPARISON_TIMESTEPS, cluster->getId(), "in angle", ref.clusterAngleList.at(i), cluster->getAngle());
			ASSERT_EQ(ref.clusterAnglularVelList.at(i), cluster->getAngularVel()) 
				<< createValueDeviationMessageForCluster(INTEGRATIONTEST_COMPARISON_TIMESTEPS, cluster->getId(), "in angular vel", ref.clusterAnglularVelList.at(i), cluster->getAngularVel());
			ASSERT_EQ(ref.clusterAnglularMassList.at(i), cluster->getAngularMass()) 
				<< createValueDeviationMessageForCluster(INTEGRATIONTEST_COMPARISON_TIMESTEPS, cluster->getId(), "in angular mass", ref.clusterAnglularMassList.at(i), cluster->getAngularMass());
            QList<QVector2D> cellPosList = ref.clusterCellPosList.at(i);
            QList<QVector2D> cellVelList = ref.clusterCellVelList.at(i);
            int minNumCell = qMin(cluster->getCellsRef().size(), cellPosList.size());
            for(int j = 0; j < minNumCell; ++j) {
				ASSERT_EQ(cellPosList.at(j), cluster->getCellsRef().at(j)->getRelPosition())
					<< createVectorDeviationMessageForCell(INTEGRATIONTEST_COMPARISON_TIMESTEPS, cluster->getId()
						, cluster->getCellsRef().at(j)->getId(), "in rel pos", cellPosList.at(j), cluster->getCellsRef().at(j)->getRelPosition());
				ASSERT_EQ(cellVelList.at(j), cluster->getCellsRef().at(j)->getVelocity())
					<< createVectorDeviationMessageForCell(INTEGRATIONTEST_COMPARISON_TIMESTEPS, cluster->getId()
						, cluster->getCellsRef().at(j)->getId(), "in vel", cellVelList.at(j), cluster->getCellsRef().at(j)->getVelocity());
            }
        }
    }
}
*/
ComparisonTest::ComparisonTest()
{
/*
	_simulationController = new SimulationController();
*/
}

ComparisonTest::~ComparisonTest()
{
	delete _simulationController;
}
/*
TEST_F (IntegrationTestComparison, testRunAndCompareSimulation)
{
	if (!loadSimulationAndReturnSuccess(_simulationController)) {
		QString msg = QString("Could not open file ") + INTEGRATIONTEST_COMPARISON_INIT + QString(" in loadDataAndReturnSuccess(...).");
		FAIL() << msg.toLatin1().data();
	}
	runSimulation(_simulationController);
    LoadedReferenceData ref = loadReferenceData();
    bool refUpdated = updateReferenceDataAndReturnSuccess(_simulationController);
    if (ref.success)
        compareReferenceWithSimulation(_simulationController, ref);
    else if (refUpdated)
        FAIL() << "Reference file does not exist. It has been created for the next cycle.";
    else
        FAIL() << "Reference file does not exist. It has not been created.";
}

*/

