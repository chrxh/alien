#include <gtest/gtest.h>
#include "tests/predicates.h"

#include "global/ServiceLocator.h"
#include "model/context/UnitContext.h"
#include "model/context/SimulationParameters.h"
#include "model/ModelSettings.h"
#include "model/BuilderFacade.h"
#include "model/context/SpaceMetric.h"
#include "model/entities/Cell.h"
#include "model/entities/Token.h"
#include "model/entities/CellCluster.h"
#include "model/entities/EntityFactory.h"

class UnitTestCellCluster : public ::testing::Test
{
public:
	UnitTestCellCluster();
	~UnitTestCellCluster();

protected:
	UnitContext* _context;
	CellCluster* _cluster;
	Cell* _cell1;
	Cell* _cell2;
	Cell* _cell3;
	Cell* _cell4;
};


UnitTestCellCluster::UnitTestCellCluster()
{
	BuilderFacade* facade = ServiceLocator::getInstance().getService<BuilderFacade>();

	_context = facade->buildSimulationContext();
	auto metric = facade->buildSpaceMetric();
	metric->init({ 1000, 1000 });
	_context->init(metric);

	QList< Cell* > cells;
	for (int i = 0; i <= 100; ++i) {
		Cell* cell = facade->buildFeaturedCell(100.0, Enums::CellFunction::COMPUTER, _context);
		cell->setRelPosition(QVector3D(i, 0.0, 0.0));
		cells << cell;
	}
	QVector3D pos(200.0, 100.0, 0.0);
	_cluster = facade->buildCellCluster(cells, 0.0, pos, 0.0, QVector3D(), _context);
	_cell1 = _cluster->getCellsRef().at(0);
	_cell2 = _cluster->getCellsRef().at(1);
	_cell3 = _cluster->getCellsRef().at(2);
	_cell4 = _cluster->getCellsRef().at(3);
}

UnitTestCellCluster::~UnitTestCellCluster()
{
	delete _context;
}

//calc cell velocities and then the cluster velocity
//and comparison with previous values (there should be no change)
TEST_F (UnitTestCellCluster, testCellVelocityDecomposition)
{
	_cluster->setAngularVel(2.0);
	_cluster->setVelocity(QVector3D(1.0, -0.5, 0.0));
	_cluster->updateCellVel(false);
	_cluster->updateVel_angularVel_via_cellVelocities();
	ASSERT_PRED2(predEqualMediumPrecision, _cluster->getAngularVel(), 2.0);
	ASSERT_PRED2(predEqualMediumPrecision, _cluster->getVelocity().x(), 1.0);
	ASSERT_PRED2(predEqualMediumPrecision, _cluster->getVelocity().y(), -0.5);
}

TEST_F (UnitTestCellCluster, testNewConnections)
{
	ASSERT_TRUE(_cluster->getCellsRef().size() > 3);
	_cell1->resetConnections(3);
	_cell2->resetConnections(1);
	_cell3->resetConnections(1);
	_cell4->resetConnections(1);

	_cell1->newConnection(_cell2);
	_cell1->newConnection(_cell3);
	_cell1->newConnection(_cell4);
	ASSERT_EQ(_cell1->getConnection(0),  _cell2);
	ASSERT_EQ(_cell1->getConnection(1),  _cell3);
	ASSERT_EQ(_cell1->getConnection(2),  _cell4);
	ASSERT_EQ(_cell2->getConnection(0),  _cell1);
	ASSERT_EQ(_cell3->getConnection(0),  _cell1);
	ASSERT_EQ(_cell4->getConnection(0), _cell1);
}

TEST_F (UnitTestCellCluster, testTokenSpreading)
{
	_cell1->resetConnections(3);
	_cell2->resetConnections(1);
	_cell3->resetConnections(1);
	_cell4->resetConnections(1);
	_cell1->newConnection(_cell2);
	_cell1->newConnection(_cell3);
	_cell1->newConnection(_cell4);
	_cell1->setBranchNumber(0);
	_cell2->setBranchNumber(1);
	_cell3->setBranchNumber(1);
	_cell4->setBranchNumber(0);
	EntityFactory* factory= ServiceLocator::getInstance().getService<EntityFactory>();
	Token* token = factory->buildToken(_context, _context->getSimulationParameters()->tokenMinEnergy * 3);
	_cell1->addToken(token, Cell::ActivateToken::NOW, Cell::UpdateTokenAccessNumber::YES);
	QList< EnergyParticle* > tempEP;
	bool tempDecomp = false;
	_cluster->processingToken(tempEP, tempDecomp);
	ASSERT_EQ(_cell1->getNumToken(true), 0);
	ASSERT_EQ(_cell2->getNumToken(true), 1);
	ASSERT_EQ(_cell3->getNumToken(true), 1);
	ASSERT_EQ(_cell4->getNumToken(true), 0);
}


