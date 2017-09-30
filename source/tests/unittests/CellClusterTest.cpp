#include <gtest/gtest.h>
#include "tests/predicates.h"

#include "Base/ServiceLocator.h"
#include "Model/Context/UnitContext.h"
#include "Model/SimulationParameters.h"
#include "Model/Settings.h"
#include "Model/ModelBuilderFacade.h"
#include "Model/Context/SpaceMetricLocal.h"
#include "Model/Entities/Cell.h"
#include "Model/Entities/Token.h"
#include "Model/Entities/Cluster.h"
#include "Model/Entities/EntityFactory.h"

class CellClusterTest : public ::testing::Test
{
public:
	CellClusterTest();
	~CellClusterTest();

protected:
	UnitContext* _context = nullptr;
	Cluster* _cluster = nullptr;
	Cell* _cell1 = nullptr;
	Cell* _cell2 = nullptr;
	Cell* _cell3 = nullptr;
	Cell* _cell4 = nullptr;
};


CellClusterTest::CellClusterTest()
{
	ModelBuilderFacade* facade = ServiceLocator::getInstance().getService<ModelBuilderFacade>();

/*
	_context = facade->buildSimulationContext();
	auto metric = facade->buildSpaceMetric();
	metric->init({ 1000, 1000 });
	_context->init(metric);

	QList< Cell* > cells;
	for (int i = 0; i <= 100; ++i) {
		Cell* cell = facade->buildFeaturedCell(100.0, Enums::CellFunction::COMPUTER, _context);
		cell->setRelPosition(QVector2D(i, 0.0, 0.0));
		cells << cell;
	}
	QVector2D pos(200.0, 100.0, 0.0);
	_cluster = facade->buildCellCluster(cells, 0.0, pos, 0.0, QVector2D(), _context);
	_cell1 = _cluster->getCellsRef().at(0);
	_cell2 = _cluster->getCellsRef().at(1);
	_cell3 = _cluster->getCellsRef().at(2);
	_cell4 = _cluster->getCellsRef().at(3);
*/
}

CellClusterTest::~CellClusterTest()
{
	delete _context;
}

//calc cell velocities and then the cluster velocity
//and comparison with previous values (there should be no change)
/*
TEST_F (UnitTestCellCluster, testCellVelocityDecomposition)
{
	_cluster->setAngularVel(2.0);
	_cluster->setVelocity(QVector2D(1.0, -0.5, 0.0));
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
*/


