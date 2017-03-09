#include <gtest/gtest.h>
#include "tests/predicates.h"

#include "model/simulationcontext.h"
#include "model/config.h"
#include "model/entities/cell.h"
#include "model/entities/token.h"
#include "model/alienfacade.h"
#include "model/entities/cellcluster.h"
#include "global/servicelocator.h"

class UnitTestCellCluster : public ::testing::Test
{
public:
	UnitTestCellCluster();
	~UnitTestCellCluster();

protected:
	SimulationContext* _context;
	CellCluster* _cluster;
	Cell* _cell1;
	Cell* _cell2;
	Cell* _cell3;
	Cell* _cell4;
};


UnitTestCellCluster::UnitTestCellCluster()
{
	FactoryFacade* facade = ServiceLocator::getInstance().getService<FactoryFacade>();

	_context = facade->buildSimulationContext();
	_context->init({ 1000, 1000 });

	QList< Cell* > cells;
	for (int i = 0; i <= 100; ++i) {
		Cell* cell = facade->buildFeaturedCell(100.0, CellFunctionType::COMPUTER, _context);
		cell->setRelPos(QVector3D(i, 0.0, 0.0));
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
	_cluster->setVel(QVector3D(1.0, -0.5, 0.0));
	_cluster->updateCellVel(false);
	_cluster->updateVel_angularVel_via_cellVelocities();
	ASSERT_PRED2(predEqualMediumPrecision, _cluster->getAngularVel(), 2.0);
	ASSERT_PRED2(predEqualMediumPrecision, _cluster->getVel().x(), 1.0);
	ASSERT_PRED2(predEqualMediumPrecision, _cluster->getVel().y(), -0.5);
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
	_cell1->setTokenAccessNumber(0);
	_cell2->setTokenAccessNumber(1);
	_cell3->setTokenAccessNumber(1);
	_cell4->setTokenAccessNumber(0);
	Token* token = new Token(simulationParameters.MIN_TOKEN_ENERGY * 3);
	_cell1->addToken(token, Cell::ACTIVATE_TOKEN::NOW, Cell::UPDATE_TOKEN_ACCESS_NUMBER::YES);
	QList< EnergyParticle* > tempEP;
	bool tempDecomp = false;
	_cluster->processingToken(tempEP, tempDecomp);
	ASSERT_EQ(_cell1->getNumToken(true), 0);
	ASSERT_EQ(_cell2->getNumToken(true), 1);
	ASSERT_EQ(_cell3->getNumToken(true), 1);
	ASSERT_EQ(_cell4->getNumToken(true), 0);
}


