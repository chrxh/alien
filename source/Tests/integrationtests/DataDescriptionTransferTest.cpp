#include <gtest/gtest.h>

#include <QEventLoop>

#include "Base/ServiceLocator.h"
#include "Base/GlobalFactory.h"
#include "Base/NumberGenerator.h"
#include "Model/ModelBuilderFacade.h"
#include "Model/Settings.h"
#include "Model/SimulationController.h"
#include "Model/Context/SimulationContext.h"
#include "Model/Context/SimulationParameters.h"
#include "Model/AccessPorts/SimulationAccess.h"

#include "tests/Predicates.h"

class DataDescriptionTransferTest : public ::testing::Test
{
public:
	DataDescriptionTransferTest();
	~DataDescriptionTransferTest();

protected:
	SimulationController* _controller = nullptr;
	SimulationContext* _context = nullptr;
	SimulationParameters* _parameters = nullptr;
	NumberGenerator* _numberGen = nullptr;
	IntVector2D _gridSize{ 6, 6 };
	IntVector2D _universeSize{ 600, 300 };
};

DataDescriptionTransferTest::DataDescriptionTransferTest()
{
	ModelBuilderFacade* facade = ServiceLocator::getInstance().getService<ModelBuilderFacade>();
	GlobalFactory* factory = ServiceLocator::getInstance().getService<GlobalFactory>();
	auto symbols = facade->buildDefaultSymbolTable();
	_parameters = facade->buildDefaultSimulationParameters();
	_controller = facade->buildSimulationController(1, _gridSize, _universeSize, symbols, _parameters);
	_context = static_cast<SimulationContext*>(_controller->getContext());
	_numberGen = factory->buildRandomNumberGenerator();
	_numberGen->init(123123, 0);
}

DataDescriptionTransferTest::~DataDescriptionTransferTest()
{
	delete _controller;
	delete _numberGen;
}

TEST_F(DataDescriptionTransferTest, testTransferRandomData)
{
	ModelBuilderFacade* facade = ServiceLocator::getInstance().getService<ModelBuilderFacade>();
	auto access = facade->buildSimulationAccess(_context);

	DataDescription dataBefore;
	QVector2D pos(_numberGen->getRandomReal(0, 599), _numberGen->getRandomReal(0, 299));
	for (int i = 0; i < 100; ++i) {
		dataBefore.addCluster(
			ClusterDescription()
			.setPos(pos)
			.setVel(QVector2D(_numberGen->getRandomReal(-1, 1), _numberGen->getRandomReal(-1, 1)))
			.addCell(
				CellDescription().setEnergy(_parameters->cellCreationEnergy).setPos(pos)
			)
		);
	}
	access->updateData(dataBefore);
	IntRect rect = { { 0, 0 }, { _universeSize.x - 1, _universeSize.y - 1 } };

	ResolveDescription resolveDesc;
	access->requireData(rect, resolveDesc);
	DataDescription dataAfter = access->retrieveData();

	ASSERT_TRUE(dataBefore.isCompatibleWith(dataAfter));
	delete access;
}

