#include <gtest/gtest.h>

#include <QEventLoop>

#include "Base/ServiceLocator.h"
#include "Base/GlobalFactory.h"
#include "Base/NumberGenerator.h"
#include "ModelInterface/Settings.h"
#include "ModelInterface/ModelBasicBuilderFacade.h"
#include "ModelInterface/SimulationContext.h"
#include "ModelInterface/SimulationController.h"
#include "ModelInterface/DescriptionHelper.h"
#include "ModelInterface/SimulationParameters.h"
#include "ModelInterface/SpaceProperties.h"
#include "ModelInterface/SimulationAccess.h"
#include "ModelInterface/Serializer.h"

#include "Tests/Predicates.h"

#include "IntegrationTestHelper.h"
#include "IntegrationTestFramework.h"

class SerializationTest
	: public IntegrationTestFramework
{
public:
	SerializationTest();
	~SerializationTest();

protected:
	string const& serializeSimulation() const;
	void deserializeSimulation(string const& data);

	SimulationController* _controller = nullptr;
	SimulationContext* _context = nullptr;
	SpaceProperties* _metric = nullptr;
	SimulationAccess* _access = nullptr;
	IntVector2D _gridSize{ 6, 6 };
	Serializer* _serializer = nullptr;
};

SerializationTest::SerializationTest()
	: IntegrationTestFramework({ 600, 300 })
{
	GlobalFactory* factory = ServiceLocator::getInstance().getService<GlobalFactory>();
	_controller = _facade->buildSimulationController(1, _gridSize, _universeSize, _symbols, _parameters);
	_context = _controller->getContext();
	_serializer = _facade->buildSerializer();
	_metric = _context->getSpaceProperties();
	_access = _facade->buildSimulationAccess();
	_serializer->init();
	_access->init(_context);
}

SerializationTest::~SerializationTest()
{
	delete _access;
	delete _controller;
}

string const & SerializationTest::serializeSimulation() const
{
	bool serializationFinished = false;
	QEventLoop pause;
	_controller->connect(_serializer, &Serializer::serializationFinished, [&]() {
		serializationFinished = true;
		pause.quit();
	});
	_serializer->serialize(_controller);
	if (!serializationFinished) {
		pause.exec();
	}
	return _serializer->retrieveSerializedSimulation();
}

void SerializationTest::deserializeSimulation(string const & data)
{
	_controller = _serializer->deserializeSimulation(data);
}

TEST_F(SerializationTest, testCheckIds)
{
	DataDescription dataBefore;
	for (int i = 1; i <= 10000; ++i) {
		QVector2D pos(_numberGen->getRandomReal(0, 599), _numberGen->getRandomReal(0, 299));
		dataBefore.addParticle(createParticleDescription());
	}
	_access->updateData(dataBefore);

	runSimulation(50, _controller);
	auto const& data = serializeSimulation();
	deserializeSimulation(data);
	runSimulation(50, _controller);

	//check result
	DataDescription extract = IntegrationTestHelper::getContent(_access, { { 0, 0 }, _universeSize });
	unordered_set<uint64_t> ids;
	if (extract.clusters) {
		for (auto const& cluster : *extract.clusters) {
			ASSERT_TRUE(ids.find(cluster.id) == ids.end());
			ids.insert(cluster.id);
			if (cluster.cells) {
				for (auto const& cell : *cluster.cells) {
					ASSERT_TRUE(ids.find(cell.id) == ids.end());
					ids.insert(cell.id);
				}
			}
		}
	}
	if (extract.particles) {
		for (auto const& particle : *extract.particles) {
			ASSERT_TRUE(ids.find(particle.id) == ids.end());
			ids.insert(particle.id);
		}
	}
}
