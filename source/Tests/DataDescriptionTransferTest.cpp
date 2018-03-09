#include <gtest/gtest.h>

#include <QEventLoop>

#include "Base/ServiceLocator.h"
#include "Base/GlobalFactory.h"
#include "Base/NumberGenerator.h"
#include "Model/Api/ModelBuilderFacade.h"
#include "Model/Api/Settings.h"
#include "Model/Api/SimulationController.h"
#include "Model/Api/DescriptionHelper.h"
#include "Model/Local/SimulationContextLocal.h"
#include "Model/Api/SimulationParameters.h"
#include "Model/Api/SpaceProperties.h"
#include "Model/Api/SimulationAccess.h"

#include "Tests/Predicates.h"

#include "IntegrationTestHelper.h"
#include "IntegrationTestFramework.h"

class DataDescriptionTransferTest
	: public IntegrationTestFramework
{
public:
	DataDescriptionTransferTest();
	~DataDescriptionTransferTest();

protected:
	SimulationController* _controller = nullptr;
	SimulationContextLocal* _context = nullptr;
	SpaceProperties* _metric = nullptr;
	SimulationAccess* _access = nullptr;
	IntVector2D _gridSize{ 6, 6 };
};

DataDescriptionTransferTest::DataDescriptionTransferTest()
	: IntegrationTestFramework({ 600, 300 })
{
	GlobalFactory* factory = ServiceLocator::getInstance().getService<GlobalFactory>();
	_controller = _facade->buildSimulationController(1, _gridSize, _universeSize, _symbols, _parameters);
	_context = static_cast<SimulationContextLocal*>(_controller->getContext());
	_metric = _context->getSpaceProperties();
	_access = _facade->buildSimulationAccess();
	_access->init(_context);
}

DataDescriptionTransferTest::~DataDescriptionTransferTest()
{
	delete _access;
	delete _controller;
}

TEST_F(DataDescriptionTransferTest, testCreateClusterDescriptionWithCompleteCell)
{
	DataDescription dataBefore;
	dataBefore.addCluster(createClusterDescriptionWithCompleteCell());
	_access->updateData(dataBefore);

	IntRect rect = { { 0, 0 }, { _universeSize.x - 1, _universeSize.y - 1 } };
	DataDescription dataAfter = IntegrationTestHelper::getContent(_access, rect);

	ASSERT_TRUE(isCompatible(dataBefore, dataAfter));
}

TEST_F(DataDescriptionTransferTest, testModifyClusterDescriptionWithCompleteCell)
{
	const uint64_t clusterId = 1;
	const uint64_t cellId = 2;
	DataDescription dataInit;
	auto cluster = ClusterDescription().addCell(
		CellDescription().setId(cellId).setCellFeature(
			CellFeatureDescription().setType(Enums::CellFunction::SCANNER)
		).setPos({ 2, 1 }).setEnergy(36).setFlagTokenBlocked(true).setMaxConnections(2)
		.setTokenBranchNumber(1).setTokens({
			TokenDescription().setEnergy(75),
			TokenDescription().setEnergy(8)
	})
	).setId(clusterId).setPos({ 2, 1 }).setVel({ -0.3f, -0.1f }).setAngle(121.3).setAngularVel(-0.2);
	dataInit.addCluster(cluster);
	_access->updateData(dataInit);

	DataDescription dataBefore;
	dataBefore.addCluster(createClusterDescriptionWithCompleteCell(clusterId, cellId));
	_access->updateData(DataChangeDescription(dataInit, dataBefore));

	IntRect rect = { { 0, 0 },{ _universeSize.x - 1, _universeSize.y - 1 } };
	DataDescription dataAfter = IntegrationTestHelper::getContent(_access, rect);

	ASSERT_TRUE(isCompatible(dataBefore, dataAfter));
}

TEST_F(DataDescriptionTransferTest, testCreateRandomData)
{
	DataDescription dataBefore;
	for (int i = 1; i <= 100; ++i) {
		dataBefore.addCluster(createClusterDescription(i));
	}
	for (int i = 1; i <= 100; ++i) {
		dataBefore.addParticle(createParticleDescription());
	}
	_access->updateData(dataBefore);

	IntRect rect = { { 0, 0 }, { _universeSize.x - 1, _universeSize.y - 1 } };
	DataDescription dataAfter = IntegrationTestHelper::getContent(_access, rect);

	ASSERT_TRUE(isCompatible(dataBefore, dataAfter));
}

TEST_F(DataDescriptionTransferTest, testCreateAndDeleteRandomData)
{
	DataDescription dataBefore;
	for (int i = 1; i <= 100; ++i) {
		dataBefore.addCluster(createClusterDescription(i));
	}
	for (int i = 1; i <= 100; ++i) {
		dataBefore.addParticle(createParticleDescription());
	}
	_access->updateData(dataBefore);

	DataChangeDescription dataChange;
	for (int i = 0; i <= 49; ++i) {
		{
			uint64_t id = dataBefore.clusters->at(i).id;
			auto pos = *dataBefore.clusters->at(i).pos;
			dataChange.addDeletedCluster(ClusterChangeDescription().setId(id).setPos(pos));
		}
		{
			uint64_t id = dataBefore.particles->at(i).id;
			auto pos = *dataBefore.particles->at(i).pos;
			dataChange.addDeletedParticle(ParticleChangeDescription().setId(id).setPos(pos));
		}
	}
	_access->updateData(dataChange);

	dataBefore.clusters->erase(dataBefore.clusters->begin(), dataBefore.clusters->begin() + 50);
	dataBefore.particles->erase(dataBefore.particles->begin(), dataBefore.particles->begin() + 50);

	IntRect rect = { { 0, 0 },{ _universeSize.x - 1, _universeSize.y - 1 } };
	DataDescription dataAfter = IntegrationTestHelper::getContent(_access, rect);

	ASSERT_TRUE(isCompatible(dataBefore, dataAfter));
}

TEST_F(DataDescriptionTransferTest, testModifyRandomParticles)
{
	DataDescription dataBefore;
	for (int i = 1; i <= 100; ++i) {
		dataBefore.addParticle(createParticleDescription());
	}
	_access->updateData(dataBefore);

	DataChangeDescription dataChange;
	for (int i = 0; i <= 49; ++i) {
		{
			auto &particle = dataBefore.particles->at(i);
			auto particleOriginal = particle;
			auto &pos = *particle.pos;
			pos = pos + QVector2D(100.0, 0);
			dataChange.addModifiedParticle(ParticleChangeDescription(particleOriginal, particle));
			_metric->correctPosition(pos);
		}
	}
	_access->updateData(dataChange);

	IntRect rect = { { 0, 0 },{ _universeSize.x - 1, _universeSize.y - 1 } };
	DataDescription dataAfter = IntegrationTestHelper::getContent(_access, rect);

	ASSERT_TRUE(isCompatible(dataBefore, dataAfter));
}

TEST_F(DataDescriptionTransferTest, testModifyRandomParticlesWithLargePositions)
{
	DataDescription dataBefore;
	for (int i = 1; i <= 100; ++i) {
		dataBefore.addParticle(createParticleDescription());
	}
	_access->updateData(dataBefore);

	DataChangeDescription dataChange;
	for (int i = 0; i <= 49; ++i) {
		{
			auto &particle = dataBefore.particles->at(i);
			auto particleOriginal = particle;
			auto &pos = *particle.pos;
			pos = pos + QVector2D(1000.0, 0);
			dataChange.addModifiedParticle(ParticleChangeDescription(particleOriginal, particle));
			_metric->correctPosition(pos);
		}
	}
	_access->updateData(dataChange);

	IntRect rect = { { 0, 0 },{ _universeSize.x - 1, _universeSize.y - 1 } };
	DataDescription dataAfter = IntegrationTestHelper::getContent(_access, rect);

	ASSERT_TRUE(isCompatible(dataBefore, dataAfter));
}

TEST_F(DataDescriptionTransferTest, testCreateAndDeleteAndModifyWithinSimulation)
{
	auto descHelper = _facade->buildDescriptionHelper();
	descHelper->init(_context);

	DataDescription dataBefore;
	for (int i = 1; i <= 10000; ++i) {
		dataBefore.addParticle(createParticleDescription());
	}
	_access->updateData(dataBefore);

	runSimulation(100, _controller);

	DataDescription extract = IntegrationTestHelper::getContent(_access, { { 0, 0 },{ _universeSize.x / 2, _universeSize.y / 2 } });
	DataDescription extractOriginal = extract;

	if (!extract.clusters || !extract.particles) {
		EXPECT_TRUE(false) << "no clusters or particles found";
		return;
	}
	unordered_set<uint64_t> cellIdsToModify;
	for (int clusterIndex = 0; clusterIndex < extract.clusters->size()/3; ++clusterIndex) {
		auto& cluster = extract.clusters->at(clusterIndex);
		for (int cellIndex = 0; cellIndex < cluster.cells->size() / 3; ++cellIndex) {
			auto& cell = cluster.cells->at(cellIndex);
			auto& pos = *cell.pos;
			pos = pos + QVector2D(1000.0, 0);
			cellIdsToModify.insert(cell.id);
		}
	}
	for (auto& particle : *extract.particles) {
		auto& pos = *particle.pos;
		pos = pos + QVector2D(1000.0, 420.0);
	}
	descHelper->reconnect(extract, extractOriginal, cellIdsToModify);
	_access->updateData(DataChangeDescription(extractOriginal, extract));

	runSimulation(100, _controller);

	EXPECT_TRUE(true);
}
