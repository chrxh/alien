#include <gtest/gtest.h>

#include <QEventLoop>

#include "Base/ServiceLocator.h"
#include "Base/GlobalFactory.h"
#include "Base/NumberGenerator.h"
#include "ModelBasic/ModelBasicBuilderFacade.h"
#include "ModelBasic/Settings.h"
#include "ModelBasic/SimulationController.h"
#include "ModelBasic/DescriptionHelper.h"
#include "ModelBasic/SimulationParameters.h"
#include "ModelBasic/SpaceProperties.h"
#include "ModelBasic/SimulationAccess.h"

#include "ModelCpu/SimulationContextCpuImpl.h"
#include "ModelCpu/SimulationControllerCpu.h"
#include "ModelCpu/SimulationAccessCpu.h"
#include "ModelCpu/ModelCpuData.h"
#include "ModelCpu/ModelCpuBuilderFacade.h"

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
	SimulationControllerCpu* _controller = nullptr;
	SimulationContextCpuImpl* _context = nullptr;
	SpaceProperties* _spaceProp = nullptr;
	SimulationAccessCpu* _access = nullptr;
	IntVector2D _gridSize{ 6, 6 };
};

DataDescriptionTransferTest::DataDescriptionTransferTest()
	: IntegrationTestFramework({ 600, 300 })
{
	_controller = _cpuFacade->buildSimulationController({ _universeSize, _symbols, _parameters }, ModelCpuData(1, _gridSize), 0);
	_context = static_cast<SimulationContextCpuImpl*>(_controller->getContext());
	_spaceProp = _context->getSpaceProperties();
	_access = _cpuFacade->buildSimulationAccess();
	_access->init(_controller);
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
			_spaceProp->correctPosition(pos);
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
			_spaceProp->correctPosition(pos);
		}
	}
	_access->updateData(dataChange);

	IntRect rect = { { 0, 0 },{ _universeSize.x - 1, _universeSize.y - 1 } };
	DataDescription dataAfter = IntegrationTestHelper::getContent(_access, rect);

	ASSERT_TRUE(isCompatible(dataBefore, dataAfter));
}

TEST_F(DataDescriptionTransferTest, testCreateAndDeleteAndModifyWithinSimulation)
{
	auto factory = ServiceLocator::getInstance().getService<GlobalFactory>();
	auto descHelper = _basicFacade->buildDescriptionHelper();
	auto numberGen = factory->buildRandomNumberGenerator();
	numberGen->init(NUMBER_GENERATOR_ARRAY_SIZE, 0);
	descHelper->init(_context, numberGen);

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
