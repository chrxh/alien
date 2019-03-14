#include <boost/range/adaptors.hpp>
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
#include "ModelBasic/SimulationContext.h"

#include "ModelGpu/SimulationControllerGpu.h"
#include "ModelGpu/SimulationAccessGpu.h"
#include "ModelGpu/ModelGpuData.h"
#include "ModelGpu/ModelGpuBuilderFacade.h"

#include "Tests/Predicates.h"

#include "IntegrationTestHelper.h"
#include "IntegrationTestFramework.h"

class DataDescriptionTransferGpuTest
	: public IntegrationTestFramework
{
public:
	DataDescriptionTransferGpuTest();
	~DataDescriptionTransferGpuTest();

protected:
	SimulationControllerGpu* _controller = nullptr;
	SimulationContext* _context = nullptr;
	SpaceProperties* _spaceProp = nullptr;
	SimulationAccessGpu* _access = nullptr;
	IntVector2D _gridSize{ 6, 6 };
};

DataDescriptionTransferGpuTest::DataDescriptionTransferGpuTest()
	: IntegrationTestFramework({ 600, 300 })
{
	_controller = _gpuFacade->buildSimulationController({ _universeSize, _symbols, _parameters }, ModelGpuData(), 0);
	_context = _controller->getContext();
	_spaceProp = _context->getSpaceProperties();
	_access = _gpuFacade->buildSimulationAccess();
	_access->init(_controller);
	_numberGen = _context->getNumberGenerator();
}

DataDescriptionTransferGpuTest::~DataDescriptionTransferGpuTest()
{
	delete _access;
	delete _controller;
}

TEST_F(DataDescriptionTransferGpuTest, DISABLED_testCreateClusterWithCompleteCell)
{
	DataDescription dataBefore;
	dataBefore.addCluster(createSingleCellClusterWithCompleteData());
	IntegrationTestHelper::updateData(_access, dataBefore);

	IntRect rect = { { 0, 0 },{ _universeSize.x - 1, _universeSize.y - 1 } };
	DataDescription dataAfter = IntegrationTestHelper::getContent(_access, rect);

	ASSERT_TRUE(isCompatible(dataBefore, dataAfter));
}

/**
* Situation: change particle properties
* Expected result: particle in simulation changed
*/
TEST_F(DataDescriptionTransferGpuTest, DISABLED_testChangeParticle)
{
	DataDescription dataBefore;
	auto particleEnergy1 = _parameters.cellMinEnergy / 2.0;
	auto particleId = _numberGen->getId();
	auto particleBefore = ParticleDescription().setId(particleId).setEnergy(particleEnergy1).setPos({ 100, 100 }).setVel({ 0.5f, 0.0f });
	dataBefore.addParticle(particleBefore);
	
	DataDescription dataChanged;
	auto particleEnergy2 = _parameters.cellMinEnergy / 3.0;
	auto particleChange = ParticleDescription().setId(particleId).setEnergy(particleEnergy2).setPos({ 150, 150 }).setVel({ 0.0f, -0.3f });
	dataChanged.addParticle(particleChange);

	IntegrationTestHelper::updateData(_access, dataBefore);
	IntegrationTestHelper::updateData(_access, DataChangeDescription(dataBefore, dataChanged));

	IntRect rect = { { 0, 0 },{ _universeSize.x - 1, _universeSize.y - 1 } };
	DataDescription dataAfter = IntegrationTestHelper::getContent(_access, rect);

	ASSERT_TRUE(isCompatible(dataChanged, dataAfter));
}

/**
* Situation: create cluster and particle at a position outside universe
* Expected result: cluster and particle should be positioned inside universe due to torus topology
*/
TEST_F(DataDescriptionTransferGpuTest, testCreateDataOutsideBoundaries)
{
	auto universeSize = _spaceProp->getSize();
	DataDescription dataBefore;
	dataBefore.addCluster(createHorizontalCluster(3, QVector2D{ 2.5f * universeSize.x, 2.5f * universeSize.y}));
	dataBefore.addParticle(createParticle(QVector2D{ 2.5f * universeSize.x + 2.0f, 2.5f * universeSize.y + 2.0f }));

	IntegrationTestHelper::updateData(_access, dataBefore);
	DataDescription dataAfter = IntegrationTestHelper::getContent(_access, { { 0, 0 }, { _universeSize.x - 1, _universeSize.y - 1 } });

	EXPECT_EQ(1, dataAfter.clusters->size());
	auto origCluster = dataBefore.clusters->at(0);
	auto newCluster = dataAfter.clusters->at(0);
	EXPECT_TRUE(isCompatible(*origCluster.pos - QVector2D{ 2.0f * universeSize.x, 2.0f * universeSize.y }, *newCluster.pos));

	unordered_map<uint64_t, CellDescription> origCellById = IntegrationTestHelper::getCellByCellId(dataBefore);
	unordered_map<uint64_t, CellDescription> newCellById = IntegrationTestHelper::getCellByCellId(dataAfter);
	for (CellDescription const& origCell : origCellById | boost::adaptors::map_values) {
		CellDescription newCell = newCellById.at(origCell.id);
		EXPECT_TRUE(isCompatible(*origCell.pos - QVector2D{ 2.0f * universeSize.x, 2.0f * universeSize.y }, *newCell.pos));
	}

	EXPECT_EQ(1, dataAfter.particles->size());
	auto origParticle = dataBefore.particles->at(0);
	auto newParticle = dataAfter.particles->at(0);
	EXPECT_TRUE(isCompatible(*origParticle.pos - QVector2D{ 2.0f * universeSize.x, 2.0f * universeSize.y }, *newParticle.pos));
}

/**
* Situation:
* 	- one cluster with 10x10 cell and one particle
*	- particle and some cells are moved via update
* Fixed error: crash after moving cells in a cluster in item view
* Expected result: no crash
*/
TEST_F(DataDescriptionTransferGpuTest, regressionTest_changeData)
{
	auto descHelper = _basicFacade->buildDescriptionHelper();
	descHelper->init(_context);

	auto size = _spaceProp->getSize();
	DataDescription dataBefore;
	dataBefore.addCluster(createRectangularCluster({ 10, 10 }, QVector2D{ size.x / 2.0f, size.y / 2.0f }, QVector2D{}));
	dataBefore.addParticle(createParticle(QVector2D{ 0, 0 }));

	IntegrationTestHelper::updateData(_access, dataBefore);

	auto dataModified = dataBefore;
	unordered_set<uint64_t> idsOfChangedCells;
	for(int i = 0; i < 10; ++i) {
		auto& cluster = dataModified.clusters->at(0);
		auto& cell = cluster.cells->at(i);
		cell.pos->setX(cell.pos->x() + 50.0f);
		idsOfChangedCells.insert(cell.id);
	}
	auto& particle = dataModified.particles->at(0);
	particle.pos->setX(particle.pos->x() + 50.0f);

	descHelper->reconnect(dataModified, dataBefore, idsOfChangedCells);
	EXPECT_NO_THROW(IntegrationTestHelper::updateData(_access, DataChangeDescription(dataBefore, dataModified)));
}