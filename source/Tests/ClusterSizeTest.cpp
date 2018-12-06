#include <gtest/gtest.h>

#include <QEventLoop>

#include "Base/ServiceLocator.h"
#include "Base/GlobalFactory.h"
#include "Base/NumberGenerator.h"
#include "ModelBasic/Settings.h"
#include "ModelBasic/ModelBasicBuilderFacade.h"
#include "ModelBasic/SimulationContext.h"
#include "ModelBasic/SimulationController.h"
#include "ModelBasic/DescriptionHelper.h"
#include "ModelBasic/SimulationParameters.h"
#include "ModelBasic/SpaceProperties.h"
#include "ModelBasic/SimulationAccess.h"
#include "ModelBasic/Serializer.h"

#include "ModelCpu/SimulationContextCpuImpl.h"
#include "ModelCpu/SimulationControllerCpu.h"
#include "ModelCpu/SimulationAccessCpu.h"
#include "ModelCpu/ModelCpuData.h"
#include "ModelCpu/ModelCpuBuilderFacade.h"

#include "Tests/Predicates.h"

#include "IntegrationTestHelper.h"
#include "IntegrationTestFramework.h"

class ClusterSizeTest
	: public IntegrationTestFramework
{
public:
	ClusterSizeTest();
	~ClusterSizeTest();

protected:
	SimulationControllerCpu* _controller = nullptr;
	SimulationContext* _context = nullptr;
	SpaceProperties* _space = nullptr;
	SimulationAccessCpu* _access = nullptr;
	IntVector2D _gridSize{ 12, 6 };
};

ClusterSizeTest::ClusterSizeTest()
	: IntegrationTestFramework({ 600, 300 })
{
	GlobalFactory* factory = ServiceLocator::getInstance().getService<GlobalFactory>();
	_controller = _cpuFacade->buildSimulationController({ _universeSize, _symbols, _parameters }, ModelCpuData(1, _gridSize), 0);
	_context = _controller->getContext();
	_space = _context->getSpaceProperties();
	_access = _cpuFacade->buildSimulationAccess();
	_access->init(_controller);
}

ClusterSizeTest::~ClusterSizeTest()
{
	delete _access;
	delete _controller;
}

TEST_F(ClusterSizeTest, testDistanceToNeighbors)
{
	DataDescription data;
	for (int i = 1; i <= 10000; ++i) {
		data.addParticle(createParticleDescription());
	}
	_access->updateData(data);

	IntegrationTestHelper::runSimulation(50, _controller);

	//check result
	DataDescription extract = IntegrationTestHelper::getContent(_access, { { 0, 0 }, _universeSize });
	DescriptionNavigator navi;
	navi.update(extract);
	if (extract.clusters) {
		for (auto const& cluster : *extract.clusters) {
			if (cluster.cells) {
				for (auto const& cell : *cluster.cells) {
					if (cell.connectingCells) {
						for (uint64_t connectingCellId : *cell.connectingCells) {
							auto const& connectingCell = cluster.cells->at(navi.cellIndicesByCellIds.at(connectingCellId));
							auto distance = *cell.pos - *connectingCell.pos;
							_space->correctDisplacement(distance);
							ASSERT_TRUE(predLessThanMediumPrecision(distance.length(), _parameters->cellMaxDistance));
						}
					}
				}
			}
		}
	}
}
