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

TEST_F(DataDescriptionTransferGpuTest, testCreateClusterDescriptionWithCompleteCell)
{
	DataDescription dataBefore;
	dataBefore.addCluster(createSingleCellClusterWithCompleteData());
	_access->updateData(dataBefore);

	IntRect rect = { { 0, 0 },{ _universeSize.x - 1, _universeSize.y - 1 } };
	DataDescription dataAfter = IntegrationTestHelper::getContent(_access, rect);

	ASSERT_TRUE(isCompatible(dataBefore, dataAfter));
}
