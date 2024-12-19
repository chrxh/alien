#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <ranges>

#include <boost/range/combine.hpp>

#include <gtest/gtest.h>

#include "EngineInterface/DescriptionEditService.h"
#include "EngineInterface/Descriptions.h"
#include "EngineInterface/GenomeDescriptionService.h"
#include "EngineInterface/SimulationFacade.h"
#include "IntegrationTestFramework.h"

class CellLifeCycleTests : public IntegrationTestFramework
{
public:
    CellLifeCycleTests()
        : IntegrationTestFramework()
    {
        _parameters.features.advancedCellLifeCycleControl = true;
        _parameters.cellMinReplicatorGenomeSize[0] = 3;
        _simulationFacade->setSimulationParameters(_parameters);
    }

    ~CellLifeCycleTests() = default;

};

TEST_F(CellLifeCycleTests, mutationCheck_replicatorWithGenomeBelowMinSize)
{
    auto subGenome = GenomeDescriptionService::get().convertDescriptionToBytes(GenomeDescription().setCells({CellGenomeDescription()}));
    auto genome = GenomeDescriptionService::get().convertDescriptionToBytes(GenomeDescription().setCells(
        {CellGenomeDescription().setCellFunction(ConstructorGenomeDescription().setMakeSelfCopy()),
         CellGenomeDescription().setCellFunction(ConstructorGenomeDescription().setGenome(subGenome))}));

    auto data = DataDescription().addCells(
        {CellDescription().setId(1).setCellFunction(ConstructorDescription().setGenome(genome)).setLivingState(LivingState_Activating)});

    _simulationFacade->setSimulationData(data);
    _simulationFacade->testOnly_mutationCheck(1);

    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(1, actualData.cells.size());
    auto actualCell = getCell(actualData, 1);
    EXPECT_EQ(LivingState_Dying, actualCell.livingState);
}

TEST_F(CellLifeCycleTests, mutationCheck_replicatorWithGenomeAboveMinSize)
{
    auto subGenome = GenomeDescriptionService::get().convertDescriptionToBytes(GenomeDescription().setCells({CellGenomeDescription()}));
    auto genome = GenomeDescriptionService::get().convertDescriptionToBytes(GenomeDescription().setCells(
        {CellGenomeDescription().setCellFunction(ConstructorGenomeDescription().setMakeSelfCopy()),
         CellGenomeDescription().setCellFunction(ConstructorGenomeDescription().setGenome(subGenome)),
         CellGenomeDescription()}));

    auto data = DataDescription().addCells(
        {CellDescription().setId(1).setCellFunction(ConstructorDescription().setGenome(genome)).setLivingState(LivingState_Activating)});

    _parameters.features.advancedCellLifeCycleControl = true;
    _parameters.cellMinReplicatorGenomeSize[0] = 3;
    _simulationFacade->setSimulationParameters(_parameters);
    _simulationFacade->setSimulationData(data);
    _simulationFacade->testOnly_mutationCheck(1);

    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(1, actualData.cells.size());
    auto actualCell = getCell(actualData, 1);
    EXPECT_EQ(LivingState_Activating, actualCell.livingState);
}
