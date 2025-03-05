#include <gtest/gtest.h>

#include "EngineInterface/DescriptionEditService.h"
#include "EngineInterface/Descriptions.h"
#include "EngineInterface/SimulationFacade.h"
#include "EngineInterface/GenomeDescriptionConverterService.h"
#include "EngineInterface/GenomeDescriptions.h"

#include "IntegrationTestFramework.h"

class EnergyFlowTests_New : public IntegrationTestFramework
{
public:
    EnergyFlowTests_New()
        : IntegrationTestFramework()
    {}

    ~EnergyFlowTests_New() = default;
};

TEST_F(EnergyFlowTests_New, energyFlowsLeadsEqualDistribution)
{
    auto genome = GenomeDescriptionConverterService::get().convertDescriptionToBytes(GenomeDescription().cells({CellGenomeDescription()}));

    DataDescription data;
    for (int i = 0; i < 20; ++i) {
        auto cell = CellDescription().id(i + 1).pos({100.0f + toFloat(i), 100.0f});
        data.addCell(cell);
        if (i > 0) {
            data.addConnection(i, i + 1);
        }
    }
    data._cells.at(0)._energy = 10000.0f;

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1000);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualCellById = getCellById(actualData);

    for (int i = 0; i < 20; ++i) {
        EXPECT_TRUE(actualCellById.at(i + 1)._energy < 600.0f);
    }
    EXPECT_TRUE(approxCompare(getEnergy(data), getEnergy(actualData)));
}

TEST_F(EnergyFlowTests_New, energyFlowsToActiveConstructor)
{
    auto genome = GenomeDescriptionConverterService::get().convertDescriptionToBytes(GenomeDescription().cells({CellGenomeDescription()}));

    DataDescription data;
    for (int i = 0; i < 20; ++i) {
        auto cell = CellDescription().id(i + 1).pos({100.0f + toFloat(i), 100.0f});
        if (i == 19) {
            cell.cellType(ConstructorDescription().genome(genome).autoTriggerInterval(0));
        }
        data.addCell(cell);
        if (i > 0) {
            data.addConnection(i, i + 1);
        }
    }
    data._cells.at(0)._energy = 10000.0f;

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(2000);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualCellById = getCellById(actualData);

    ASSERT_EQ(20, actualData._cells.size());

    for (int i = 1; i < 21; ++i) {
        if (i == 20) {
            EXPECT_TRUE(actualCellById.at(i)._energy > 10000.0f - 400.0f);
        } else {
            EXPECT_TRUE(actualCellById.at(i)._energy < 200.0f);
        }
    }
}

TEST_F(EnergyFlowTests_New, energyFlowsToClosestActiveConstructor)
{
    auto genome = GenomeDescriptionConverterService::get().convertDescriptionToBytes(GenomeDescription().cells({CellGenomeDescription()}));

    auto constructorId1 = 10 + 1;
    auto constructorId2 = 20 + 19 + 1;

    DataDescription data;
    for (int j = 0; j < 2; ++j) {
        for (int i = 0; i < 20; ++i) {
            auto id = i + j * 20 + 1;
            auto cell = CellDescription().id(id).pos({100.0f + toFloat(i), 100.0f});
            if (id == constructorId1 || id == constructorId2) {
                cell.cellType(ConstructorDescription().genome(genome).autoTriggerInterval(0));
            }
            data.addCell(cell);
            if (i > 0) {
                data.addConnection(id - 1, id);
            }
            if (j == 1) {
                data.addConnection(id - 20, id);
            }
        }
    }
    data._cells.at(0)._energy = 10000.0f;

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1000);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualCellById = getCellById(actualData);

    for (int i = 1; i < 41; ++i) {
        if (i == constructorId1) {
            EXPECT_TRUE(actualCellById.at(i)._energy > 10000.0f - 400.0f);
        } else {
            EXPECT_TRUE(actualCellById.at(i)._energy < 200.0f);
        }
    }
}

TEST_F(EnergyFlowTests_New, energyFlowsNotToActiveConstructor)
{
    auto genome = GenomeDescriptionConverterService::get().convertDescriptionToBytes(
        GenomeDescription().header(GenomeHeaderDescription().separateConstruction(false)).cells({CellGenomeDescription()}));

    DataDescription data;
    for (int i = 0; i < 20; ++i) {
        auto cell = CellDescription().id(i + 1).pos({100.0f + toFloat(i), 100.0f});
        if (i == 19) {
            cell.cellType(ConstructorDescription().genome(genome).autoTriggerInterval(0).genomeCurrentBranch(1));
        }
        data.addCell(cell);
        if (i > 0) {
            data.addConnection(i, i + 1);
        }
    }
    data._cells.at(0)._energy = 10000.0f;

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1000);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualCellById = getCellById(actualData);

    for (int i = 0; i < 20; ++i) {
        EXPECT_TRUE(actualCellById.at(i + 1)._energy < 600.0f);
    }
    EXPECT_TRUE(approxCompare(getEnergy(data), getEnergy(actualData)));
}
