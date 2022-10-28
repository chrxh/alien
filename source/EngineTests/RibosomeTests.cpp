#include <gtest/gtest.h>

#include "Base/NumberGenerator.h"
#include "EngineInterface/DescriptionHelper.h"
#include "EngineInterface/Descriptions.h"
#include "EngineInterface/SimulationController.h"
#include "IntegrationTestFramework.h"

class RibosomeTests : public IntegrationTestFramework
{
public:
    RibosomeTests()
        : IntegrationTestFramework({1000, 1000})
    {}

    ~RibosomeTests() = default;
};

TEST_F(RibosomeTests, constructNeuronCell_noSeparation)
{
    auto parameters = _simController->getSimulationParameters();

    DataDescription data;
    data.addCell(CellDescription()
                     .setId(1)
                     .setEnergy(parameters.cellNormalEnergy * 3)
                     .setMaxConnections(1)
                     .setExecutionOrderNumber(0)
                     .setCellFunction(
                         RibosomeDescription().setGenome({CellDescription().setCellFunction(NeuronDescription()).setColor(2).setExecutionOrderNumber(4)}).setSeparateConstruction(false)));

    _simController->setSimulationData(data);
    _simController->calcSingleTimestep();
    auto actualData = _simController->getSimulationData();

    EXPECT_EQ(2, actualData.cells.size());

    bool hostCellChecked = false;
    bool constructedCellChecked = false;
    for (auto const& cell : actualData.cells) {
        if (cell.id == 1) {
            hostCellChecked = true;
            EXPECT_EQ(1, cell.connections.size());
            expectApproxEqual(parameters.cellNormalEnergy * 2, cell.energy);
        } else {
            constructedCellChecked = true;
            EXPECT_EQ(1, cell.connections.size());
            EXPECT_EQ(1, cell.maxConnections);
            EXPECT_EQ(2, cell.color);
            EXPECT_EQ(4, cell.executionOrderNumber);
            EXPECT_EQ(Enums::CellFunction_Neuron, cell.getCellFunctionType());
            EXPECT_FALSE(cell.underConstruction);
            expectApproxEqual(parameters.cellNormalEnergy, cell.energy);
        }
    }
    EXPECT_TRUE(hostCellChecked);
    EXPECT_TRUE(constructedCellChecked);
}
