#include <gtest/gtest.h>

#include "Base/Math.h"
#include "EngineInterface/DescriptionHelper.h"
#include "EngineInterface/Descriptions.h"
#include "EngineInterface/GenomeEncoder.h"
#include "EngineInterface/SimulationController.h"
#include "IntegrationTestFramework.h"
#include "Base/NumberGenerator.h"

class ConstructorTests : public IntegrationTestFramework
{
public:
    ConstructorTests()
        : IntegrationTestFramework()
    {}

    ~ConstructorTests() = default;

protected:
    void expectLowPrecisionEqual(float left, float right) const { EXPECT_TRUE(std::abs(left - right) < 0.05f); }

    std::vector<uint8_t> createRandomGenome(int size) const
    {
        std::vector<uint8_t> result;
        result.reserve(size);
        for (int i = 0; i < size; ++i) {
            result.emplace_back(static_cast<uint8_t>(NumberGenerator::getInstance().getRandomInt(256)));
        }
        return result;
    }
};

TEST_F(ConstructorTests, noEnergy)
{
    auto genome = GenomeEncoder::encode({CellDescription()});

    DataDescription data;
    data.addCell(
        CellDescription()
            .setId(1)
            .setEnergy(_parameters.cellNormalEnergy * 2 - 1.0f)
            .setMaxConnections(1)
            .setExecutionOrderNumber(0)
                     .setCellFunction(ConstructorDescription().setGenome(genome).setSeparateConstruction(false)));

    _simController->setSimulationData(data);
    _simController->calcSingleTimestep();
    auto actualData = _simController->getSimulationData();

    EXPECT_EQ(1, actualData.cells.size());
    auto actualHostCell = getCell(actualData, 1);

    EXPECT_EQ(0, actualHostCell.connections.size());
    EXPECT_EQ(0, std::get<ConstructorDescription>(*actualHostCell.cellFunction).currentGenomePos);
    expectApproxEqual(_parameters.cellNormalEnergy * 2 - 1.0f, actualHostCell.energy);
    expectApproxEqual(0.0f, actualHostCell.activity.channels[0]);
}

TEST_F(ConstructorTests, alreadyFinished)
{
    DataDescription data;

    auto genome = GenomeEncoder::encode({CellDescription()});

    auto constructor = ConstructorDescription().setGenome(genome).setSingleConstruction(true);
    constructor.setCurrentGenomePos(constructor.genome.size());

    data.addCell(
        CellDescription()
            .setId(1)
            .setEnergy(_parameters.cellNormalEnergy * 3)
            .setMaxConnections(1)
            .setExecutionOrderNumber(0)
            .setCellFunction(constructor));

    _simController->setSimulationData(data);
    _simController->calcSingleTimestep();
    auto actualData = _simController->getSimulationData();

    EXPECT_EQ(1, actualData.cells.size());
    auto actualHostCell = getCell(actualData, 1);
    auto actualConstructor = std::get<ConstructorDescription>(*actualHostCell.cellFunction);
    EXPECT_EQ(0, actualHostCell.connections.size());
    EXPECT_EQ(actualConstructor.genome.size(), actualConstructor.currentGenomePos);
    expectApproxEqual(0.0f, actualHostCell.activity.channels[0]);
}

TEST_F(ConstructorTests, manualConstruction_noInputActivity)
{
    auto genome = GenomeEncoder::encode({CellDescription()});

    DataDescription data;
    data.addCell(
        CellDescription()
            .setId(1)
            .setEnergy(_parameters.cellNormalEnergy * 3)
            .setMaxConnections(1)
            .setExecutionOrderNumber(0)
                     .setCellFunction(
                         ConstructorDescription().setMode(Enums::ConstructionMode_Manual).setGenome(genome)));

    _simController->setSimulationData(data);
    _simController->calcSingleTimestep();
    auto actualData = _simController->getSimulationData();

    EXPECT_EQ(1, actualData.cells.size());
    auto actualHostCell = getCell(actualData, 1);

    EXPECT_EQ(0, actualHostCell.connections.size());
    EXPECT_EQ(0, std::get<ConstructorDescription>(*actualHostCell.cellFunction).currentGenomePos);
    expectApproxEqual(_parameters.cellNormalEnergy * 3, actualHostCell.energy);
    expectApproxEqual(0.0f, actualHostCell.activity.channels[0]);
}


TEST_F(ConstructorTests, constructSingleCell_noSeparation)
{
    auto genome = GenomeEncoder::encode({CellDescription().setColor(2).setExecutionOrderNumber(4)});

    DataDescription data;
    data.addCell(CellDescription()
                     .setId(1)
                     .setPos({10.0f, 10.0f})
                     .setEnergy(_parameters.cellNormalEnergy * 3)
                     .setMaxConnections(1)
                     .setExecutionOrderNumber(0)
                     .setCellFunction(ConstructorDescription().setGenome(genome).setSeparateConstruction(false)));

    _simController->setSimulationData(data);
    _simController->calcSingleTimestep();
    auto actualData = _simController->getSimulationData();

    EXPECT_EQ(2, actualData.cells.size());
    auto actualHostCell = getCell(actualData, 1);
    auto actualConstructedCell = getOtherCell(actualData, 1);

    EXPECT_EQ(1, actualHostCell.connections.size());
    EXPECT_EQ(0, std::get<ConstructorDescription>(*actualHostCell.cellFunction).currentGenomePos);
    expectApproxEqual(_parameters.cellNormalEnergy * 2, actualHostCell.energy);
    expectApproxEqual(1.0f, actualHostCell.activity.channels[0]);

    EXPECT_EQ(1, actualConstructedCell.connections.size());
    EXPECT_EQ(1, actualConstructedCell.maxConnections);
    EXPECT_EQ(2, actualConstructedCell.color);
    EXPECT_EQ(4, actualConstructedCell.executionOrderNumber);
    EXPECT_FALSE(actualConstructedCell.underConstruction);
    EXPECT_EQ(Enums::CellFunction_None, actualConstructedCell.getCellFunctionType());
    expectApproxEqual(_parameters.cellNormalEnergy, actualConstructedCell.energy);
    expectApproxEqual(1.6f, Math::length(actualHostCell.pos - actualConstructedCell.pos));
}

TEST_F(ConstructorTests, constructSingleCell_separation)
{
    auto genome = GenomeEncoder::encode({CellDescription()});

    DataDescription data;
    data.addCell(CellDescription()
                     .setId(1)
                     .setEnergy(_parameters.cellNormalEnergy * 3)
                     .setMaxConnections(1)
                     .setExecutionOrderNumber(0)
                     .setCellFunction(ConstructorDescription()
                                          .setGenome(genome)
                                          .setSeparateConstruction(true)));

    _simController->setSimulationData(data);
    _simController->calcSingleTimestep();
    auto actualData = _simController->getSimulationData();

    EXPECT_EQ(2, actualData.cells.size());
    auto actualHostCell = getCell(actualData, 1);
    auto actualConstructedCell = getOtherCell(actualData, 1);

    EXPECT_EQ(0, actualHostCell.connections.size());
    EXPECT_EQ(0, actualHostCell.maxConnections);

    EXPECT_EQ(0, actualConstructedCell.connections.size());
    EXPECT_EQ(0, actualConstructedCell.maxConnections);
    EXPECT_FALSE(actualConstructedCell.underConstruction);
}

TEST_F(ConstructorTests, constructSingleCell_makeSticky)
{
    auto genome = GenomeEncoder::encode({CellDescription().setMaxConnections(3)});

    DataDescription data;
    data.addCell(CellDescription()
                     .setId(1)
                     .setEnergy(_parameters.cellNormalEnergy * 3)
                     .setMaxConnections(1)
                     .setExecutionOrderNumber(0)
                     .setCellFunction(ConstructorDescription().setGenome(genome).setSeparateConstruction(true).setMakeSticky(true)));

    _simController->setSimulationData(data);
    _simController->calcSingleTimestep();
    auto actualData = _simController->getSimulationData();

    EXPECT_EQ(2, actualData.cells.size());
    auto actualHostCell = getCell(actualData, 1);
    auto actualConstructedCell = getOtherCell(actualData, 1);

    EXPECT_EQ(0, actualHostCell.connections.size());
    EXPECT_EQ(1, actualHostCell.maxConnections);

    EXPECT_EQ(0, actualConstructedCell.connections.size());
    EXPECT_EQ(3, actualConstructedCell.maxConnections);
    EXPECT_FALSE(actualConstructedCell.underConstruction);
}

TEST_F(ConstructorTests, constructSingleCell_singleConstruction)
{
    auto genome = GenomeEncoder::encode({CellDescription()});

    DataDescription data;
    data.addCell(
        CellDescription()
            .setId(1)
            .setEnergy(_parameters.cellNormalEnergy * 3)
            .setMaxConnections(1)
            .setExecutionOrderNumber(0)
                     .setCellFunction(ConstructorDescription().setGenome(genome).setSeparateConstruction(true).setSingleConstruction(true)));

    _simController->setSimulationData(data);
    _simController->calcSingleTimestep();
    auto actualData = _simController->getSimulationData();

    EXPECT_EQ(2, actualData.cells.size());
    auto actualHostCell = getCell(actualData, 1);
    auto actualConstructedCell = getOtherCell(actualData, 1);

    EXPECT_EQ(0, actualHostCell.connections.size());
    auto const& constructor = std::get<ConstructorDescription>(*actualHostCell.cellFunction);
    EXPECT_EQ(constructor.genome.size(), constructor.currentGenomePos);

    EXPECT_EQ(0, actualConstructedCell.connections.size());
    EXPECT_FALSE(actualConstructedCell.underConstruction);
}

TEST_F(ConstructorTests, constructSingleCell_manualConstruction)
{
    auto genome = GenomeEncoder::encode({CellDescription()});

    DataDescription data;
    data.addCells({
       CellDescription()
            .setId(1)
            .setPos({10.0f, 10.0f})
            .setEnergy(_parameters.cellNormalEnergy * 3)
            .setMaxConnections(2)
            .setExecutionOrderNumber(0)
             .setCellFunction(ConstructorDescription().setMode(Enums::ConstructionMode_Manual).setGenome(genome)),
        CellDescription()
             .setId(2)
             .setPos({11.0f, 10.0f})
             .setEnergy(100)
             .setMaxConnections(1)
             .setExecutionOrderNumber(5)
             .setCellFunction(NerveDescription())
             .setActivity({1, 0, 0, 0, 0, 0, 0, 0})
    });
    data.addConnection(1, 2);

    _simController->setSimulationData(data);
    _simController->calcSingleTimestep();
    auto actualData = _simController->getSimulationData();

    EXPECT_EQ(3, actualData.cells.size());
    auto actualHostCell = getCell(actualData, 1);
    auto actualConstructedCell = getOtherCell(actualData, {1, 2});

    EXPECT_EQ(1, actualHostCell.connections.size());

    EXPECT_EQ(0, actualConstructedCell.connections.size());
    EXPECT_FALSE(actualConstructedCell.underConstruction);

    expectApproxEqual(10.0f - 1.6f, actualConstructedCell.pos.x);
    expectApproxEqual(10.0f, actualConstructedCell.pos.y);
}

TEST_F(ConstructorTests, constructSingleCell_differentAngle1)
{
    auto genome = GenomeEncoder::encode({CellDescription()}, 90.0f);

    DataDescription data;
    data.addCells({
        CellDescription()
             .setId(1)
             .setPos({10.0f, 10.0f})
             .setEnergy(_parameters.cellNormalEnergy * 3)
             .setMaxConnections(2)
             .setExecutionOrderNumber(0)
             .setCellFunction(ConstructorDescription().setMode(Enums::ConstructionMode_Manual).setGenome(genome)),
        CellDescription()
             .setId(2)
             .setPos({11.0f, 10.0f})
             .setEnergy(100)
             .setMaxConnections(1)
             .setExecutionOrderNumber(5)
             .setCellFunction(NerveDescription())
             .setActivity({1, 0, 0, 0, 0, 0, 0, 0})});
    data.addConnection(1, 2);

    _simController->setSimulationData(data);
    _simController->calcSingleTimestep();
    auto actualData = _simController->getSimulationData();

    EXPECT_EQ(3, actualData.cells.size());
    auto actualHostCell = getCell(actualData, 1);
    auto actualConstructedCell = getOtherCell(actualData, {1, 2});

    expectApproxEqual(10.0f, actualConstructedCell.pos.x);
    expectApproxEqual(10.0f - 1.6f, actualConstructedCell.pos.y);
}

TEST_F(ConstructorTests, constructSingleCell_differentAngle2)
{
    auto genome = GenomeEncoder::encode({CellDescription()}, -90.0f);

    DataDescription data;
    data.addCells(
        {CellDescription()
             .setId(1)
             .setPos({10.0f, 10.0f})
             .setEnergy(_parameters.cellNormalEnergy * 3)
             .setMaxConnections(2)
             .setExecutionOrderNumber(0)
             .setCellFunction(ConstructorDescription().setMode(Enums::ConstructionMode_Manual).setGenome(genome)),
         CellDescription()
             .setId(2)
             .setPos({11.0f, 10.0f})
             .setEnergy(100)
             .setMaxConnections(1)
             .setExecutionOrderNumber(5)
             .setCellFunction(NerveDescription())
             .setActivity({1, 0, 0, 0, 0, 0, 0, 0})});
    data.addConnection(1, 2);

    _simController->setSimulationData(data);
    _simController->calcSingleTimestep();
    auto actualData = _simController->getSimulationData();

    EXPECT_EQ(3, actualData.cells.size());
    auto actualHostCell = getCell(actualData, 1);
    auto actualConstructedCell = getOtherCell(actualData, {1, 2});

    expectApproxEqual(10.0f, actualConstructedCell.pos.x);
    expectApproxEqual(10.0f + 1.6f, actualConstructedCell.pos.y);
}

TEST_F(ConstructorTests, constructNeuronCell)
{
    auto neuron = NeuronDescription();
    neuron.weights[1][7] = 1.0f;
    neuron.weights[7][1] = -1.0f;
    neuron.bias[3] = 1.8f;

    auto genome = GenomeEncoder::encode({CellDescription().setCellFunction(neuron)});

    DataDescription data;
    data.addCell(
        CellDescription()
            .setId(1)
            .setEnergy(_parameters.cellNormalEnergy * 3)
            .setMaxConnections(1)
            .setExecutionOrderNumber(0)
                     .setCellFunction(ConstructorDescription().setGenome(genome)));

    _simController->setSimulationData(data);
    _simController->calcSingleTimestep();
    auto actualData = _simController->getSimulationData();

    EXPECT_EQ(2, actualData.cells.size());
    auto actualConstructedCell = getOtherCell(actualData, 1);

    EXPECT_EQ(Enums::CellFunction_Neuron, actualConstructedCell.getCellFunctionType());

    auto actualNeuron = std::get<NeuronDescription>(*actualConstructedCell.cellFunction);
    for (int row = 0; row < MAX_CHANNELS; ++row) {
        for (int col = 0; col < MAX_CHANNELS; ++col) {
            expectLowPrecisionEqual(neuron.weights[row][col], actualNeuron.weights[row][col]);
        }
    }
    for (int i = 0; i < MAX_CHANNELS; ++i) {
        expectLowPrecisionEqual(neuron.bias[i], actualNeuron.bias[i]);
    }
}

TEST_F(ConstructorTests, constructConstructorCell)
{
    auto constructedConstructor = ConstructorDescription()
                           .setMode(Enums::ConstructionMode_Manual)
                           .setSingleConstruction(true)
                           .setSeparateConstruction(false)
                           .setMakeSticky(true)
                           .setAngleAlignment(2)
                           .setGenome(createRandomGenome(MAX_GENOME_BYTES / 2));

    auto genome = GenomeEncoder::encode({CellDescription().setCellFunction(constructedConstructor)});

    DataDescription data;

    data.addCell(CellDescription()
                     .setId(1)
                     .setEnergy(_parameters.cellNormalEnergy * 3)
                     .setMaxConnections(1)
                     .setExecutionOrderNumber(0)
                     .setCellFunction(ConstructorDescription().setGenome(genome)));

    _simController->setSimulationData(data);
    _simController->calcSingleTimestep();
    auto actualData = _simController->getSimulationData();

    EXPECT_EQ(2, actualData.cells.size());
    auto actualConstructedCell = getOtherCell(actualData, 1);

    EXPECT_EQ(Enums::CellFunction_Constructor, actualConstructedCell.getCellFunctionType());

    auto actualConstructor = std::get<ConstructorDescription>(*actualConstructedCell.cellFunction);
    EXPECT_EQ(constructedConstructor, actualConstructor);
}

TEST_F(ConstructorTests, constructConstructorCell_nestingGenomeTooLarge)
{
    auto constructedConstructor = ConstructorDescription()
                           .setMode(Enums::ConstructionMode_Manual)
                           .setSingleConstruction(true)
                           .setSeparateConstruction(false)
                           .setMakeSticky(true)
                           .setAngleAlignment(2)
                           .setGenome(createRandomGenome(MAX_GENOME_BYTES));
    auto genome = GenomeEncoder::encode({CellDescription().setCellFunction(constructedConstructor)});

    DataDescription data;

    data.addCell(CellDescription()
                     .setId(1)
                     .setEnergy(_parameters.cellNormalEnergy * 3)
                     .setMaxConnections(1)
                     .setExecutionOrderNumber(0)
                     .setCellFunction(ConstructorDescription().setGenome(genome)));

    _simController->setSimulationData(data);
    _simController->calcSingleTimestep();
    auto actualData = _simController->getSimulationData();

    EXPECT_EQ(2, actualData.cells.size());
    auto actualCell = getOtherCell(actualData, 1);
    auto actualConstructedCell = getOtherCell(actualData, 1);

    EXPECT_EQ(Enums::CellFunction_Constructor, actualConstructedCell.getCellFunctionType());

    auto actualConstructor = std::get<ConstructorDescription>(*actualCell.cellFunction);
    auto actualConstructedConstructor = std::get<ConstructorDescription>(*actualConstructedCell.cellFunction);
    EXPECT_TRUE(actualConstructor.genome.size() <= MAX_GENOME_BYTES);
    EXPECT_TRUE(constructedConstructor.genome.size() <= MAX_GENOME_BYTES);
}
