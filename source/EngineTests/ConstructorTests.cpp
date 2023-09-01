#include <gtest/gtest.h>

#include "Base/Math.h"
#include "Base/NumberGenerator.h"
#include "EngineInterface/GenomeConstants.h"
#include "EngineInterface/DescriptionHelper.h"
#include "EngineInterface/Descriptions.h"
#include "EngineInterface/GenomeDescriptionConverter.h"
#include "EngineInterface/SimulationController.h"
#include "IntegrationTestFramework.h"

class ConstructorTests : public IntegrationTestFramework
{
public:
    static SimulationParameters getParameters()
    {
        SimulationParameters result;
        result.innerFriction = 0;
        result.baseValues.friction = 0;
        for (int i = 0; i < MAX_COLORS; ++i) {
            result.baseValues.radiationCellAgeStrength[i] = 0;
        }
        return result;
    }

    ConstructorTests()
        : IntegrationTestFramework(getParameters())
    {}

    ~ConstructorTests() = default;

protected:
    bool lowPrecisionCompare(float expected, float actual) const { return approxCompare(expected, actual, 0.01f); }

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
    auto genome = GenomeDescriptionConverter::convertDescriptionToBytes(
        GenomeDescription().setInfo(GenomeHeaderDescription().setSeparateConstruction(false)).setCells({CellGenomeDescription()}));

    DataDescription data;
    data.addCell(
        CellDescription()
            .setId(1)
            .setEnergy(_parameters.cellNormalEnergy[0] * 2 - 1.0f)
            .setMaxConnections(1)
            .setExecutionOrderNumber(0)
                     .setCellFunction(ConstructorDescription().setGenome(genome)));

    _simController->setSimulationData(data);
    _simController->calcSingleTimestep();
    auto actualData = _simController->getSimulationData();

    ASSERT_EQ(1, actualData.cells.size());
    auto actualHostCell = getCell(actualData, 1);

    EXPECT_EQ(0, actualHostCell.connections.size());
    EXPECT_EQ(0, std::get<ConstructorDescription>(*actualHostCell.cellFunction).genomeReadPosition);
    EXPECT_TRUE(approxCompare(_parameters.cellNormalEnergy[0] * 2 - 1.0f, actualHostCell.energy));
    EXPECT_TRUE(approxCompare(0.0f, actualHostCell.activity.channels[0]));
}

TEST_F(ConstructorTests, alreadyFinished)
{
    DataDescription data;

    auto genome = GenomeDescriptionConverter::convertDescriptionToBytes(
        GenomeDescription().setInfo(GenomeHeaderDescription().setSingleConstruction(true)).setCells({CellGenomeDescription()}));

    auto constructor = ConstructorDescription().setGenome(genome);
    constructor.setGenomeReadPosition(constructor.genome.size());

    data.addCell(
        CellDescription()
            .setId(1)
            .setEnergy(_parameters.cellNormalEnergy[0] * 3)
            .setMaxConnections(1)
            .setExecutionOrderNumber(0)
            .setCellFunction(constructor));

    _simController->setSimulationData(data);
    _simController->calcSingleTimestep();
    auto actualData = _simController->getSimulationData();

    ASSERT_EQ(1, actualData.cells.size());
    auto actualHostCell = getCell(actualData, 1);
    auto actualConstructor = std::get<ConstructorDescription>(*actualHostCell.cellFunction);
    EXPECT_EQ(0, actualHostCell.connections.size());
    EXPECT_EQ(actualConstructor.genome.size(), actualConstructor.genomeReadPosition);
    EXPECT_TRUE(approxCompare(0.0f, actualHostCell.activity.channels[0]));
}

TEST_F(ConstructorTests, notActivated)
{
    DataDescription data;

    auto genome = GenomeDescriptionConverter::convertDescriptionToBytes(
        GenomeDescription().setInfo(GenomeHeaderDescription().setSingleConstruction(true)).setCells({CellGenomeDescription()}));
    auto constructor = ConstructorDescription().setGenome(genome);

    data.addCell(CellDescription()
                     .setId(1)
                     .setEnergy(_parameters.cellNormalEnergy[0] * 3)
                     .setMaxConnections(1)
                     .setExecutionOrderNumber(0)
                     .setCellFunction(constructor)
                     .setActivationTime(2));

    _simController->setSimulationData(data);
    _simController->calcSingleTimestep();
    auto actualData = _simController->getSimulationData();

    ASSERT_EQ(1, actualData.cells.size());
    auto actualHostCell = getCell(actualData, 1);
    auto actualConstructor = std::get<ConstructorDescription>(*actualHostCell.cellFunction);
    EXPECT_TRUE(approxCompare(0.0f, actualHostCell.activity.channels[0]));
}

TEST_F(ConstructorTests, manualConstruction_noInputActivity)
{
    auto genome = GenomeDescriptionConverter::convertDescriptionToBytes(GenomeDescription().setCells({CellGenomeDescription()}));

    DataDescription data;
    data.addCell(
        CellDescription()
            .setId(1)
            .setEnergy(_parameters.cellNormalEnergy[0] * 3)
            .setMaxConnections(1)
            .setExecutionOrderNumber(0)
                     .setCellFunction(
                         ConstructorDescription().setActivationMode(0).setGenome(genome)));

    _simController->setSimulationData(data);
    _simController->calcSingleTimestep();
    auto actualData = _simController->getSimulationData();

    ASSERT_EQ(1, actualData.cells.size());
    auto actualHostCell = getCell(actualData, 1);

    EXPECT_EQ(0, actualHostCell.connections.size());
    EXPECT_EQ(0, std::get<ConstructorDescription>(*actualHostCell.cellFunction).genomeReadPosition);
    EXPECT_TRUE(approxCompare(_parameters.cellNormalEnergy[0] * 3, actualHostCell.energy));
    EXPECT_TRUE(approxCompare(0.0f, actualHostCell.activity.channels[0]));
}

TEST_F(ConstructorTests, constructFirstCell_correctCycle)
{
    auto genome = GenomeDescriptionConverter::convertDescriptionToBytes(GenomeDescription().setCells({CellGenomeDescription()}));

    _simController->calcSingleTimestep();

    DataDescription data;
    data.addCell(CellDescription()
                     .setId(1)
                     .setEnergy(_parameters.cellNormalEnergy[0] * 3)
                     .setMaxConnections(1)
                     .setExecutionOrderNumber(0)
                     .setCellFunction(ConstructorDescription().setActivationMode(3).setGenome(genome)));

    _simController->setSimulationData(data);
    for (int i = 0; i < _parameters.cellNumExecutionOrderNumbers * 3; ++i) {
        _simController->calcSingleTimestep();
    }
    auto actualData = _simController->getSimulationData();

    ASSERT_EQ(2, actualData.cells.size());
}

TEST_F(ConstructorTests, constructFirstCell_wrongCycle)
{
    auto genome = GenomeDescriptionConverter::convertDescriptionToBytes(
        GenomeDescription().setCells({CellGenomeDescription()}));

    _simController->calcSingleTimestep();

    DataDescription data;
    data.addCell(CellDescription()
                     .setId(1)
                     .setEnergy(_parameters.cellNormalEnergy[0] * 3)
                     .setMaxConnections(1)
                     .setExecutionOrderNumber(0)
                     .setCellFunction(ConstructorDescription().setActivationMode(3).setGenome(genome)));

    _simController->setSimulationData(data);
    for (int i = 0; i < _parameters.cellNumExecutionOrderNumbers * 3 - 1; ++i) {
        _simController->calcSingleTimestep();
    }
    auto actualData = _simController->getSimulationData();

    ASSERT_EQ(1, actualData.cells.size());
}

TEST_F(ConstructorTests, constructFirstCell_completenessCheck_notReady)
{
    auto constructorGenome = ConstructorGenomeDescription().setMode(0).setConstructionActivationTime(123).setMakeGenomeCopy();
    auto genome =
        GenomeDescriptionConverter::convertDescriptionToBytes(GenomeDescription().setCells({CellGenomeDescription().setCellFunction(constructorGenome)}));
    auto otherGenome = GenomeDescriptionConverter::convertDescriptionToBytes(GenomeDescription().setCells({CellGenomeDescription()}));

    DataDescription data;
    data.addCells({
        CellDescription()
            .setId(1)
            .setPos({10.0f, 10.0f})
            .setEnergy(_parameters.cellNormalEnergy[0] * 3)
            .setMaxConnections(2)
            .setExecutionOrderNumber(0)
            .setCellFunction(ConstructorDescription().setGenome(genome)),
        CellDescription().setId(2).setPos({11.0f, 10.0f}).setEnergy(100).setMaxConnections(2).setExecutionOrderNumber(5).setCellFunction(NerveDescription()),
        CellDescription()
            .setId(3)
            .setPos({12.0f, 10.0f})
            .setEnergy(100)
            .setMaxConnections(1)
            .setExecutionOrderNumber(4)
            .setCellFunction(ConstructorDescription().setGenome(otherGenome).setGenomeReadPosition(Const::GenomeHeaderSize)),
    });
    data.addConnection(1, 2);
    data.addConnection(2, 3);

    _parameters.cellFunctionConstructorCheckCompletenessForSelfReplication = true;
    _simController->setSimulationParameters(_parameters);
    _simController->setSimulationData(data);
    _simController->calcSingleTimestep();
    auto actualData = _simController->getSimulationData();

    ASSERT_EQ(3, actualData.cells.size());
}

TEST_F(ConstructorTests, constructFirstCell_completenessCheck_ready)
{
    auto constructorGenome = ConstructorGenomeDescription().setMode(0).setConstructionActivationTime(123).setMakeGenomeCopy();
    auto genome =
        GenomeDescriptionConverter::convertDescriptionToBytes(GenomeDescription().setCells({CellGenomeDescription().setCellFunction(constructorGenome)}));
    auto otherGenome = GenomeDescriptionConverter::convertDescriptionToBytes(GenomeDescription().setCells({CellGenomeDescription()}));

    DataDescription data;
    data.addCells({
        CellDescription()
            .setId(1)
            .setPos({10.0f, 10.0f})
            .setEnergy(_parameters.cellNormalEnergy[0] * 3)
            .setMaxConnections(2)
            .setExecutionOrderNumber(0)
            .setCellFunction(ConstructorDescription().setGenome(genome)),
        CellDescription().setId(2).setPos({11.0f, 10.0f}).setEnergy(100).setMaxConnections(2).setExecutionOrderNumber(5).setCellFunction(NerveDescription()),
        CellDescription()
            .setId(3)
            .setPos({12.0f, 10.0f})
            .setEnergy(100)
            .setMaxConnections(1)
            .setExecutionOrderNumber(4)
            .setCellFunction(ConstructorDescription().setGenome(otherGenome).setGenomeReadPosition(toInt(otherGenome.size()))),
    });
    data.addConnection(1, 2);
    data.addConnection(2, 3);

    _parameters.cellFunctionConstructorCheckCompletenessForSelfReplication = true;
    _simController->setSimulationParameters(_parameters);
    _simController->setSimulationData(data);
    _simController->calcSingleTimestep();
    auto actualData = _simController->getSimulationData();

    ASSERT_EQ(4, actualData.cells.size());
}

TEST_F(ConstructorTests, constructFirstCell_completenessCheck_largeCluster)
{
    auto constexpr RectLength = 50;
    auto rect = DescriptionHelper::createRect(DescriptionHelper::CreateRectParameters().height(RectLength).width(RectLength));

    auto constructorGenome = ConstructorGenomeDescription().setMode(0).setConstructionActivationTime(123).setMakeGenomeCopy();
    auto genome =
        GenomeDescriptionConverter::convertDescriptionToBytes(GenomeDescription().setCells({CellGenomeDescription().setCellFunction(constructorGenome)}));
    auto otherGenome = GenomeDescriptionConverter::convertDescriptionToBytes(GenomeDescription().setCells({CellGenomeDescription()}));

    auto& cell1 = rect.cells.at(0);
    cell1.setEnergy(_parameters.cellNormalEnergy[0] * 3).setExecutionOrderNumber(0).setCellFunction(ConstructorDescription().setGenome(genome));

    _parameters.cellFunctionConstructorCheckCompletenessForSelfReplication = true;
    _simController->setSimulationParameters(_parameters);
    _simController->setSimulationData(rect);
    _simController->calcSingleTimestep();
    auto actualData = _simController->getSimulationData();

    ASSERT_EQ(RectLength * RectLength + 1, actualData.cells.size());
}

TEST_F(ConstructorTests, constructFirstCell_noSeparation)
{
    auto genome = GenomeDescriptionConverter::convertDescriptionToBytes(
        GenomeDescription()
            .setInfo(GenomeHeaderDescription().setSeparateConstruction(false).setStiffness(0.35f))
            .setCells({CellGenomeDescription().setColor(2).setExecutionOrderNumber(4).setInputExecutionOrderNumber(5).setOutputBlocked(true)}));

    DataDescription data;
    data.addCell(
        CellDescription()
            .setId(1)
            .setPos({10.0f, 10.0f})
            .setEnergy(_parameters.cellNormalEnergy[0] * 3)
            .setMaxConnections(1)
            .setExecutionOrderNumber(0)
            .setCellFunction(ConstructorDescription().setGenome(genome).setConstructionActivationTime(123)));

    _simController->setSimulationData(data);
    _simController->calcSingleTimestep();
    auto actualData = _simController->getSimulationData();

    ASSERT_EQ(2, actualData.cells.size());
    auto actualHostCell = getCell(actualData, 1);
    auto actualConstructedCell = getOtherCell(actualData, 1);

    EXPECT_EQ(1, actualHostCell.connections.size());
    EXPECT_EQ(0, std::get<ConstructorDescription>(*actualHostCell.cellFunction).genomeReadPosition);
    EXPECT_TRUE(approxCompare(_parameters.cellNormalEnergy[0] * 2, actualHostCell.energy));
    EXPECT_TRUE(approxCompare(1.0f, actualHostCell.activity.channels[0]));
    EXPECT_EQ(LivingState_JustReady, actualConstructedCell.livingState);

    EXPECT_EQ(1, actualConstructedCell.connections.size());
    EXPECT_EQ(1, actualConstructedCell.maxConnections);
    EXPECT_EQ(2, actualConstructedCell.color);
    EXPECT_EQ(4, actualConstructedCell.executionOrderNumber);
    EXPECT_EQ(5, actualConstructedCell.inputExecutionOrderNumber);
    EXPECT_TRUE(actualConstructedCell.outputBlocked);
    EXPECT_EQ(CellFunction_None, actualConstructedCell.getCellFunctionType());
    EXPECT_EQ(123, actualConstructedCell.activationTime);
    EXPECT_TRUE(approxCompare(0.35f, actualConstructedCell.stiffness, 0.01f));
    EXPECT_TRUE(approxCompare(_parameters.cellNormalEnergy[0], actualConstructedCell.energy));
    EXPECT_TRUE(approxCompare(1.0f, Math::length(actualHostCell.pos - actualConstructedCell.pos)));
}

TEST_F(ConstructorTests, constructFirstCell_notFinished)
{
    auto genome = GenomeDescriptionConverter::convertDescriptionToBytes(
        GenomeDescription()
            .setInfo(GenomeHeaderDescription().setSeparateConstruction(false)).setCells({CellGenomeDescription(), CellGenomeDescription()}));

    DataDescription data;
    data.addCell(CellDescription()
                     .setId(1)
                     .setPos({10.0f, 10.0f})
                     .setEnergy(_parameters.cellNormalEnergy[0] * 3)
                     .setMaxConnections(1)
                     .setExecutionOrderNumber(0)
                     .setCellFunction(ConstructorDescription().setGenome(genome)));

    _simController->setSimulationData(data);
    _simController->calcSingleTimestep();
    auto actualData = _simController->getSimulationData();

    ASSERT_EQ(2, actualData.cells.size());
    auto actualHostCell = getCell(actualData, 1);
    auto actualConstructedCell = getOtherCell(actualData, 1);

    EXPECT_EQ(1, actualHostCell.connections.size());
    EXPECT_EQ(LivingState_Ready, actualHostCell.livingState);

    EXPECT_EQ(1, actualConstructedCell.connections.size());
    EXPECT_EQ(LivingState_UnderConstruction, actualConstructedCell.livingState);
}

TEST_F(ConstructorTests, constructFirstCell_separation)
{
    auto genome = GenomeDescriptionConverter::convertDescriptionToBytes(
        GenomeDescription().setInfo(GenomeHeaderDescription().setSeparateConstruction(true)).setCells({CellGenomeDescription()}));

    DataDescription data;
    data.addCell(CellDescription()
                     .setId(1)
                     .setEnergy(_parameters.cellNormalEnergy[0] * 3)
                     .setMaxConnections(1)
                     .setExecutionOrderNumber(0)
                     .setCellFunction(ConstructorDescription()
                                          .setGenome(genome)));

    _simController->setSimulationData(data);
    _simController->calcSingleTimestep();
    auto actualData = _simController->getSimulationData();

    ASSERT_EQ(2, actualData.cells.size());
    auto actualHostCell = getCell(actualData, 1);
    auto actualConstructedCell = getOtherCell(actualData, 1);

    EXPECT_EQ(0, actualHostCell.connections.size());
    EXPECT_EQ(1, actualHostCell.maxConnections);

    EXPECT_EQ(0, actualConstructedCell.connections.size());
    EXPECT_EQ(0, actualConstructedCell.maxConnections);
    EXPECT_EQ(LivingState_JustReady, actualConstructedCell.livingState);
}

TEST_F(ConstructorTests, constructFirstCell_singleConstruction)
{
    auto genome = GenomeDescriptionConverter::convertDescriptionToBytes(
        GenomeDescription().setInfo(GenomeHeaderDescription().setSeparateConstruction(true).setSingleConstruction(true)).setCells({CellGenomeDescription()}));

    DataDescription data;
    data.addCell(CellDescription()
                     .setId(1)
                     .setEnergy(_parameters.cellNormalEnergy[0] * 3)
                     .setMaxConnections(1)
                     .setExecutionOrderNumber(0)
                     .setCellFunction(ConstructorDescription().setGenome(genome)));

    _simController->setSimulationData(data);
    _simController->calcSingleTimestep();
    auto actualData = _simController->getSimulationData();

    ASSERT_EQ(2, actualData.cells.size());
    auto actualHostCell = getCell(actualData, 1);
    auto actualConstructedCell = getOtherCell(actualData, 1);

    EXPECT_EQ(0, actualHostCell.connections.size());
    auto const& constructor = std::get<ConstructorDescription>(*actualHostCell.cellFunction);
    EXPECT_EQ(constructor.genome.size(), constructor.genomeReadPosition);

    EXPECT_EQ(0, actualConstructedCell.connections.size());
    EXPECT_EQ(LivingState_JustReady, actualConstructedCell.livingState);
}

TEST_F(ConstructorTests, constructFirstCell_manualConstruction)
{
    auto genome = GenomeDescriptionConverter::convertDescriptionToBytes(
        GenomeDescription().setCells({CellGenomeDescription()}));

    DataDescription data;
    data.addCells({
       CellDescription()
            .setId(1)
            .setPos({10.0f, 10.0f})
            .setEnergy(_parameters.cellNormalEnergy[0] * 3)
            .setMaxConnections(2)
            .setExecutionOrderNumber(0)
            .setInputExecutionOrderNumber(5)
            .setCellFunction(ConstructorDescription().setActivationMode(0).setGenome(genome)),
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

    ASSERT_EQ(3, actualData.cells.size());
    auto actualHostCell = getCell(actualData, 1);
    auto actualConstructedCell = getOtherCell(actualData, {1, 2});

    EXPECT_EQ(1, actualHostCell.connections.size());

    EXPECT_EQ(0, actualConstructedCell.connections.size());
    EXPECT_EQ(LivingState_JustReady, actualConstructedCell.livingState);

    EXPECT_TRUE(approxCompare(10.0f - 1.0f, actualConstructedCell.pos.x));
    EXPECT_TRUE(approxCompare(10.0f, actualConstructedCell.pos.y));
}

TEST_F(ConstructorTests, constructFirstCell_differentAngle1)
{
    auto genome = GenomeDescriptionConverter::convertDescriptionToBytes(GenomeDescription().setCells({CellGenomeDescription()}));

    DataDescription data;
    data.addCells({
        CellDescription()
             .setId(1)
             .setPos({10.0f, 10.0f})
             .setEnergy(_parameters.cellNormalEnergy[0] * 3)
             .setMaxConnections(2)
             .setExecutionOrderNumber(0)
             .setInputExecutionOrderNumber(5)
             .setCellFunction(ConstructorDescription().setActivationMode(0).setGenome(genome).setConstructionAngle2(90.0f)),
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

    ASSERT_EQ(3, actualData.cells.size());
    auto actualHostCell = getCell(actualData, 1);
    auto actualConstructedCell = getOtherCell(actualData, {1, 2});

    EXPECT_TRUE(approxCompare(10.0f, actualConstructedCell.pos.x));
    EXPECT_TRUE(approxCompare(10.0f - 1.0f, actualConstructedCell.pos.y));
}

TEST_F(ConstructorTests, constructFirstCell_differentAngle2)
{
    auto genome = GenomeDescriptionConverter::convertDescriptionToBytes(GenomeDescription().setCells({CellGenomeDescription().setReferenceAngle(-90.0f)}));

    DataDescription data;
    data.addCells(
        {CellDescription()
             .setId(1)
             .setPos({10.0f, 10.0f})
             .setEnergy(_parameters.cellNormalEnergy[0] * 3)
             .setMaxConnections(2)
             .setExecutionOrderNumber(0)
             .setInputExecutionOrderNumber(5)
             .setCellFunction(ConstructorDescription().setActivationMode(0).setGenome(genome).setConstructionAngle2(-90.0f)),
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

    ASSERT_EQ(3, actualData.cells.size());
    auto actualHostCell = getCell(actualData, 1);
    auto actualConstructedCell = getOtherCell(actualData, {1, 2});

    EXPECT_TRUE(approxCompare(10.0f, actualConstructedCell.pos.x));
    EXPECT_TRUE(approxCompare(10.0f + 1.0f, actualConstructedCell.pos.y));
}

TEST_F(ConstructorTests, constructNeuronCell)
{
    auto neuron = NeuronGenomeDescription();
    neuron.weights[1][7] = 3.9f;
    neuron.weights[7][1] = -1.9f;
    neuron.biases[3] = 3.8f;

    auto genome = GenomeDescriptionConverter::convertDescriptionToBytes(GenomeDescription().setCells({CellGenomeDescription().setCellFunction(neuron)}));

    DataDescription data;
    data.addCell(
        CellDescription()
            .setId(1)
            .setEnergy(_parameters.cellNormalEnergy[0] * 3)
            .setMaxConnections(1)
            .setExecutionOrderNumber(0)
                     .setCellFunction(ConstructorDescription().setGenome(genome)));

    _simController->setSimulationData(data);
    _simController->calcSingleTimestep();
    auto actualData = _simController->getSimulationData();

    ASSERT_EQ(2, actualData.cells.size());
    auto actualConstructedCell = getOtherCell(actualData, 1);

    EXPECT_EQ(CellFunction_Neuron, actualConstructedCell.getCellFunctionType());

    auto actualNeuron = std::get<NeuronDescription>(*actualConstructedCell.cellFunction);
    for (int row = 0; row < MAX_CHANNELS; ++row) {
        for (int col = 0; col < MAX_CHANNELS; ++col) {
            EXPECT_TRUE(lowPrecisionCompare(neuron.weights[row][col], actualNeuron.weights[row][col]));
        }
    }
    for (int i = 0; i < MAX_CHANNELS; ++i) {
        EXPECT_TRUE(lowPrecisionCompare(neuron.biases[i], actualNeuron.biases[i]));
    }
}

TEST_F(ConstructorTests, constructConstructorCell)
{
    auto constructorGenome = ConstructorGenomeDescription().setMode(0).setConstructionActivationTime(123).setGenome(createRandomGenome(MAX_GENOME_BYTES / 2));

    auto genome =
        GenomeDescriptionConverter::convertDescriptionToBytes(GenomeDescription().setCells({CellGenomeDescription().setCellFunction(constructorGenome)}));

    DataDescription data;

    data.addCell(CellDescription()
                     .setId(1)
                     .setEnergy(_parameters.cellNormalEnergy[0] * 3)
                     .setMaxConnections(1)
                     .setExecutionOrderNumber(0)
                     .setCellFunction(ConstructorDescription().setGenome(genome)));

    _simController->setSimulationData(data);
    _simController->calcSingleTimestep();
    auto actualData = _simController->getSimulationData();

    ASSERT_EQ(2, actualData.cells.size());
    auto actualConstructedCell = getOtherCell(actualData, 1);

    EXPECT_EQ(CellFunction_Constructor, actualConstructedCell.getCellFunctionType());

    auto actualConstructor = std::get<ConstructorDescription>(*actualConstructedCell.cellFunction);
    EXPECT_EQ(constructorGenome.mode, actualConstructor.activationMode);
    EXPECT_EQ(constructorGenome.constructionActivationTime, actualConstructor.constructionActivationTime);
    EXPECT_EQ(constructorGenome.getGenomeData(), actualConstructor.genome);
}

TEST_F(ConstructorTests, constructNerveCell)
{
    auto nerveDesc = NerveGenomeDescription().setPulseMode(2).setAlternationMode(4);
    auto genome = GenomeDescriptionConverter::convertDescriptionToBytes(GenomeDescription().setCells({CellGenomeDescription().setCellFunction(nerveDesc)}));

    DataDescription data;
    data.addCell(CellDescription()
                     .setId(1)
                     .setEnergy(_parameters.cellNormalEnergy[0] * 3)
                     .setMaxConnections(1)
                     .setExecutionOrderNumber(0)
                     .setCellFunction(ConstructorDescription().setGenome(genome)));

    _simController->setSimulationData(data);
    _simController->calcSingleTimestep();
    auto actualData = _simController->getSimulationData();

    ASSERT_EQ(2, actualData.cells.size());
    auto actualConstructedCell = getOtherCell(actualData, 1);
    auto actualNerve = std::get<NerveDescription>(*actualConstructedCell.cellFunction);

    EXPECT_EQ(CellFunction_Nerve, actualConstructedCell.getCellFunctionType());
    EXPECT_EQ(nerveDesc.pulseMode, actualNerve.pulseMode);
    EXPECT_EQ(nerveDesc.alternationMode, actualNerve.alternationMode);
}

TEST_F(ConstructorTests, constructAttackerCell)
{
    auto attackerDesc = AttackerGenomeDescription().setMode(EnergyDistributionMode_TransmittersAndConstructors);
    auto genome = GenomeDescriptionConverter::convertDescriptionToBytes(GenomeDescription().setCells({CellGenomeDescription().setCellFunction(attackerDesc)}));

    DataDescription data;
    data.addCell(CellDescription()
                     .setId(1)
                     .setEnergy(_parameters.cellNormalEnergy[0] * 3)
                     .setMaxConnections(1)
                     .setExecutionOrderNumber(0)
                     .setCellFunction(ConstructorDescription().setGenome(genome)));

    _simController->setSimulationData(data);
    _simController->calcSingleTimestep();
    auto actualData = _simController->getSimulationData();

    ASSERT_EQ(2, actualData.cells.size());
    auto actualConstructedCell = getOtherCell(actualData, 1);

    EXPECT_EQ(CellFunction_Attacker, actualConstructedCell.getCellFunctionType());

    auto actualAttacker = std::get<AttackerDescription>(*actualConstructedCell.cellFunction);
    EXPECT_EQ(attackerDesc.mode, actualAttacker.mode);
}

TEST_F(ConstructorTests, constructDefenderCell)
{
    auto defenderDesc = DefenderGenomeDescription().setMode(DefenderMode_DefendAgainstInjector);
    auto genome = GenomeDescriptionConverter::convertDescriptionToBytes(GenomeDescription().setCells({CellGenomeDescription().setCellFunction(defenderDesc)}));

    DataDescription data;
    data.addCell(CellDescription()
                     .setId(1)
                     .setEnergy(_parameters.cellNormalEnergy[0] * 3)
                     .setMaxConnections(1)
                     .setExecutionOrderNumber(0)
                     .setCellFunction(ConstructorDescription().setGenome(genome)));

    _simController->setSimulationData(data);
    _simController->calcSingleTimestep();
    auto actualData = _simController->getSimulationData();

    ASSERT_EQ(2, actualData.cells.size());
    auto actualConstructedCell = getOtherCell(actualData, 1);

    EXPECT_EQ(CellFunction_Defender, actualConstructedCell.getCellFunctionType());

    auto actualDefender = std::get<DefenderDescription>(*actualConstructedCell.cellFunction);
    EXPECT_EQ(defenderDesc.mode, actualDefender.mode);
}

TEST_F(ConstructorTests, constructTransmitterCell)
{
    auto transmitterDesc = TransmitterGenomeDescription().setMode(EnergyDistributionMode_TransmittersAndConstructors);
    auto genome =
        GenomeDescriptionConverter::convertDescriptionToBytes(GenomeDescription().setCells({CellGenomeDescription().setCellFunction(transmitterDesc)}));

    DataDescription data;
    data.addCell(CellDescription()
                     .setId(1)
                     .setEnergy(_parameters.cellNormalEnergy[0] * 3)
                     .setMaxConnections(1)
                     .setExecutionOrderNumber(0)
                     .setCellFunction(ConstructorDescription().setGenome(genome)));

    _simController->setSimulationData(data);
    _simController->calcSingleTimestep();
    auto actualData = _simController->getSimulationData();

    ASSERT_EQ(2, actualData.cells.size());
    auto actualConstructedCell = getOtherCell(actualData, 1);

    EXPECT_EQ(CellFunction_Transmitter, actualConstructedCell.getCellFunctionType());

    auto actualTransmitter = std::get<TransmitterDescription>(*actualConstructedCell.cellFunction);
    EXPECT_EQ(transmitterDesc.mode, actualTransmitter.mode);
}

TEST_F(ConstructorTests, constructMuscleCell)
{
    auto muscleDesc = MuscleGenomeDescription().setMode(MuscleMode_ContractionExpansion);
    auto genome = GenomeDescriptionConverter::convertDescriptionToBytes(GenomeDescription().setCells({CellGenomeDescription().setCellFunction(muscleDesc)}));

    DataDescription data;
    data.addCell(CellDescription()
                     .setId(1)
                     .setEnergy(_parameters.cellNormalEnergy[0] * 3)
                     .setMaxConnections(1)
                     .setExecutionOrderNumber(0)
                     .setCellFunction(ConstructorDescription().setGenome(genome)));

    _simController->setSimulationData(data);
    _simController->calcSingleTimestep();
    auto actualData = _simController->getSimulationData();

    ASSERT_EQ(2, actualData.cells.size());
    auto actualConstructedCell = getOtherCell(actualData, 1);

    EXPECT_EQ(CellFunction_Muscle, actualConstructedCell.getCellFunctionType());

    auto actualMuscle = std::get<MuscleDescription>(*actualConstructedCell.cellFunction);
    EXPECT_EQ(muscleDesc.mode, actualMuscle.mode);
    EXPECT_EQ(0, actualMuscle.lastBendingDirection);
    EXPECT_EQ(0.0f, actualMuscle.consecutiveBendingAngle);
}

TEST_F(ConstructorTests, constructSensorCell)
{
    auto sensorDesc = SensorGenomeDescription().setFixedAngle(90.0f).setColor(2).setMinDensity(0.5f);
    auto genome = GenomeDescriptionConverter::convertDescriptionToBytes(GenomeDescription().setCells({CellGenomeDescription().setCellFunction(sensorDesc)}));

    DataDescription data;
    data.addCell(CellDescription()
                     .setId(1)
                     .setEnergy(_parameters.cellNormalEnergy[0] * 3)
                     .setMaxConnections(1)
                     .setExecutionOrderNumber(0)
                     .setCellFunction(ConstructorDescription().setGenome(genome)));

    _simController->setSimulationData(data);
    _simController->calcSingleTimestep();
    auto actualData = _simController->getSimulationData();

    ASSERT_EQ(2, actualData.cells.size());
    auto actualConstructedCell = getOtherCell(actualData, 1);

    EXPECT_EQ(CellFunction_Sensor, actualConstructedCell.getCellFunctionType());

    auto actualSensor = std::get<SensorDescription>(*actualConstructedCell.cellFunction);
    EXPECT_EQ(sensorDesc.fixedAngle.has_value(), actualSensor.fixedAngle.has_value());
    EXPECT_TRUE(lowPrecisionCompare(*sensorDesc.fixedAngle, *actualSensor.fixedAngle));
    EXPECT_TRUE(lowPrecisionCompare(sensorDesc.minDensity, actualSensor.minDensity));
    EXPECT_EQ(sensorDesc.color, actualSensor.color);
}

TEST_F(ConstructorTests, constructInjectorCell)
{
    auto injectorDesc = InjectorGenomeDescription().setMode(InjectorMode_InjectOnlyEmptyCells).setGenome(createRandomGenome(MAX_GENOME_BYTES / 2));
    auto genome = GenomeDescriptionConverter::convertDescriptionToBytes(GenomeDescription().setCells({CellGenomeDescription().setCellFunction(injectorDesc)}));

    DataDescription data;
    data.addCell(CellDescription()
                     .setId(1)
                     .setEnergy(_parameters.cellNormalEnergy[0] * 3)
                     .setMaxConnections(1)
                     .setExecutionOrderNumber(0)
                     .setCellFunction(ConstructorDescription().setGenome(genome)));

    _simController->setSimulationData(data);
    _simController->calcSingleTimestep();
    auto actualData = _simController->getSimulationData();

    ASSERT_EQ(2, actualData.cells.size());
    auto actualConstructedCell = getOtherCell(actualData, 1);

    EXPECT_EQ(CellFunction_Injector, actualConstructedCell.getCellFunctionType());

    auto actualInjector = std::get<InjectorDescription>(*actualConstructedCell.cellFunction);
    EXPECT_EQ(injectorDesc.mode, actualInjector.mode);
    EXPECT_EQ(injectorDesc.getGenomeData(), actualInjector.genome);
}

TEST_F(ConstructorTests, constructConstructorCell_nestingGenomeTooLarge)
{
    auto constructedConstructor = ConstructorGenomeDescription().setMode(0).setGenome(createRandomGenome(MAX_GENOME_BYTES));
    auto genome =
        GenomeDescriptionConverter::convertDescriptionToBytes(GenomeDescription().setCells({CellGenomeDescription().setCellFunction(constructedConstructor)}));


    DataDescription data;

    data.addCell(CellDescription()
                     .setId(1)
                     .setEnergy(_parameters.cellNormalEnergy[0] * 3)
                     .setMaxConnections(1)
                     .setExecutionOrderNumber(0)
                     .setCellFunction(ConstructorDescription().setGenome(genome)));

    _simController->setSimulationData(data);
    _simController->calcSingleTimestep();
    auto actualData = _simController->getSimulationData();

    ASSERT_EQ(2, actualData.cells.size());
    auto actualCell = getOtherCell(actualData, 1);
    auto actualConstructedCell = getOtherCell(actualData, 1);

    EXPECT_EQ(CellFunction_Constructor, actualConstructedCell.getCellFunctionType());

    auto actualConstructor = std::get<ConstructorDescription>(*actualCell.cellFunction);
    auto actualConstructedConstructor = std::get<ConstructorDescription>(*actualConstructedCell.cellFunction);
    EXPECT_TRUE(actualConstructor.genome.size() <= MAX_GENOME_BYTES);
    EXPECT_TRUE(constructedConstructor.getGenomeData().size() <= MAX_GENOME_BYTES);
}

TEST_F(ConstructorTests, constructConstructorCell_copyGenome)
{
    auto constructedConstructor = ConstructorGenomeDescription().setMode(0).setMakeGenomeCopy();

    auto genome =
        GenomeDescriptionConverter::convertDescriptionToBytes(GenomeDescription().setCells({CellGenomeDescription().setCellFunction(constructedConstructor)}));

    DataDescription data;
    data.addCell(CellDescription()
                     .setId(1)
                     .setEnergy(_parameters.cellNormalEnergy[0] * 3)
                     .setMaxConnections(1)
                     .setExecutionOrderNumber(0)
                     .setCellFunction(ConstructorDescription().setGenome(genome)));

    _simController->setSimulationData(data);
    _simController->calcSingleTimestep();
    auto actualData = _simController->getSimulationData();

    ASSERT_EQ(2, actualData.cells.size());
    auto actualConstructedCell = getOtherCell(actualData, 1);

    EXPECT_EQ(CellFunction_Constructor, actualConstructedCell.getCellFunctionType());

    auto actualConstructor = std::get<ConstructorDescription>(*actualConstructedCell.cellFunction);
    EXPECT_EQ(genome, actualConstructor.genome);
}

TEST_F(ConstructorTests, constructSecondCell_separation)
{
    auto genome = GenomeDescriptionConverter::convertDescriptionToBytes(
        GenomeDescription().setInfo(GenomeHeaderDescription().setSeparateConstruction(true)).setCells({CellGenomeDescription()}));

    DataDescription data;
    data.addCells({
        CellDescription()
            .setId(1)
            .setPos({10.0f, 10.0f})
            .setEnergy(_parameters.cellNormalEnergy[0] * 3)
            .setMaxConnections(1)
            .setExecutionOrderNumber(0)
            .setCellFunction(ConstructorDescription().setGenome(genome)),
        CellDescription()
            .setId(2)
            .setPos({10.0f - _parameters.cellFunctionConstructorOffspringDistance[0], 10.0f})
            .setEnergy(100)
            .setMaxConnections(1)
            .setExecutionOrderNumber(5)
            .setCellFunction(NerveDescription())
            .setLivingState(LivingState_UnderConstruction),
    });
    data.addConnection(1, 2);

    _simController->setSimulationData(data);
    _simController->calcSingleTimestep();
    auto actualData = _simController->getSimulationData();

    ASSERT_EQ(3, actualData.cells.size());
    auto actualHostCell = getCell(actualData, 1);
    auto actualPrevConstructedCell = getCell(actualData, 2);
    auto actualConstructedCell = getOtherCell(actualData, {1, 2});

    EXPECT_EQ(0, actualHostCell.connections.size());
    EXPECT_EQ(LivingState_Ready, actualHostCell.livingState);

    ASSERT_EQ(1, actualPrevConstructedCell.connections.size());
    EXPECT_EQ(LivingState_UnderConstruction, actualPrevConstructedCell.livingState);
    EXPECT_TRUE(lowPrecisionCompare(1.0f, actualPrevConstructedCell.connections[0].distance));

    ASSERT_EQ(1, actualConstructedCell.connections.size());
    EXPECT_EQ(LivingState_JustReady, actualConstructedCell.livingState);
    EXPECT_TRUE(lowPrecisionCompare(1.0f, actualConstructedCell.connections[0].distance));
}

TEST_F(ConstructorTests, constructSecondCell_constructionStateTransitions)
{
    auto genome = GenomeDescriptionConverter::convertDescriptionToBytes(
        GenomeDescription().setInfo(GenomeHeaderDescription().setSeparateConstruction(true)).setCells({CellGenomeDescription()}));

    DataDescription data;
    data.addCells({
        CellDescription()
            .setId(1)
            .setPos({10.0f, 10.0f})
            .setEnergy(_parameters.cellNormalEnergy[0] * 3)
            .setMaxConnections(1)
            .setExecutionOrderNumber(0)
            .setCellFunction(ConstructorDescription().setGenome(genome)),
        CellDescription()
            .setId(2)
            .setPos({10.0f - _parameters.cellFunctionConstructorOffspringDistance[0], 10.0f})
            .setEnergy(100)
            .setMaxConnections(1)
            .setExecutionOrderNumber(5)
            .setCellFunction(NerveDescription())
            .setLivingState(LivingState_UnderConstruction),
    });
    data.addConnection(1, 2);

    _simController->setSimulationData(data);
    _simController->calcSingleTimestep();
    _simController->calcSingleTimestep();
    {
        auto actualData = _simController->getSimulationData();

        ASSERT_EQ(3, actualData.cells.size());
        auto actualHostCell = getCell(actualData, 1);
        auto actualPrevConstructedCell = getCell(actualData, 2);
        auto actualConstructedCell = getOtherCell(actualData, {1, 2});

        EXPECT_EQ(LivingState_Ready, actualHostCell.livingState);
        EXPECT_EQ(LivingState_JustReady, actualPrevConstructedCell.livingState);
        EXPECT_EQ(LivingState_Ready, actualConstructedCell.livingState);
    }
    _simController->calcSingleTimestep();
    {
        auto actualData = _simController->getSimulationData();

        ASSERT_EQ(3, actualData.cells.size());
        auto actualHostCell = getCell(actualData, 1);
        auto actualPrevConstructedCell = getCell(actualData, 2);
        auto actualConstructedCell = getOtherCell(actualData, {1, 2});

        EXPECT_EQ(LivingState_Ready, actualHostCell.livingState);
        EXPECT_EQ(LivingState_Ready, actualPrevConstructedCell.livingState);
        EXPECT_EQ(LivingState_Ready, actualConstructedCell.livingState);
    }
}

TEST_F(ConstructorTests, constructSecondCell_noSeparation)
{
    auto genome = GenomeDescriptionConverter::convertDescriptionToBytes(
        GenomeDescription().setInfo(GenomeHeaderDescription().setSeparateConstruction(false)).setCells({CellGenomeDescription()}));

    DataDescription data;
    data.addCells({
        CellDescription()
            .setId(1)
            .setPos({10.0f, 10.0f})
            .setEnergy(_parameters.cellNormalEnergy[0] * 3)
            .setMaxConnections(1)
            .setExecutionOrderNumber(0)
            .setCellFunction(ConstructorDescription().setGenome(genome)),
        CellDescription()
            .setId(2)
            .setPos({10.0f - _parameters.cellFunctionConstructorOffspringDistance[0], 10.0f})
            .setEnergy(100)
            .setMaxConnections(1)
            .setExecutionOrderNumber(5)
            .setCellFunction(NerveDescription())
            .setLivingState(LivingState_UnderConstruction),
    });
    data.addConnection(1, 2);

    _simController->setSimulationData(data);
    _simController->calcSingleTimestep();
    auto actualData = _simController->getSimulationData();

    ASSERT_EQ(3, actualData.cells.size());
    auto actualHostCell = getCell(actualData, 1);
    auto actualPrevConstructedCell = getCell(actualData, 2);
    auto actualConstructedCell = getOtherCell(actualData, {1, 2});

    EXPECT_EQ(1, actualHostCell.connections.size());
    EXPECT_EQ(LivingState_Ready, actualHostCell.livingState);
    EXPECT_TRUE(lowPrecisionCompare(1.0f, actualPrevConstructedCell.connections[0].distance));

    ASSERT_EQ(2, actualConstructedCell.connections.size());
    EXPECT_EQ(LivingState_JustReady, actualConstructedCell.livingState);
    std::map<uint64_t, ConnectionDescription> connectionById;
    for (auto const& connection : actualConstructedCell.connections) {
        connectionById.emplace(connection.cellId, connection);
    }
    EXPECT_TRUE(lowPrecisionCompare(1.0f, connectionById.at(1).distance));
    EXPECT_TRUE(approxCompare(180.0f, connectionById.at(1).angleFromPrevious));
    EXPECT_TRUE(lowPrecisionCompare(1.0f, connectionById.at(2).distance));
    EXPECT_TRUE(approxCompare(180.0f, connectionById.at(2).angleFromPrevious));

    ASSERT_EQ(1, actualPrevConstructedCell.connections.size());
    EXPECT_EQ(LivingState_UnderConstruction, actualPrevConstructedCell.livingState);
    EXPECT_TRUE(lowPrecisionCompare(1.0f, actualPrevConstructedCell.connections[0].distance));
}

TEST_F(ConstructorTests, constructSecondCell_noSpace)
{
    auto genome = GenomeDescriptionConverter::convertDescriptionToBytes(
        GenomeDescription().setInfo(GenomeHeaderDescription().setSeparateConstruction(false)).setCells({CellGenomeDescription()}));

    DataDescription data;
    data.addCells({
        CellDescription()
            .setId(1)
            .setPos({10.0f, 10.0f})
            .setEnergy(_parameters.cellNormalEnergy[0] * 3)
            .setMaxConnections(1)
            .setExecutionOrderNumber(0)
            .setCellFunction(ConstructorDescription().setGenome(genome)),
        CellDescription()
            .setId(2)
            .setPos({10.0f - 1.0f - _parameters.cellMinDistance/2, 10.0f})
            .setEnergy(100)
            .setMaxConnections(1)
            .setExecutionOrderNumber(5)
            .setCellFunction(NerveDescription())
            .setLivingState(LivingState_UnderConstruction),
    });
    data.addConnection(1, 2);

    _simController->setSimulationData(data);
    _simController->calcSingleTimestep();
    auto actualData = _simController->getSimulationData();

    ASSERT_EQ(2, actualData.cells.size());
    auto actualHostCell = getCell(actualData, 1);
    auto actualPrevConstructedCell = getCell(actualData, 2);

    EXPECT_EQ(1, actualHostCell.connections.size());
    EXPECT_TRUE(approxCompare(0.0f, actualHostCell.activity.channels[0]));
    ASSERT_EQ(1, actualPrevConstructedCell.connections.size());
    auto actualConstructor = std::get<ConstructorDescription>(*actualHostCell.cellFunction);
    EXPECT_EQ(0, actualConstructor.genomeReadPosition);
}

TEST_F(ConstructorTests, constructSecondCell_notFinished)
{
    auto genome = GenomeDescriptionConverter::convertDescriptionToBytes(
        GenomeDescription().setInfo(GenomeHeaderDescription().setSeparateConstruction(false)).setCells({CellGenomeDescription(), CellGenomeDescription()}));

    DataDescription data;
    data.addCells({
        CellDescription()
            .setId(1)
            .setPos({10.0f, 10.0f})
            .setEnergy(_parameters.cellNormalEnergy[0] * 3)
            .setMaxConnections(1)
            .setExecutionOrderNumber(0)
            .setCellFunction(ConstructorDescription().setGenome(genome)),
        CellDescription()
            .setId(2)
            .setPos({10.0f - _parameters.cellFunctionConstructorOffspringDistance[0], 10.0f})
            .setEnergy(100)
            .setMaxConnections(1)
            .setExecutionOrderNumber(5)
            .setCellFunction(NerveDescription())
            .setLivingState(LivingState_UnderConstruction),
    });
    data.addConnection(1, 2);

    _simController->setSimulationData(data);
    _simController->calcSingleTimestep();
    auto actualData = _simController->getSimulationData();

    ASSERT_EQ(3, actualData.cells.size());
    auto actualHostCell = getCell(actualData, 1);
    auto actualPrevConstructedCell = getCell(actualData, 2);
    auto actualConstructedCell = getOtherCell(actualData, {1, 2});

    EXPECT_EQ(1, actualHostCell.connections.size());
    EXPECT_EQ(LivingState_Ready, actualHostCell.livingState);

    ASSERT_EQ(2, actualConstructedCell.connections.size());
    EXPECT_EQ(LivingState_UnderConstruction, actualConstructedCell.livingState);

    ASSERT_EQ(1, actualPrevConstructedCell.connections.size());
    EXPECT_EQ(LivingState_UnderConstruction, actualPrevConstructedCell.livingState);
}

TEST_F(ConstructorTests, constructSecondCell_differentAngle1)
{
    auto genome = GenomeDescriptionConverter::convertDescriptionToBytes(
        GenomeDescription().setInfo(GenomeHeaderDescription().setSeparateConstruction(false)).setCells({CellGenomeDescription()}));

    DataDescription data;
    data.addCells({
        CellDescription()
            .setId(1)
            .setPos({10.0f, 10.0f})
            .setEnergy(_parameters.cellNormalEnergy[0] * 3)
            .setMaxConnections(1)
            .setExecutionOrderNumber(0)
            .setCellFunction(ConstructorDescription().setGenome(genome).setConstructionAngle2(90.0f)),
        CellDescription()
            .setId(2)
            .setPos({10.0f - _parameters.cellFunctionConstructorOffspringDistance[0], 10.0f})
            .setEnergy(100)
            .setMaxConnections(1)
            .setExecutionOrderNumber(5)
            .setCellFunction(NerveDescription())
            .setLivingState(LivingState_UnderConstruction),
    });
    data.addConnection(1, 2);

    _simController->setSimulationData(data);
    _simController->calcSingleTimestep();
    auto actualData = _simController->getSimulationData();

    ASSERT_EQ(3, actualData.cells.size());
    auto actualHostCell = getCell(actualData, 1);
    auto actualPrevConstructedCell = getCell(actualData, 2);
    auto actualConstructedCell = getOtherCell(actualData, {1, 2});

    EXPECT_EQ(1, actualHostCell.connections.size());

    ASSERT_EQ(2, actualConstructedCell.connections.size());
    std::map<uint64_t, ConnectionDescription> connectionById;
    for (auto const& connection : actualConstructedCell.connections) {
        connectionById.emplace(connection.cellId, connection);
    }
    EXPECT_TRUE(lowPrecisionCompare(1.0f, connectionById.at(1).distance));
    EXPECT_TRUE(lowPrecisionCompare(270.0f, connectionById.at(1).angleFromPrevious));
    EXPECT_TRUE(lowPrecisionCompare(1.0f, connectionById.at(2).distance));
    EXPECT_TRUE(lowPrecisionCompare(90.0f, connectionById.at(2).angleFromPrevious));

    ASSERT_EQ(1, actualPrevConstructedCell.connections.size());
}

TEST_F(ConstructorTests, constructSecondCell_differentAngle2)
{
    auto genome = GenomeDescriptionConverter::convertDescriptionToBytes(
        GenomeDescription().setInfo(GenomeHeaderDescription().setSeparateConstruction(false)).setCells({CellGenomeDescription()}));

    DataDescription data;
    data.addCells({
        CellDescription()
            .setId(1)
            .setPos({10.0f, 10.0f})
            .setEnergy(_parameters.cellNormalEnergy[0] * 3)
            .setMaxConnections(1)
            .setExecutionOrderNumber(0)
            .setCellFunction(ConstructorDescription().setGenome(genome).setConstructionAngle2(-90.0f)),
        CellDescription()
            .setId(2)
            .setPos({10.0f - _parameters.cellFunctionConstructorOffspringDistance[0], 10.0f})
            .setEnergy(100)
            .setMaxConnections(1)
            .setExecutionOrderNumber(5)
            .setCellFunction(NerveDescription())
            .setLivingState(LivingState_UnderConstruction),
    });
    data.addConnection(1, 2);

    _simController->setSimulationData(data);
    _simController->calcSingleTimestep();
    auto actualData = _simController->getSimulationData();

    ASSERT_EQ(3, actualData.cells.size());
    auto actualHostCell = getCell(actualData, 1);
    auto actualPrevConstructedCell = getCell(actualData, 2);
    auto actualConstructedCell = getOtherCell(actualData, {1, 2});

    EXPECT_EQ(1, actualHostCell.connections.size());

    ASSERT_EQ(2, actualConstructedCell.connections.size());
    std::map<uint64_t, ConnectionDescription> connectionById;
    for (auto const& connection : actualConstructedCell.connections) {
        connectionById.emplace(connection.cellId, connection);
    }
    EXPECT_TRUE(lowPrecisionCompare(1.0f, connectionById.at(1).distance));
    EXPECT_TRUE(lowPrecisionCompare(90.0f, connectionById.at(1).angleFromPrevious));
    EXPECT_TRUE(lowPrecisionCompare(1.0f, connectionById.at(2).distance));
    EXPECT_TRUE(lowPrecisionCompare(270.0f, connectionById.at(2).angleFromPrevious));

    ASSERT_EQ(1, actualPrevConstructedCell.connections.size());
}

TEST_F(ConstructorTests, constructThirdCell_multipleConnections_upperPart)
{
    auto genome = GenomeDescriptionConverter::convertDescriptionToBytes(
        GenomeDescription().setInfo(GenomeHeaderDescription().setSeparateConstruction(false)).setCells({CellGenomeDescription()}));

    DataDescription data;
    data.addCells({
        CellDescription()
            .setId(1)
            .setPos({10.0f, 10.0f})
            .setEnergy(_parameters.cellNormalEnergy[0] * 3)
            .setMaxConnections(2)
            .setExecutionOrderNumber(0)
            .setCellFunction(ConstructorDescription().setGenome(genome)),
        CellDescription()
            .setId(2)
            .setPos({10.0f - _parameters.cellFunctionConstructorOffspringDistance[0], 10.0f})
            .setEnergy(100)
            .setMaxConnections(2)
            .setExecutionOrderNumber(5)
            .setCellFunction(NerveDescription())
            .setLivingState(LivingState_UnderConstruction),
        CellDescription()
            .setId(3)
            .setPos({10.0f - _parameters.cellFunctionConstructorOffspringDistance[0], 9.0f})
            .setEnergy(100)
            .setMaxConnections(2)
            .setExecutionOrderNumber(5)
            .setCellFunction(NerveDescription())
            .setLivingState(LivingState_UnderConstruction),
        CellDescription().setId(4).setPos({10.0f, 9.5f}).setEnergy(_parameters.cellNormalEnergy[0] * 3).setMaxConnections(2).setExecutionOrderNumber(0),
        CellDescription().setId(5).setPos({10.0f, 9.0f}).setEnergy(_parameters.cellNormalEnergy[0] * 3).setMaxConnections(2).setExecutionOrderNumber(0),
    });
    data.addConnection(1, 2);
    data.addConnection(2, 3);
    data.addConnection(1, 4);
    data.addConnection(4, 5);

    _simController->setSimulationData(data);
    _simController->calcSingleTimestep();
    auto actualData = _simController->getSimulationData();

    EXPECT_EQ(6, actualData.cells.size());
    auto actualHostCell = getCell(actualData, 1);
    auto uninvolvedCell1 = getCell(actualData, 4);
    auto uninvolvedCell2 = getCell(actualData, 5);
    auto actualPrevConstructedCell = getCell(actualData, 2);
    auto actualPrevPrevConstructedCell = getCell(actualData, 3);
    auto actualConstructedCell = getOtherCell(actualData, {1, 2, 3, 4, 5});

    EXPECT_EQ(2, uninvolvedCell1.connections.size());
    EXPECT_EQ(1, uninvolvedCell2.connections.size());
    EXPECT_EQ(2, actualHostCell.connections.size());
    ASSERT_EQ(3, actualConstructedCell.connections.size());
    ASSERT_EQ(2, actualPrevPrevConstructedCell.connections.size());
    ASSERT_EQ(2, actualPrevConstructedCell.connections.size());
}

TEST_F(ConstructorTests, constructThirdCell_multipleConnections_bottomPart)
{
    auto genome = GenomeDescriptionConverter::convertDescriptionToBytes(
        GenomeDescription().setInfo(GenomeHeaderDescription().setSeparateConstruction(false)).setCells({CellGenomeDescription()}));

    DataDescription data;
    data.addCells({
        CellDescription()
            .setId(1)
            .setPos({10.0f, 10.0f})
            .setEnergy(_parameters.cellNormalEnergy[0] * 3)
            .setMaxConnections(2)
            .setExecutionOrderNumber(0)
            .setCellFunction(ConstructorDescription().setGenome(genome)),
        CellDescription()
            .setId(2)
            .setPos({10.0f - _parameters.cellFunctionConstructorOffspringDistance[0], 10.0f})
            .setEnergy(100)
            .setMaxConnections(2)
            .setExecutionOrderNumber(5)
            .setCellFunction(NerveDescription())
            .setLivingState(LivingState_UnderConstruction),
        CellDescription()
            .setId(3)
            .setPos({10.0f - _parameters.cellFunctionConstructorOffspringDistance[0], 11.0f})
            .setEnergy(100)
            .setMaxConnections(2)
            .setExecutionOrderNumber(5)
            .setCellFunction(NerveDescription())
            .setLivingState(LivingState_UnderConstruction),
        CellDescription().setId(4).setPos({10.0f, 10.5f}).setEnergy(_parameters.cellNormalEnergy[0] * 3).setMaxConnections(2).setExecutionOrderNumber(0),
        CellDescription().setId(5).setPos({10.0f, 11.0f}).setEnergy(_parameters.cellNormalEnergy[0] * 3).setMaxConnections(2).setExecutionOrderNumber(0),
    });
    data.addConnection(1, 2);
    data.addConnection(2, 3);
    data.addConnection(1, 4);
    data.addConnection(4, 5);

    _simController->setSimulationData(data);
    _simController->calcSingleTimestep();
    auto actualData = _simController->getSimulationData();

    EXPECT_EQ(6, actualData.cells.size());
    auto actualHostCell = getCell(actualData, 1);
    auto uninvolvedCell1 = getCell(actualData, 4);
    auto uninvolvedCell2 = getCell(actualData, 5);
    auto actualPrevConstructedCell = getCell(actualData, 2);
    auto actualPrevPrevConstructedCell = getCell(actualData, 3);
    auto actualConstructedCell = getOtherCell(actualData, {1, 2, 3, 4, 5});

    EXPECT_EQ(2, uninvolvedCell1.connections.size());
    EXPECT_EQ(1, uninvolvedCell2.connections.size());
    EXPECT_EQ(2, actualHostCell.connections.size());
    ASSERT_EQ(3, actualConstructedCell.connections.size());
    ASSERT_EQ(2, actualPrevPrevConstructedCell.connections.size());
    ASSERT_EQ(2, actualPrevConstructedCell.connections.size());
}

TEST_F(ConstructorTests, constructSecondCell_noSeparation_singleConstruction)
{
    auto genome = GenomeDescriptionConverter::convertDescriptionToBytes(
        GenomeDescription().setInfo(GenomeHeaderDescription().setSeparateConstruction(false).setSingleConstruction(true)).setCells({CellGenomeDescription()}));

    DataDescription data;
    data.addCells({
        CellDescription()
            .setId(1)
            .setPos({10.0f, 10.0f})
            .setEnergy(_parameters.cellNormalEnergy[0] * 3)
            .setMaxConnections(1)
            .setExecutionOrderNumber(0)
            .setCellFunction(ConstructorDescription().setGenome(genome)),
        CellDescription()
            .setId(2)
            .setPos({10.0f - _parameters.cellFunctionConstructorOffspringDistance[0], 10.0f})
            .setEnergy(100)
            .setMaxConnections(1)
            .setExecutionOrderNumber(5)
            .setCellFunction(NerveDescription())
            .setLivingState(LivingState_UnderConstruction),
    });
    data.addConnection(1, 2);

    _simController->setSimulationData(data);
    _simController->calcSingleTimestep();
    auto actualData = _simController->getSimulationData();
    auto actualHostCell = getCell(actualData, 1);

    EXPECT_TRUE(lowPrecisionCompare(1.0f, actualHostCell.connections[0].distance));
}

TEST_F(ConstructorTests, constructFourthCell_noOverlappingConnection)
{
    auto genome = GenomeDescriptionConverter::convertDescriptionToBytes(
        GenomeDescription().setInfo(GenomeHeaderDescription().setSeparateConstruction(false)).setCells({CellGenomeDescription()}));

    DataDescription data;
    data.addCells({
        CellDescription()
            .setId(1)
            .setPos({10.0f, 10.0f})
            .setEnergy(_parameters.cellNormalEnergy[0] * 3)
            .setMaxConnections(2)
            .setExecutionOrderNumber(0)
            .setCellFunction(ConstructorDescription().setGenome(genome)),
        CellDescription()
            .setId(2)
            .setPos({10.0f - _parameters.cellFunctionConstructorOffspringDistance[0], 10.0f})
            .setEnergy(100)
            .setMaxConnections(3)
            .setExecutionOrderNumber(5)
            .setCellFunction(NerveDescription())
            .setLivingState(LivingState_UnderConstruction),
        CellDescription()
            .setId(3)
            .setPos({10.0f - _parameters.cellFunctionConstructorOffspringDistance[0], 11.0f})
            .setEnergy(100)
            .setMaxConnections(2)
            .setExecutionOrderNumber(5)
            .setCellFunction(NerveDescription())
            .setLivingState(LivingState_UnderConstruction),
        CellDescription()
            .setId(4)
            .setPos({10.0f - _parameters.cellFunctionConstructorOffspringDistance[0] + 1.0f, 11.0f})
            .setMaxConnections(2)
            .setExecutionOrderNumber(0)
            .setLivingState(LivingState_UnderConstruction),
    });
    data.addConnection(1, 2);
    data.addConnection(2, 3);
    data.addConnection(3, 4);
    data.addConnection(4, 2);

    _simController->setSimulationData(data);
    _simController->calcSingleTimestep();
    auto actualData = _simController->getSimulationData();

    EXPECT_EQ(5, actualData.cells.size());
    auto actualHostCell = getCell(actualData, 1);
    auto actualPrevConstructedCell = getCell(actualData, 2);
    auto actualPrevPrevConstructedCell = getCell(actualData, 3);
    auto actualPrevPrevPrevConstructedCell = getCell(actualData, 4);
    auto actualConstructedCell = getOtherCell(actualData, {1, 2, 3, 4});

    EXPECT_EQ(1, actualHostCell.connections.size());
    ASSERT_EQ(3, actualConstructedCell.connections.size());
    ASSERT_EQ(3, actualConstructedCell.maxConnections);
    ASSERT_EQ(3, actualPrevConstructedCell.connections.size());
    ASSERT_EQ(3, actualPrevConstructedCell.maxConnections);
    ASSERT_EQ(2, actualPrevPrevConstructedCell.connections.size());
    ASSERT_EQ(2, actualPrevPrevConstructedCell.maxConnections);
    ASSERT_EQ(3, actualPrevPrevPrevConstructedCell.connections.size());
    ASSERT_EQ(3, actualPrevPrevPrevConstructedCell.maxConnections);
    EXPECT_TRUE(hasConnection(actualData, actualConstructedCell.id, 1));
    EXPECT_TRUE(hasConnection(actualData, actualConstructedCell.id, 2));
    EXPECT_TRUE(hasConnection(actualData, actualConstructedCell.id, 4));
}
