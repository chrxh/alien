#include <gtest/gtest.h>

#include "Base/Math.h"
#include "Base/NumberGenerator.h"
#include "EngineInterface/GenomeConstants.h"
#include "EngineInterface/DescriptionEditService.h"
#include "EngineInterface/Descriptions.h"
#include "EngineInterface/GenomeDescriptionService.h"
#include "EngineInterface/SimulationController.h"
#include "IntegrationTestFramework.h"

class ConstructorTests : public IntegrationTestFramework
{
public:
    static SimulationParameters getParameters()
    {
        SimulationParameters result;
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
    auto genome = GenomeDescriptionService::convertDescriptionToBytes(
        GenomeDescription().setHeader(GenomeHeaderDescription().setSeparateConstruction(false)).setCells({CellGenomeDescription()}));

    DataDescription data;
    data.addCell(
        CellDescription()
            .setId(1)
            .setEnergy(_parameters.cellNormalEnergy[0] * 2 - 1.0f)
            .setMaxConnections(1)
            .setExecutionOrderNumber(0)
                     .setCellFunction(ConstructorDescription().setGenome(genome)));

    _simController->setSimulationData(data);
    _simController->calcTimesteps(1);
    auto actualData = _simController->getSimulationData();

    ASSERT_EQ(1, actualData.cells.size());
    auto actualHostCell = getCell(actualData, 1);

    EXPECT_EQ(0, actualHostCell.connections.size());
    EXPECT_EQ(0, std::get<ConstructorDescription>(*actualHostCell.cellFunction).genomeCurrentNodeIndex);
    EXPECT_TRUE(approxCompare(_parameters.cellNormalEnergy[0] * 2 - 1.0f, actualHostCell.energy));
    EXPECT_TRUE(approxCompare(0.0f, actualHostCell.activity.channels[0]));
}

TEST_F(ConstructorTests, alreadyFinished)
{
    DataDescription data;

    auto genome = GenomeDescriptionService::convertDescriptionToBytes(
        GenomeDescription().setHeader(GenomeHeaderDescription().setNumBranches(1)).setCells({CellGenomeDescription()}));

    auto constructor = ConstructorDescription().setGenome(genome).setCurrentBranch(1);

    data.addCell(
        CellDescription()
            .setId(1)
            .setEnergy(_parameters.cellNormalEnergy[0] * 3)
            .setMaxConnections(1)
            .setExecutionOrderNumber(0)
            .setCellFunction(constructor));

    _simController->setSimulationData(data);
    _simController->calcTimesteps(1);
    auto actualData = _simController->getSimulationData();

    ASSERT_EQ(1, actualData.cells.size());
    auto actualHostCell = getCell(actualData, 1);
    auto actualConstructor = std::get<ConstructorDescription>(*actualHostCell.cellFunction);
    EXPECT_EQ(0, actualHostCell.connections.size());
    EXPECT_EQ(0, actualConstructor.genomeCurrentNodeIndex);
    EXPECT_TRUE(approxCompare(0.0f, actualHostCell.activity.channels[0]));
}

TEST_F(ConstructorTests, notActivated)
{
    DataDescription data;

    auto genome = GenomeDescriptionService::convertDescriptionToBytes(
        GenomeDescription().setHeader(GenomeHeaderDescription().setNumBranches(1)).setCells({CellGenomeDescription()}));
    auto constructor = ConstructorDescription().setGenome(genome);

    data.addCell(CellDescription()
                     .setId(1)
                     .setEnergy(_parameters.cellNormalEnergy[0] * 3)
                     .setMaxConnections(1)
                     .setExecutionOrderNumber(0)
                     .setCellFunction(constructor)
                     .setActivationTime(2));

    _simController->setSimulationData(data);
    _simController->calcTimesteps(1);
    auto actualData = _simController->getSimulationData();

    ASSERT_EQ(1, actualData.cells.size());
    auto actualHostCell = getCell(actualData, 1);
    auto actualConstructor = std::get<ConstructorDescription>(*actualHostCell.cellFunction);
    EXPECT_TRUE(approxCompare(0.0f, actualHostCell.activity.channels[0]));
}

TEST_F(ConstructorTests, manualConstruction_noInputActivity)
{
    auto genome = GenomeDescriptionService::convertDescriptionToBytes(GenomeDescription().setCells({CellGenomeDescription()}));

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
    _simController->calcTimesteps(1);
    auto actualData = _simController->getSimulationData();

    ASSERT_EQ(1, actualData.cells.size());
    auto actualHostCell = getCell(actualData, 1);

    EXPECT_EQ(0, actualHostCell.connections.size());
    EXPECT_EQ(0, std::get<ConstructorDescription>(*actualHostCell.cellFunction).genomeCurrentNodeIndex);
    EXPECT_TRUE(approxCompare(_parameters.cellNormalEnergy[0] * 3, actualHostCell.energy));
    EXPECT_TRUE(approxCompare(0.0f, actualHostCell.activity.channels[0]));
}

TEST_F(ConstructorTests, constructFirstCell_correctCycle)
{
    auto genome = GenomeDescriptionService::convertDescriptionToBytes(GenomeDescription().setCells({CellGenomeDescription()}));

    _simController->calcTimesteps(1);

    DataDescription data;
    data.addCell(CellDescription()
                     .setId(1)
                     .setEnergy(_parameters.cellNormalEnergy[0] * 3)
                     .setMaxConnections(1)
                     .setExecutionOrderNumber(0)
                     .setCellFunction(ConstructorDescription().setActivationMode(3).setGenome(genome)));

    _simController->setSimulationData(data);
    for (int i = 0; i < _parameters.cellNumExecutionOrderNumbers * 3; ++i) {
        _simController->calcTimesteps(1);
    }
    auto actualData = _simController->getSimulationData();

    ASSERT_EQ(2, actualData.cells.size());
}

TEST_F(ConstructorTests, constructFirstCell_oneCellGenome_infiniteRepetitions)
{
    auto genome = GenomeDescriptionService::convertDescriptionToBytes(
        GenomeDescription().setHeader(GenomeHeaderDescription().setInfiniteRepetitions()).setCells({CellGenomeDescription()}));

    _simController->calcTimesteps(1);

    DataDescription data;
    data.addCell(CellDescription()
                     .setId(1)
                     .setEnergy(_parameters.cellNormalEnergy[0] * 3)
                     .setMaxConnections(1)
                     .setExecutionOrderNumber(0)
                     .setCellFunction(ConstructorDescription().setActivationMode(3).setGenome(genome)));

    _simController->setSimulationData(data);
    for (int i = 0; i < _parameters.cellNumExecutionOrderNumbers * 3; ++i) {
        _simController->calcTimesteps(1);
    }
    auto actualData = _simController->getSimulationData();
    ASSERT_EQ(2, actualData.cells.size());

    auto actualHostCell = getCell(actualData, 1);
    auto actualConstructedCell = getOtherCell(actualData, {1});
    EXPECT_EQ(0, std::get<ConstructorDescription>(*actualHostCell.cellFunction).genomeCurrentRepetition);
    EXPECT_EQ(0, std::get<ConstructorDescription>(*actualHostCell.cellFunction).currentBranch);
    EXPECT_EQ(LivingState_Activating, actualConstructedCell.livingState);
}

TEST_F(ConstructorTests, constructFirstCell_twoCellGenome_infiniteRepetitions)
{
    auto genome = GenomeDescriptionService::convertDescriptionToBytes(
        GenomeDescription().setHeader(GenomeHeaderDescription().setInfiniteRepetitions()).setCells({CellGenomeDescription(), CellGenomeDescription()}));

    _simController->calcTimesteps(1);

    DataDescription data;
    data.addCell(CellDescription()
                     .setId(1)
                     .setEnergy(_parameters.cellNormalEnergy[0] * 3)
                     .setMaxConnections(1)
                     .setExecutionOrderNumber(0)
                     .setCellFunction(ConstructorDescription().setActivationMode(3).setGenome(genome)));

    _simController->setSimulationData(data);
    for (int i = 0; i < _parameters.cellNumExecutionOrderNumbers * 3; ++i) {
        _simController->calcTimesteps(1);
    }
    auto actualData = _simController->getSimulationData();
    ASSERT_EQ(2, actualData.cells.size());

    auto actualHostCell = getCell(actualData, 1);
    auto actualConstructedCell = getOtherCell(actualData, {1});
    EXPECT_EQ(0, std::get<ConstructorDescription>(*actualHostCell.cellFunction).genomeCurrentRepetition);
    EXPECT_EQ(0, std::get<ConstructorDescription>(*actualHostCell.cellFunction).currentBranch);
    EXPECT_EQ(LivingState_UnderConstruction, actualConstructedCell.livingState);
}

TEST_F(ConstructorTests, constructFirstCell_wrongCycle)
{
    auto genome = GenomeDescriptionService::convertDescriptionToBytes(
        GenomeDescription().setCells({CellGenomeDescription()}));

    _simController->calcTimesteps(1);

    DataDescription data;
    data.addCell(CellDescription()
                     .setId(1)
                     .setEnergy(_parameters.cellNormalEnergy[0] * 3)
                     .setMaxConnections(1)
                     .setExecutionOrderNumber(0)
                     .setCellFunction(ConstructorDescription().setActivationMode(3).setGenome(genome)));

    _simController->setSimulationData(data);
    for (int i = 0; i < _parameters.cellNumExecutionOrderNumbers * 3 - 1; ++i) {
        _simController->calcTimesteps(1);
    }
    auto actualData = _simController->getSimulationData();

    ASSERT_EQ(1, actualData.cells.size());
}

TEST_F(ConstructorTests, constructFirstCell_completenessCheck_constructionNotBuilt)
{
    auto constructorGenome = ConstructorGenomeDescription().setMode(0).setConstructionActivationTime(123).setMakeSelfCopy();
    auto genome =
        GenomeDescriptionService::convertDescriptionToBytes(GenomeDescription().setCells({CellGenomeDescription().setCellFunction(constructorGenome)}));
    auto otherGenome = GenomeDescriptionService::convertDescriptionToBytes(GenomeDescription().setCells({CellGenomeDescription()}));

    DataDescription data;
    data.addCells({
        CellDescription()
            .setId(1)
            .setPos({10.0f, 10.0f})
            .setEnergy(_parameters.cellNormalEnergy[0] * 3)
            .setMaxConnections(2)
            .setExecutionOrderNumber(0)
            .setCellFunction(ConstructorDescription().setGenome(genome).setNumInheritedGenomeNodes(4)),
        CellDescription().setId(2).setPos({11.0f, 10.0f}).setEnergy(100).setMaxConnections(2).setExecutionOrderNumber(5).setCellFunction(NerveDescription()),
        CellDescription()
            .setId(3)
            .setPos({12.0f, 10.0f})
            .setEnergy(100)
            .setMaxConnections(1)
            .setExecutionOrderNumber(4)
            .setCellFunction(ConstructorDescription().setGenome(otherGenome)),
    });
    data.addConnection(1, 2);
    data.addConnection(2, 3);

    _parameters.cellFunctionConstructorCheckCompletenessForSelfReplication = true;
    _simController->setSimulationParameters(_parameters);
    _simController->setSimulationData(data);
    _simController->calcTimesteps(1);
    auto actualData = _simController->getSimulationData();

    ASSERT_EQ(3, actualData.cells.size());
}

TEST_F(ConstructorTests, constructFirstCell_completenessCheck_repeatedConstructionNotBuilt)
{
    auto constructorGenome = ConstructorGenomeDescription().setMode(0).setConstructionActivationTime(123).setMakeSelfCopy();
    auto genome =
        GenomeDescriptionService::convertDescriptionToBytes(GenomeDescription().setCells({CellGenomeDescription().setCellFunction(constructorGenome)}));
    auto otherGenome = GenomeDescriptionService::convertDescriptionToBytes(
        GenomeDescription().setHeader(GenomeHeaderDescription().setNumRepetitions(2)).setCells({CellGenomeDescription()}));

    DataDescription data;
    data.addCells({
        CellDescription()
            .setId(1)
            .setPos({10.0f, 10.0f})
            .setEnergy(_parameters.cellNormalEnergy[0] * 3)
            .setMaxConnections(2)
            .setExecutionOrderNumber(0)
            .setCellFunction(ConstructorDescription().setGenome(genome).setNumInheritedGenomeNodes(5)),
        CellDescription().setId(2).setPos({11.0f, 10.0f}).setEnergy(100).setMaxConnections(2).setExecutionOrderNumber(5).setCellFunction(NerveDescription()),
        CellDescription()
            .setId(3)
            .setPos({12.0f, 10.0f})
            .setEnergy(100)
            .setMaxConnections(2)
            .setExecutionOrderNumber(4)
            .setCellFunction(ConstructorDescription().setGenome(otherGenome).setGenomeCurrentRepetition(1)),
        CellDescription().setId(4).setPos({10.0f, 11.0f}).setEnergy(100).setMaxConnections(1),
    });
    data.addConnection(1, 2);
    data.addConnection(2, 3);
    data.addConnection(3, 4);

    _parameters.cellFunctionConstructorCheckCompletenessForSelfReplication = true;
    _simController->setSimulationParameters(_parameters);
    _simController->setSimulationData(data);
    _simController->calcTimesteps(1);
    auto actualData = _simController->getSimulationData();

    ASSERT_EQ(4, actualData.cells.size());
}

TEST_F(ConstructorTests, constructFirstCell_completenessCheck_constructionBuilt)
{
    auto otherGenome = GenomeDescriptionService::convertDescriptionToBytes(GenomeDescription().setCells({CellGenomeDescription()}));
    auto otherConstructorGenome = ConstructorGenomeDescription().setMode(0).setConstructionActivationTime(123).setGenome(otherGenome);

    auto constructorGenome = ConstructorGenomeDescription().setMode(0).setConstructionActivationTime(123).setMakeSelfCopy();
    auto genome = GenomeDescriptionService::convertDescriptionToBytes(GenomeDescription().setCells(
        {CellGenomeDescription().setCellFunction(constructorGenome), CellGenomeDescription(), CellGenomeDescription().setCellFunction(otherConstructorGenome)}));

    DataDescription data;
    data.addCells({
        CellDescription()
            .setId(1)
            .setPos({10.0f, 10.0f})
            .setEnergy(_parameters.cellNormalEnergy[0] * 3)
            .setMaxConnections(2)
            .setExecutionOrderNumber(0)
            .setCellFunction(ConstructorDescription().setGenome(genome).setNumInheritedGenomeNodes(4)),
        CellDescription().setId(2).setPos({11.0f, 10.0f}).setEnergy(100).setMaxConnections(2).setExecutionOrderNumber(5).setCellFunction(NerveDescription()),
        CellDescription()
            .setId(3)
            .setPos({12.0f, 10.0f})
            .setEnergy(100)
            .setMaxConnections(2)
            .setExecutionOrderNumber(4)
            .setCellFunction(ConstructorDescription().setGenome(otherGenome).setCurrentBranch(1)),
        CellDescription().setId(4).setPos({10.0f, 11.0f}).setEnergy(100).setMaxConnections(1),
    });
    data.addConnection(1, 2);
    data.addConnection(2, 3);
    data.addConnection(3, 4);

    _parameters.cellFunctionConstructorCheckCompletenessForSelfReplication = true;
    _simController->setSimulationParameters(_parameters);
    _simController->setSimulationData(data);
    _simController->calcTimesteps(1);
    auto actualData = _simController->getSimulationData();

    ASSERT_EQ(5, actualData.cells.size());
}

TEST_F(ConstructorTests, constructFirstCell_completenessCheck_infiniteConstructionsBuilt)
{
    auto otherGenome = GenomeDescriptionService::convertDescriptionToBytes(
        GenomeDescription().setHeader(GenomeHeaderDescription().setInfiniteRepetitions()).setCells({CellGenomeDescription()}));
    auto otherConstructorGenome = ConstructorGenomeDescription().setMode(0).setConstructionActivationTime(123).setGenome(otherGenome);

    auto constructorGenome = ConstructorGenomeDescription().setMode(0).setConstructionActivationTime(123).setMakeSelfCopy();
    auto genome = GenomeDescriptionService::convertDescriptionToBytes(GenomeDescription().setCells(
        {CellGenomeDescription().setCellFunction(constructorGenome), CellGenomeDescription(), CellGenomeDescription().setCellFunction(otherConstructorGenome)}));

    DataDescription data;
    data.addCells({
        CellDescription()
            .setId(1)
            .setPos({10.0f, 10.0f})
            .setEnergy(_parameters.cellNormalEnergy[0] * 3)
            .setMaxConnections(2)
            .setExecutionOrderNumber(0)
            .setCellFunction(ConstructorDescription().setGenome(genome).setNumInheritedGenomeNodes(4)),
        CellDescription().setId(2).setPos({11.0f, 10.0f}).setEnergy(100).setMaxConnections(2).setExecutionOrderNumber(5).setCellFunction(NerveDescription()),
        CellDescription()
            .setId(3)
            .setPos({12.0f, 10.0f})
            .setEnergy(100)
            .setMaxConnections(2)
            .setExecutionOrderNumber(4)
            .setCellFunction(ConstructorDescription().setGenome(otherGenome).setCurrentBranch(1)),
        CellDescription().setId(4).setPos({10.0f, 11.0f}).setEnergy(100).setMaxConnections(1),
    });
    data.addConnection(1, 2);
    data.addConnection(2, 3);
    data.addConnection(3, 4);

    _parameters.cellFunctionConstructorCheckCompletenessForSelfReplication = true;
    _simController->setSimulationParameters(_parameters);
    _simController->setSimulationData(data);
    _simController->calcTimesteps(1);
    auto actualData = _simController->getSimulationData();

    ASSERT_EQ(5, actualData.cells.size());
}

TEST_F(ConstructorTests, constructFirstCell_completenessCheck_largeCluster)
{
    auto constexpr RectLength = 50;
    auto rect = DescriptionEditService::createRect(DescriptionEditService::CreateRectParameters().height(RectLength).width(RectLength));

    auto constructorGenome = ConstructorGenomeDescription().setMode(0).setConstructionActivationTime(123).setMakeSelfCopy();
    auto genome =
        GenomeDescriptionService::convertDescriptionToBytes(GenomeDescription().setCells({CellGenomeDescription().setCellFunction(constructorGenome)}));
    auto otherGenome = GenomeDescriptionService::convertDescriptionToBytes(GenomeDescription().setCells({CellGenomeDescription()}));

    auto& cell1 = rect.cells.at(0);
    cell1.setEnergy(_parameters.cellNormalEnergy[0] * 3)
        .setExecutionOrderNumber(0)
        .setCellFunction(ConstructorDescription().setGenome(genome).setNumInheritedGenomeNodes(RectLength * RectLength));

    _parameters.cellFunctionConstructorCheckCompletenessForSelfReplication = true;
    _simController->setSimulationParameters(_parameters);
    _simController->setSimulationData(rect);
    _simController->calcTimesteps(1);
    auto actualData = _simController->getSimulationData();

    ASSERT_EQ(RectLength * RectLength + 1, actualData.cells.size());
}

TEST_F(ConstructorTests, constructFirstCell_completenessCheck_thinCluster)
{
    auto constructorGenome = ConstructorGenomeDescription().setMode(0).setConstructionActivationTime(123).setMakeSelfCopy();
    auto genome = GenomeDescriptionService::convertDescriptionToBytes(GenomeDescription()
                                                                          .setHeader(GenomeHeaderDescription().setNumBranches(2))
                                                                          .setCells(
                                                                              {CellGenomeDescription(),
                                                                               CellGenomeDescription(),
                                                                               CellGenomeDescription().setCellFunction(constructorGenome),
                                                                               CellGenomeDescription(),
                                                                               CellGenomeDescription()}));

    DataDescription data;
    data.addCells({
        CellDescription()
            .setId(1)
            .setPos({10.0f, 10.0f})
            .setEnergy(_parameters.cellNormalEnergy[0] * 3)
            .setMaxConnections(3)
            .setExecutionOrderNumber(0)
            .setCellFunction(ConstructorDescription().setGenome(genome).setNumInheritedGenomeNodes(5))
            .setCreatureId(1),
        CellDescription()
            .setId(2)
            .setPos({10.0f, 9.0f})
            .setEnergy(100)
            .setMaxConnections(2)
            .setExecutionOrderNumber(5)
            .setCellFunction(NerveDescription())
            .setCreatureId(1),
        CellDescription()
            .setId(3)
            .setPos({10.0f, 8.0f})
            .setEnergy(100)
            .setMaxConnections(1)
            .setExecutionOrderNumber(4)
            .setCellFunction(NerveDescription())
            .setCreatureId(1),
        CellDescription()
            .setId(4)
            .setPos({10.0f, 11.0f})
            .setEnergy(100)
            .setMaxConnections(2)
            .setExecutionOrderNumber(4)
            .setCellFunction(NerveDescription())
            .setCreatureId(1),
        CellDescription()
            .setId(5)
            .setPos({10.0f, 12.0f})
            .setEnergy(100)
            .setMaxConnections(1)
            .setExecutionOrderNumber(4)
            .setCellFunction(NerveDescription())
            .setCreatureId(1),
        CellDescription()
            .setId(6)
            .setPos({11.0f, 10.0f})
            .setEnergy(100)
            .setMaxConnections(1)
            .setExecutionOrderNumber(4)
            .setCellFunction(NerveDescription())
            .setCreatureId(2),
    });
    data.addConnection(1, 2);
    data.addConnection(1, 6);
    data.addConnection(1, 4);
    data.addConnection(2, 3);
    data.addConnection(4, 5);

    auto& firstCell = data.cells.at(0);
    while (true) {
        if (firstCell.connections.at(0).cellId != 2) {
            std::vector<ConnectionDescription> newConnections;
            newConnections.emplace_back(firstCell.connections.back());
            for (int j = 0; j < firstCell.connections.size() - 1; ++j) {
                newConnections.emplace_back(firstCell.connections.at(j));
            }
            firstCell.connections = newConnections;
        } else {
            break;
        }
    }

    _parameters.cellFunctionConstructorCheckCompletenessForSelfReplication = true;
    _simController->setSimulationParameters(_parameters);
    _simController->setSimulationData(data);
    _simController->calcTimesteps(1);
    auto actualData = _simController->getSimulationData();

    ASSERT_EQ(7, actualData.cells.size());
}

/**
 * Completeness check needs to inspect cells under construction because when a constructor is finished its construction
 * is still in state "under construction" for some time steps but needs to be inspected
 *
 * UPDATE: Test does not make sense with new completeness check
 */
TEST_F(ConstructorTests, DISABLED_constructFirstCell_completenessCheck_underConstruction)
{
    auto constructorGenome = ConstructorGenomeDescription().setMode(0).setConstructionActivationTime(123).setMakeSelfCopy();
    auto genome =
        GenomeDescriptionService::convertDescriptionToBytes(GenomeDescription().setCells({CellGenomeDescription().setCellFunction(constructorGenome)}));
    auto otherGenome = GenomeDescriptionService::convertDescriptionToBytes(GenomeDescription().setCells({CellGenomeDescription()}));

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
            .setPos({11.0f, 10.0f})
            .setEnergy(100)
            .setMaxConnections(2)
            .setExecutionOrderNumber(4)
            .setCellFunction(ConstructorDescription().setGenome(otherGenome).setCurrentBranch(1)),
        CellDescription()
            .setId(3)
            .setPos({12.0f, 10.0f})
            .setEnergy(100)
            .setMaxConnections(2)
            .setLivingState(LivingState_UnderConstruction)
            .setExecutionOrderNumber(4)
            .setCellFunction(ConstructorDescription().setGenome(otherGenome).setCurrentBranch(0)),
    });
    data.addConnection(1, 2);
    data.addConnection(2, 3);

    _parameters.cellFunctionConstructorCheckCompletenessForSelfReplication = true;
    _simController->setSimulationParameters(_parameters);
    _simController->setSimulationData(data);
    _simController->calcTimesteps(1);
    auto actualData = _simController->getSimulationData();

    ASSERT_EQ(3, actualData.cells.size());
}

TEST_F(ConstructorTests, constructFirstCell_noSeparation)
{
    auto genome = GenomeDescriptionService::convertDescriptionToBytes(
        GenomeDescription()
            .setHeader(GenomeHeaderDescription().setSeparateConstruction(false).setStiffness(0.35f))
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
    _simController->calcTimesteps(1);
    auto actualData = _simController->getSimulationData();

    ASSERT_EQ(2, actualData.cells.size());
    auto actualHostCell = getCell(actualData, 1);
    auto actualConstructedCell = getOtherCell(actualData, 1);

    EXPECT_EQ(1, actualHostCell.connections.size());

    auto const& actualConstructor = std::get<ConstructorDescription>(*actualHostCell.cellFunction);
    EXPECT_EQ(0, actualConstructor.genomeCurrentNodeIndex);
    EXPECT_EQ(0, actualConstructor.genomeCurrentRepetition);
    EXPECT_EQ(1, actualConstructor.currentBranch);
    EXPECT_TRUE(approxCompare(_parameters.cellNormalEnergy[0] * 2, actualHostCell.energy));
    EXPECT_TRUE(approxCompare(1.0f, actualHostCell.activity.channels[0]));
    EXPECT_EQ(LivingState_Activating, actualConstructedCell.livingState);

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
    auto genome = GenomeDescriptionService::convertDescriptionToBytes(
        GenomeDescription()
            .setHeader(GenomeHeaderDescription().setSeparateConstruction(false)).setCells({CellGenomeDescription(), CellGenomeDescription()}));

    DataDescription data;
    data.addCell(CellDescription()
                     .setId(1)
                     .setPos({10.0f, 10.0f})
                     .setEnergy(_parameters.cellNormalEnergy[0] * 3)
                     .setMaxConnections(1)
                     .setExecutionOrderNumber(0)
                     .setCellFunction(ConstructorDescription().setGenome(genome)));

    _simController->setSimulationData(data);
    _simController->calcTimesteps(1);
    auto actualData = _simController->getSimulationData();

    ASSERT_EQ(2, actualData.cells.size());
    auto actualHostCell = getCell(actualData, 1);
    auto actualConstructedCell = getOtherCell(actualData, 1);
    auto const& actualConstructor = std::get<ConstructorDescription>(*actualHostCell.cellFunction);
    EXPECT_EQ(1, actualConstructor.genomeCurrentNodeIndex);
    EXPECT_EQ(0, actualConstructor.genomeCurrentRepetition);
    EXPECT_EQ(0, actualConstructor.currentBranch);

    EXPECT_EQ(1, actualHostCell.connections.size());
    EXPECT_EQ(LivingState_Ready, actualHostCell.livingState);

    EXPECT_EQ(1, actualConstructedCell.connections.size());
    EXPECT_EQ(LivingState_UnderConstruction, actualConstructedCell.livingState);
}

TEST_F(ConstructorTests, constructFirstCell_separation)
{
    auto genome = GenomeDescriptionService::convertDescriptionToBytes(
        GenomeDescription().setHeader(GenomeHeaderDescription().setSeparateConstruction(true)).setCells({CellGenomeDescription()}));

    DataDescription data;
    data.addCell(CellDescription()
                     .setId(1)
                     .setEnergy(_parameters.cellNormalEnergy[0] * 3)
                     .setMaxConnections(1)
                     .setExecutionOrderNumber(0)
                     .setCellFunction(ConstructorDescription()
                                          .setGenome(genome)));

    _simController->setSimulationData(data);
    _simController->calcTimesteps(1);
    auto actualData = _simController->getSimulationData();

    ASSERT_EQ(2, actualData.cells.size());
    auto actualHostCell = getCell(actualData, 1);
    auto actualConstructedCell = getOtherCell(actualData, 1);

    EXPECT_EQ(0, actualHostCell.connections.size());
    EXPECT_EQ(1, actualHostCell.maxConnections);
    auto const& actualConstructor = std::get<ConstructorDescription>(*actualHostCell.cellFunction);
    EXPECT_EQ(0, actualConstructor.genomeCurrentNodeIndex);
    EXPECT_EQ(0, actualConstructor.genomeCurrentRepetition);
    EXPECT_EQ(0, actualConstructor.currentBranch);

    EXPECT_EQ(0, actualConstructedCell.connections.size());
    EXPECT_EQ(0, actualConstructedCell.maxConnections);
    EXPECT_EQ(LivingState_Activating, actualConstructedCell.livingState);
}

TEST_F(ConstructorTests, constructFirstCell_manualConstruction)
{
    auto genome = GenomeDescriptionService::convertDescriptionToBytes(
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
    _simController->calcTimesteps(1);
    auto actualData = _simController->getSimulationData();

    ASSERT_EQ(3, actualData.cells.size());
    auto actualHostCell = getCell(actualData, 1);
    auto actualConstructedCell = getOtherCell(actualData, {1, 2});

    EXPECT_EQ(1, actualHostCell.connections.size());

    EXPECT_EQ(0, actualConstructedCell.connections.size());
    EXPECT_EQ(LivingState_Activating, actualConstructedCell.livingState);

    EXPECT_TRUE(approxCompare(10.0f - 1.0f, actualConstructedCell.pos.x));
    EXPECT_TRUE(approxCompare(10.0f, actualConstructedCell.pos.y));
}

TEST_F(ConstructorTests, constructFirstCell_differentAngle1)
{
    auto genome = GenomeDescriptionService::convertDescriptionToBytes(GenomeDescription().setCells({CellGenomeDescription()}));

    DataDescription data;
    data.addCells({
        CellDescription()
             .setId(1)
             .setPos({10.0f, 10.0f})
             .setEnergy(_parameters.cellNormalEnergy[0] * 3)
             .setMaxConnections(2)
             .setExecutionOrderNumber(0)
             .setInputExecutionOrderNumber(5)
             .setCellFunction(ConstructorDescription().setActivationMode(0).setGenome(genome).setConstructionAngle1(90.0f)),
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
    _simController->calcTimesteps(1);
    auto actualData = _simController->getSimulationData();

    ASSERT_EQ(3, actualData.cells.size());
    auto actualHostCell = getCell(actualData, 1);
    auto actualConstructedCell = getOtherCell(actualData, {1, 2});

    EXPECT_TRUE(approxCompare(10.0f, actualConstructedCell.pos.x));
    EXPECT_TRUE(approxCompare(10.0f - 1.0f, actualConstructedCell.pos.y));
}

TEST_F(ConstructorTests, constructFirstCell_differentAngle2)
{
    auto genome = GenomeDescriptionService::convertDescriptionToBytes(GenomeDescription().setCells({CellGenomeDescription().setReferenceAngle(-90.0f)}));

    DataDescription data;
    data.addCells(
        {CellDescription()
             .setId(1)
             .setPos({10.0f, 10.0f})
             .setEnergy(_parameters.cellNormalEnergy[0] * 3)
             .setMaxConnections(2)
             .setExecutionOrderNumber(0)
             .setInputExecutionOrderNumber(5)
             .setCellFunction(ConstructorDescription().setActivationMode(0).setGenome(genome).setConstructionAngle1(-90.0f)),
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
    _simController->calcTimesteps(1);
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

    auto genome = GenomeDescriptionService::convertDescriptionToBytes(GenomeDescription().setCells({CellGenomeDescription().setCellFunction(neuron)}));

    DataDescription data;
    data.addCell(
        CellDescription()
            .setId(1)
            .setEnergy(_parameters.cellNormalEnergy[0] * 3)
            .setMaxConnections(1)
            .setExecutionOrderNumber(0)
                     .setCellFunction(ConstructorDescription().setGenome(genome)));

    _simController->setSimulationData(data);
    _simController->calcTimesteps(1);
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
        GenomeDescriptionService::convertDescriptionToBytes(GenomeDescription().setCells({CellGenomeDescription().setCellFunction(constructorGenome)}));

    DataDescription data;

    data.addCell(CellDescription()
                     .setId(1)
                     .setEnergy(_parameters.cellNormalEnergy[0] * 3)
                     .setMaxConnections(1)
                     .setExecutionOrderNumber(0)
                     .setCellFunction(ConstructorDescription().setGenome(genome)));

    _simController->setSimulationData(data);
    _simController->calcTimesteps(1);
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
    auto genome = GenomeDescriptionService::convertDescriptionToBytes(GenomeDescription().setCells({CellGenomeDescription().setCellFunction(nerveDesc)}));

    DataDescription data;
    data.addCell(CellDescription()
                     .setId(1)
                     .setEnergy(_parameters.cellNormalEnergy[0] * 3)
                     .setMaxConnections(1)
                     .setExecutionOrderNumber(0)
                     .setCellFunction(ConstructorDescription().setGenome(genome)));

    _simController->setSimulationData(data);
    _simController->calcTimesteps(1);
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
    auto genome = GenomeDescriptionService::convertDescriptionToBytes(GenomeDescription().setCells({CellGenomeDescription().setCellFunction(attackerDesc)}));

    DataDescription data;
    data.addCell(CellDescription()
                     .setId(1)
                     .setEnergy(_parameters.cellNormalEnergy[0] * 3)
                     .setMaxConnections(1)
                     .setExecutionOrderNumber(0)
                     .setCellFunction(ConstructorDescription().setGenome(genome)));

    _simController->setSimulationData(data);
    _simController->calcTimesteps(1);
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
    auto genome = GenomeDescriptionService::convertDescriptionToBytes(GenomeDescription().setCells({CellGenomeDescription().setCellFunction(defenderDesc)}));

    DataDescription data;
    data.addCell(CellDescription()
                     .setId(1)
                     .setEnergy(_parameters.cellNormalEnergy[0] * 3)
                     .setMaxConnections(1)
                     .setExecutionOrderNumber(0)
                     .setCellFunction(ConstructorDescription().setGenome(genome)));

    _simController->setSimulationData(data);
    _simController->calcTimesteps(1);
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
        GenomeDescriptionService::convertDescriptionToBytes(GenomeDescription().setCells({CellGenomeDescription().setCellFunction(transmitterDesc)}));

    DataDescription data;
    data.addCell(CellDescription()
                     .setId(1)
                     .setEnergy(_parameters.cellNormalEnergy[0] * 3)
                     .setMaxConnections(1)
                     .setExecutionOrderNumber(0)
                     .setCellFunction(ConstructorDescription().setGenome(genome)));

    _simController->setSimulationData(data);
    _simController->calcTimesteps(1);
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
    auto genome = GenomeDescriptionService::convertDescriptionToBytes(GenomeDescription().setCells({CellGenomeDescription().setCellFunction(muscleDesc)}));

    DataDescription data;
    data.addCell(CellDescription()
                     .setId(1)
                     .setEnergy(_parameters.cellNormalEnergy[0] * 3)
                     .setMaxConnections(1)
                     .setExecutionOrderNumber(0)
                     .setCellFunction(ConstructorDescription().setGenome(genome)));

    _simController->setSimulationData(data);
    _simController->calcTimesteps(1);
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
    auto genome = GenomeDescriptionService::convertDescriptionToBytes(GenomeDescription().setCells({CellGenomeDescription().setCellFunction(sensorDesc)}));

    DataDescription data;
    data.addCell(CellDescription()
                     .setId(1)
                     .setEnergy(_parameters.cellNormalEnergy[0] * 3)
                     .setMaxConnections(1)
                     .setExecutionOrderNumber(0)
                     .setCellFunction(ConstructorDescription().setGenome(genome)));

    _simController->setSimulationData(data);
    _simController->calcTimesteps(1);
    auto actualData = _simController->getSimulationData();

    ASSERT_EQ(2, actualData.cells.size());
    auto actualConstructedCell = getOtherCell(actualData, 1);

    EXPECT_EQ(CellFunction_Sensor, actualConstructedCell.getCellFunctionType());

    auto actualSensor = std::get<SensorDescription>(*actualConstructedCell.cellFunction);
    EXPECT_EQ(sensorDesc.fixedAngle.has_value(), actualSensor.fixedAngle.has_value());
    EXPECT_TRUE(lowPrecisionCompare(*sensorDesc.fixedAngle, *actualSensor.fixedAngle));
    EXPECT_TRUE(lowPrecisionCompare(sensorDesc.minDensity, actualSensor.minDensity));
    EXPECT_EQ(sensorDesc.restrictToColor, actualSensor.restrictToColor);
}

TEST_F(ConstructorTests, constructInjectorCell)
{
    auto injectorDesc = InjectorGenomeDescription().setMode(InjectorMode_InjectOnlyEmptyCells).setGenome(createRandomGenome(MAX_GENOME_BYTES / 2));
    auto genome = GenomeDescriptionService::convertDescriptionToBytes(GenomeDescription().setCells({CellGenomeDescription().setCellFunction(injectorDesc)}));

    DataDescription data;
    data.addCell(CellDescription()
                     .setId(1)
                     .setEnergy(_parameters.cellNormalEnergy[0] * 3)
                     .setMaxConnections(1)
                     .setExecutionOrderNumber(0)
                     .setCellFunction(ConstructorDescription().setGenome(genome)));

    _simController->setSimulationData(data);
    _simController->calcTimesteps(1);
    auto actualData = _simController->getSimulationData();

    ASSERT_EQ(2, actualData.cells.size());
    auto actualConstructedCell = getOtherCell(actualData, 1);

    EXPECT_EQ(CellFunction_Injector, actualConstructedCell.getCellFunctionType());

    auto actualInjector = std::get<InjectorDescription>(*actualConstructedCell.cellFunction);
    EXPECT_EQ(injectorDesc.mode, actualInjector.mode);
    EXPECT_EQ(injectorDesc.getGenomeData(), actualInjector.genome);
}

TEST_F(ConstructorTests, constructReconnectorCell)
{
    auto reconnectorDesc = ReconnectorGenomeDescription().setRestrictToColor(2).setRestrictToMutation(ReconnectorRestrictToMutation_RestrictToSameMutants);
    auto genome =
        GenomeDescriptionService::convertDescriptionToBytes(GenomeDescription().setCells({CellGenomeDescription().setCellFunction(reconnectorDesc)}));

    DataDescription data;
    data.addCell(CellDescription()
                     .setId(1)
                     .setEnergy(_parameters.cellNormalEnergy[0] * 3)
                     .setMaxConnections(1)
                     .setExecutionOrderNumber(0)
                     .setCellFunction(ConstructorDescription().setGenome(genome)));

    _simController->setSimulationData(data);
    _simController->calcTimesteps(1);
    auto actualData = _simController->getSimulationData();

    ASSERT_EQ(2, actualData.cells.size());
    auto actualConstructedCell = getOtherCell(actualData, 1);

    EXPECT_EQ(CellFunction_Reconnector, actualConstructedCell.getCellFunctionType());

    auto actualReconnector = std::get<ReconnectorDescription>(*actualConstructedCell.cellFunction);
    EXPECT_EQ(reconnectorDesc.restrictToColor, actualReconnector.restrictToColor);
    EXPECT_EQ(reconnectorDesc.restrictToMutation, actualReconnector.restrictToMutation);
}

TEST_F(ConstructorTests, constructDetonatorCell)
{
    auto detonatorDesc = DetonatorGenomeDescription().setCountDown(25);
    auto genome = GenomeDescriptionService::convertDescriptionToBytes(GenomeDescription().setCells({CellGenomeDescription().setCellFunction(detonatorDesc)}));

    DataDescription data;
    data.addCell(CellDescription()
                     .setId(1)
                     .setEnergy(_parameters.cellNormalEnergy[0] * 3)
                     .setMaxConnections(1)
                     .setExecutionOrderNumber(0)
                     .setCellFunction(ConstructorDescription().setGenome(genome)));

    _simController->setSimulationData(data);
    _simController->calcTimesteps(1);
    auto actualData = _simController->getSimulationData();

    ASSERT_EQ(2, actualData.cells.size());
    auto actualConstructedCell = getOtherCell(actualData, 1);

    EXPECT_EQ(CellFunction_Detonator, actualConstructedCell.getCellFunctionType());

    auto actualDetonator = std::get<DetonatorDescription>(*actualConstructedCell.cellFunction);
    EXPECT_EQ(detonatorDesc.countdown, actualDetonator.countdown);
}

TEST_F(ConstructorTests, constructConstructorCell_nestingGenomeTooLarge)
{
    auto constructedConstructor = ConstructorGenomeDescription().setMode(0).setGenome(createRandomGenome(MAX_GENOME_BYTES));
    auto genome =
        GenomeDescriptionService::convertDescriptionToBytes(GenomeDescription().setCells({CellGenomeDescription().setCellFunction(constructedConstructor)}));


    DataDescription data;

    data.addCell(CellDescription()
                     .setId(1)
                     .setEnergy(_parameters.cellNormalEnergy[0] * 3)
                     .setMaxConnections(1)
                     .setExecutionOrderNumber(0)
                     .setCellFunction(ConstructorDescription().setGenome(genome)));

    _simController->setSimulationData(data);
    _simController->calcTimesteps(1);
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
    auto constructedConstructor = ConstructorGenomeDescription().setMode(0).setMakeSelfCopy();

    auto genome =
        GenomeDescriptionService::convertDescriptionToBytes(GenomeDescription().setCells({CellGenomeDescription().setCellFunction(constructedConstructor)}));

    DataDescription data;
    data.addCell(CellDescription()
                     .setId(1)
                     .setEnergy(_parameters.cellNormalEnergy[0] * 3)
                     .setMaxConnections(1)
                     .setExecutionOrderNumber(0)
                     .setCellFunction(ConstructorDescription().setGenome(genome)));

    _simController->setSimulationData(data);
    _simController->calcTimesteps(1);
    auto actualData = _simController->getSimulationData();

    ASSERT_EQ(2, actualData.cells.size());
    auto actualConstructedCell = getOtherCell(actualData, 1);

    EXPECT_EQ(CellFunction_Constructor, actualConstructedCell.getCellFunctionType());

    auto actualConstructor = std::get<ConstructorDescription>(*actualConstructedCell.cellFunction);
    EXPECT_EQ(genome, actualConstructor.genome);
}

TEST_F(ConstructorTests, constructSecondCell_separation)
{
    auto genome = GenomeDescriptionService::convertDescriptionToBytes(
        GenomeDescription().setHeader(GenomeHeaderDescription().setSeparateConstruction(true)).setCells({CellGenomeDescription(), CellGenomeDescription()}));

    DataDescription data;
    data.addCells({
        CellDescription()
            .setId(1)
            .setPos({10.0f, 10.0f})
            .setEnergy(_parameters.cellNormalEnergy[0] * 3)
            .setMaxConnections(1)
            .setExecutionOrderNumber(0)
            .setCellFunction(ConstructorDescription().setGenomeCurrentNodeIndex(1).setGenome(genome)),
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
    _simController->calcTimesteps(1);
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
    EXPECT_EQ(LivingState_Activating, actualConstructedCell.livingState);
    EXPECT_TRUE(lowPrecisionCompare(1.0f, actualConstructedCell.connections[0].distance));
}

TEST_F(ConstructorTests, constructSecondCell_constructionStateTransitions)
{
    auto genome = GenomeDescriptionService::convertDescriptionToBytes(
        GenomeDescription().setHeader(GenomeHeaderDescription().setSeparateConstruction(true)).setCells({CellGenomeDescription(), CellGenomeDescription()}));

    DataDescription data;
    data.addCells({
        CellDescription()
            .setId(1)
            .setPos({10.0f, 10.0f})
            .setEnergy(_parameters.cellNormalEnergy[0] * 3)
            .setMaxConnections(1)
            .setExecutionOrderNumber(0)
            .setCellFunction(ConstructorDescription().setGenomeCurrentNodeIndex(1).setGenome(genome)),
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
    _simController->calcTimesteps(1);
    _simController->calcTimesteps(1);
    {
        auto actualData = _simController->getSimulationData();

        ASSERT_EQ(3, actualData.cells.size());
        auto actualHostCell = getCell(actualData, 1);
        auto actualPrevConstructedCell = getCell(actualData, 2);
        auto actualConstructedCell = getOtherCell(actualData, {1, 2});

        EXPECT_EQ(LivingState_Ready, actualHostCell.livingState);
        EXPECT_EQ(LivingState_Activating, actualPrevConstructedCell.livingState);
        EXPECT_EQ(LivingState_Ready, actualConstructedCell.livingState);
    }
    _simController->calcTimesteps(1);
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
    auto genome = GenomeDescriptionService::convertDescriptionToBytes(
        GenomeDescription().setHeader(GenomeHeaderDescription().setSeparateConstruction(false)).setCells({CellGenomeDescription(), CellGenomeDescription()}));

    DataDescription data;
    data.addCells({
        CellDescription()
            .setId(1)
            .setPos({10.0f, 10.0f})
            .setEnergy(_parameters.cellNormalEnergy[0] * 3)
            .setMaxConnections(1)
            .setExecutionOrderNumber(0)
            .setCellFunction(ConstructorDescription().setGenomeCurrentNodeIndex(1).setGenome(genome)),
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
    _simController->calcTimesteps(1);
    auto actualData = _simController->getSimulationData();

    ASSERT_EQ(3, actualData.cells.size());
    auto actualHostCell = getCell(actualData, 1);
    auto actualPrevConstructedCell = getCell(actualData, 2);
    auto actualConstructedCell = getOtherCell(actualData, {1, 2});

    EXPECT_EQ(1, actualHostCell.connections.size());
    EXPECT_EQ(LivingState_Ready, actualHostCell.livingState);
    EXPECT_TRUE(lowPrecisionCompare(1.0f, actualPrevConstructedCell.connections[0].distance));

    ASSERT_EQ(2, actualConstructedCell.connections.size());
    EXPECT_EQ(LivingState_Activating, actualConstructedCell.livingState);
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
    auto genome = GenomeDescriptionService::convertDescriptionToBytes(
        GenomeDescription().setHeader(GenomeHeaderDescription().setSeparateConstruction(false)).setCells({CellGenomeDescription(), CellGenomeDescription()}));

    DataDescription data;
    data.addCells({
        CellDescription()
            .setId(1)
            .setPos({10.0f, 10.0f})
            .setEnergy(_parameters.cellNormalEnergy[0] * 3)
            .setMaxConnections(1)
            .setExecutionOrderNumber(0)
            .setCellFunction(ConstructorDescription().setGenomeCurrentNodeIndex(1).setGenome(genome)),
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
    _simController->calcTimesteps(1);
    auto actualData = _simController->getSimulationData();

    ASSERT_EQ(2, actualData.cells.size());
    auto actualHostCell = getCell(actualData, 1);
    auto actualPrevConstructedCell = getCell(actualData, 2);

    EXPECT_EQ(1, actualHostCell.connections.size());
    EXPECT_TRUE(approxCompare(0.0f, actualHostCell.activity.channels[0]));
    ASSERT_EQ(1, actualPrevConstructedCell.connections.size());
    auto actualConstructor = std::get<ConstructorDescription>(*actualHostCell.cellFunction);
    EXPECT_EQ(1, actualConstructor.genomeCurrentNodeIndex);
}

TEST_F(ConstructorTests, constructSecondCell_notFinished)
{
    auto genome = GenomeDescriptionService::convertDescriptionToBytes(GenomeDescription()
                                                                .setHeader(GenomeHeaderDescription().setSeparateConstruction(false))
                                                                .setCells({CellGenomeDescription(), CellGenomeDescription(), CellGenomeDescription()}));

    DataDescription data;
    data.addCells({
        CellDescription()
            .setId(1)
            .setPos({10.0f, 10.0f})
            .setEnergy(_parameters.cellNormalEnergy[0] * 3)
            .setMaxConnections(1)
            .setExecutionOrderNumber(0)
            .setCellFunction(ConstructorDescription().setGenomeCurrentNodeIndex(1).setGenome(genome)),
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
    _simController->calcTimesteps(1);
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
    auto genome = GenomeDescriptionService::convertDescriptionToBytes(
        GenomeDescription().setHeader(GenomeHeaderDescription().setSeparateConstruction(false)).setCells({CellGenomeDescription(), CellGenomeDescription()}));

    DataDescription data;
    data.addCells({
        CellDescription()
            .setId(1)
            .setPos({10.0f, 10.0f})
            .setEnergy(_parameters.cellNormalEnergy[0] * 3)
            .setMaxConnections(1)
            .setExecutionOrderNumber(0)
            .setCellFunction(ConstructorDescription().setGenome(genome).setConstructionAngle2(90.0f).setGenomeCurrentNodeIndex(1)),
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
    _simController->calcTimesteps(1);
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
    auto genome = GenomeDescriptionService::convertDescriptionToBytes(
        GenomeDescription().setHeader(GenomeHeaderDescription().setSeparateConstruction(false)).setCells({CellGenomeDescription(), CellGenomeDescription()}));

    DataDescription data;
    data.addCells({
        CellDescription()
            .setId(1)
            .setPos({10.0f, 10.0f})
            .setEnergy(_parameters.cellNormalEnergy[0] * 3)
            .setMaxConnections(1)
            .setExecutionOrderNumber(0)
            .setCellFunction(ConstructorDescription().setGenome(genome).setConstructionAngle2(-90.0f).setGenomeCurrentNodeIndex(1)),
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
    _simController->calcTimesteps(1);
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

TEST_F(ConstructorTests, constructSecondCell_twoCellGenome_infiniteRepetitions)
{
    auto genome = GenomeDescriptionService::convertDescriptionToBytes(
        GenomeDescription().setHeader(GenomeHeaderDescription().setInfiniteRepetitions()).setCells({CellGenomeDescription(), CellGenomeDescription()}));

    DataDescription data;
    data.addCells({
        CellDescription()
            .setId(1)
            .setPos({10.0f - _parameters.cellFunctionConstructorOffspringDistance[0], 10.0f})
            .setEnergy(_parameters.cellNormalEnergy[0] * 3)
            .setMaxConnections(1)
            .setExecutionOrderNumber(0)
            .setCellFunction(ConstructorDescription().setGenomeCurrentNodeIndex(1).setGenome(genome)),
        CellDescription().setId(2).setPos({11.0f, 10.0f}).setMaxConnections(1).setLivingState(LivingState_UnderConstruction),
    });
    data.addConnection(1, 2);

    _simController->setSimulationData(data);
    _simController->calcTimesteps(1);

    auto actualData = _simController->getSimulationData();
    ASSERT_EQ(3, actualData.cells.size());

    auto actualHostCell = getCell(actualData, 1);
    auto actualConstructedCell = getOtherCell(actualData, {1, 2});
    EXPECT_EQ(0, std::get<ConstructorDescription>(*actualHostCell.cellFunction).currentBranch);
    EXPECT_EQ(LivingState_Activating, actualConstructedCell.livingState);
}

TEST_F(ConstructorTests, constructThirdCell_multipleConnections_upperPart)
{
    auto genome = GenomeDescriptionService::convertDescriptionToBytes(GenomeDescription()
                                                                .setHeader(GenomeHeaderDescription().setSeparateConstruction(false))
                                                                .setCells({CellGenomeDescription(), CellGenomeDescription(), CellGenomeDescription()}));

    DataDescription data;
    data.addCells({
        CellDescription()
            .setId(1)
            .setPos({10.0f, 10.0f})
            .setEnergy(_parameters.cellNormalEnergy[0] * 3)
            .setMaxConnections(2)
            .setExecutionOrderNumber(0)
            .setCellFunction(ConstructorDescription().setGenomeCurrentNodeIndex(2).setGenome(genome)),
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
    _simController->calcTimesteps(1);
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
    auto genome = GenomeDescriptionService::convertDescriptionToBytes(GenomeDescription()
                                                                .setHeader(GenomeHeaderDescription().setSeparateConstruction(false))
                                                                .setCells({CellGenomeDescription(), CellGenomeDescription(), CellGenomeDescription()}));

    DataDescription data;
    data.addCells({
        CellDescription()
            .setId(1)
            .setPos({10.0f, 10.0f})
            .setEnergy(_parameters.cellNormalEnergy[0] * 3)
            .setMaxConnections(2)
            .setExecutionOrderNumber(0)
            .setCellFunction(ConstructorDescription().setGenomeCurrentNodeIndex(2).setGenome(genome)),
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
    _simController->calcTimesteps(1);
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
    auto genome = GenomeDescriptionService::convertDescriptionToBytes(
        GenomeDescription().setHeader(GenomeHeaderDescription().setSeparateConstruction(false).setNumBranches(1)).setCells({CellGenomeDescription()}));

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
    _simController->calcTimesteps(1);
    auto actualData = _simController->getSimulationData();
    auto actualHostCell = getCell(actualData, 1);

    bool found = false;
    for (auto const& connection : actualHostCell.connections) {
        if (connection.cellId != 1 && connection.cellId != 2) {
            EXPECT_TRUE(lowPrecisionCompare(1.0f, connection.distance));
            found = true;
        }
    }
    EXPECT_TRUE(found);
}

TEST_F(ConstructorTests, constructFourthCell_noOverlappingConnection)
{
    auto genome = GenomeDescriptionService::convertDescriptionToBytes(GenomeDescription()
                                                                .setHeader(GenomeHeaderDescription().setSeparateConstruction(false))
            .setCells({CellGenomeDescription(), CellGenomeDescription(), CellGenomeDescription(), CellGenomeDescription(), CellGenomeDescription()}));

    DataDescription data;
    data.addCells({
        CellDescription()
            .setId(1)
            .setPos({10.0f, 10.0f})
            .setEnergy(_parameters.cellNormalEnergy[0] * 3)
            .setMaxConnections(2)
            .setExecutionOrderNumber(0)
            .setCellFunction(ConstructorDescription().setGenomeCurrentNodeIndex(4).setGenome(genome)),
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
    _simController->calcTimesteps(1);
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

TEST_F(ConstructorTests, constructLastCellFirstRepetition)
{
    auto genome = GenomeDescriptionService::convertDescriptionToBytes(
        GenomeDescription().setHeader(GenomeHeaderDescription().setNumRepetitions(2)).setCells({CellGenomeDescription(), CellGenomeDescription()}));

    DataDescription data;
    data.addCells({
        CellDescription()
            .setId(1)
            .setPos({10.0f, 10.0f})
            .setEnergy(_parameters.cellNormalEnergy[0] * 3)
            .setMaxConnections(1)
            .setExecutionOrderNumber(0)
            .setCellFunction(ConstructorDescription().setGenome(genome).setGenomeCurrentNodeIndex(1)),
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
    _simController->calcTimesteps(1);
    auto actualData = _simController->getSimulationData();

    ASSERT_EQ(3, actualData.cells.size());
    auto actualConstructedCell = getOtherCell(actualData, {1, 2});

    EXPECT_EQ(LivingState_UnderConstruction, actualConstructedCell.livingState);
}

TEST_F(ConstructorTests, constructLastCellLastRepetition)
{
    auto genome = GenomeDescriptionService::convertDescriptionToBytes(
        GenomeDescription().setHeader(GenomeHeaderDescription().setNumRepetitions(2)).setCells({CellGenomeDescription(), CellGenomeDescription()}));

    DataDescription data;
    data.addCells({
        CellDescription()
            .setId(1)
            .setPos({10.0f, 10.0f})
            .setEnergy(_parameters.cellNormalEnergy[0] * 3)
            .setMaxConnections(1)
            .setExecutionOrderNumber(0)
            .setCellFunction(ConstructorDescription().setGenome(genome).setGenomeCurrentNodeIndex(1).setGenomeCurrentRepetition(1)),
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
            .setPos({9.0f - _parameters.cellFunctionConstructorOffspringDistance[0], 10.0f})
            .setEnergy(100)
            .setMaxConnections(1)
            .setExecutionOrderNumber(5)
            .setCellFunction(NerveDescription())
            .setLivingState(LivingState_UnderConstruction),
    });
    data.addConnection(1, 2);
    data.addConnection(2, 3);

    _simController->setSimulationData(data);
    _simController->calcTimesteps(1);
    auto actualData = _simController->getSimulationData();

    ASSERT_EQ(4, actualData.cells.size());
    auto actualConstructedCell = getOtherCell(actualData, {1, 2, 3});

    EXPECT_EQ(LivingState_Activating, actualConstructedCell.livingState);
}

TEST_F(ConstructorTests, restartIfNoLastConstructedCellFound)
{
    auto genome =
        GenomeDescriptionService::convertDescriptionToBytes(GenomeDescription()
                                                                  .setHeader(GenomeHeaderDescription().setNumRepetitions(2))
                                                                  .setCells({CellGenomeDescription(), CellGenomeDescription(), CellGenomeDescription()}));

    DataDescription data;
    data.addCells({
        CellDescription()
            .setId(1)
            .setPos({10.0f, 10.0f})
            .setEnergy(_parameters.cellNormalEnergy[0] * 3)
            .setMaxConnections(1)
            .setExecutionOrderNumber(0)
            .setCellFunction(ConstructorDescription().setGenome(genome).setGenomeCurrentNodeIndex(1).setGenomeCurrentRepetition(1)),
        CellDescription()
            .setId(2)
            .setPos({10.0f - _parameters.cellFunctionConstructorOffspringDistance[0], 10.0f})
            .setEnergy(100)
            .setMaxConnections(1)
            .setExecutionOrderNumber(5)
            .setCellFunction(NerveDescription())
            .setActivity({1, 0, 0, 0, 0, 0, 0, 0}),
    });
    data.addConnection(1, 2);

    _simController->setSimulationData(data);
    _simController->calcTimesteps(1);
    auto actualData = _simController->getSimulationData();

    ASSERT_EQ(3, actualData.cells.size());
    auto actualHostCell = getCell(actualData, 1);

    auto actualConstructor = std::get<ConstructorDescription>(*actualHostCell.cellFunction);
    EXPECT_EQ(1, actualConstructor.genomeCurrentNodeIndex);
    EXPECT_EQ(0, actualConstructor.genomeCurrentRepetition);
}

TEST_F(ConstructorTests, restartIfLastConstructedCellHasLowNumConnections)
{
    auto genome =
        GenomeDescriptionService::convertDescriptionToBytes(GenomeDescription()
                                                                  .setHeader(GenomeHeaderDescription().setNumRepetitions(2))
                                                                  .setCells({CellGenomeDescription(), CellGenomeDescription(), CellGenomeDescription()}));

    DataDescription data;
    data.addCells({
        CellDescription()
            .setId(1)
            .setPos({10.0f, 10.0f})
            .setEnergy(_parameters.cellNormalEnergy[0] * 3)
            .setMaxConnections(1)
            .setExecutionOrderNumber(0)
            .setCellFunction(ConstructorDescription().setGenome(genome).setGenomeCurrentNodeIndex(1).setGenomeCurrentRepetition(1)),
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
    _simController->calcTimesteps(1);
    auto actualData = _simController->getSimulationData();

    ASSERT_EQ(3, actualData.cells.size());
    auto actualHostCell = getCell(actualData, 1);

    auto actualConstructor = std::get<ConstructorDescription>(*actualHostCell.cellFunction);
    EXPECT_EQ(1, actualConstructor.genomeCurrentNodeIndex);
    EXPECT_EQ(0, actualConstructor.genomeCurrentRepetition);
}

TEST_F(ConstructorTests, allowLargeConstructionAngle1)
{
    auto genome =
        GenomeDescriptionService::convertDescriptionToBytes(GenomeDescription().setHeader(GenomeHeaderDescription()).setCells({CellGenomeDescription()}));

    DataDescription data;
    data.addCells({
        CellDescription()
            .setId(1)
            .setPos({10.0f, 10.0f})
            .setEnergy(_parameters.cellNormalEnergy[0] * 3)
            .setMaxConnections(2)
            .setExecutionOrderNumber(0)
            .setCellFunction(ConstructorDescription().setGenome(genome).setConstructionAngle1(180.0f)),
        CellDescription().setId(2).setPos({11.0f, 9.0f}).setEnergy(100).setMaxConnections(1).setExecutionOrderNumber(5),
        CellDescription().setId(3).setPos({11.0f, 11.0f}).setEnergy(100).setMaxConnections(1).setExecutionOrderNumber(5),
    });
    data.addConnection(1, 2);
    data.addConnection(1, 3);

    _simController->setSimulationData(data);
    _simController->calcTimesteps(1);
    auto actualData = _simController->getSimulationData();

    ASSERT_EQ(4, actualData.cells.size());
    auto actualConstructedCell = getOtherCell(actualData, {1, 2, 3});

    EXPECT_TRUE(approxCompare(11.0f, actualConstructedCell.pos.x));
    EXPECT_TRUE(approxCompare(10.0f, actualConstructedCell.pos.y));
}

TEST_F(ConstructorTests, allowLargeConstructionAngle2)
{
    auto genome =
        GenomeDescriptionService::convertDescriptionToBytes(GenomeDescription().setHeader(GenomeHeaderDescription()).setCells({CellGenomeDescription()}));

    DataDescription data;
    data.addCells({
        CellDescription()
            .setId(1)
            .setPos({10.0f, 10.0f})
            .setEnergy(_parameters.cellNormalEnergy[0] * 3)
            .setMaxConnections(2)
            .setExecutionOrderNumber(0)
            .setCellFunction(ConstructorDescription().setGenome(genome).setConstructionAngle1(-180.0f)),
        CellDescription().setId(2).setPos({11.0f, 9.0f}).setEnergy(100).setMaxConnections(1).setExecutionOrderNumber(5),
        CellDescription().setId(3).setPos({11.0f, 11.0f}).setEnergy(100).setMaxConnections(1).setExecutionOrderNumber(5),
    });
    data.addConnection(1, 2);
    data.addConnection(1, 3);

    _simController->setSimulationData(data);
    _simController->calcTimesteps(1);
    auto actualData = _simController->getSimulationData();

    ASSERT_EQ(4, actualData.cells.size());
    auto actualConstructedCell = getOtherCell(actualData, {1, 2, 3});

    EXPECT_TRUE(approxCompare(11.0f, actualConstructedCell.pos.x));
    EXPECT_TRUE(approxCompare(10.0f, actualConstructedCell.pos.y));
}

TEST_F(ConstructorTests, repetitionsAndBranches)
{
    auto genome = GenomeDescriptionService::convertDescriptionToBytes(
        GenomeDescription()
            .setHeader(GenomeHeaderDescription().setNumBranches(3).setNumRepetitions(4).setSeparateConstruction(false))
            .setCells({CellGenomeDescription(), CellGenomeDescription(), CellGenomeDescription()}));

    DataDescription data;
    data.addCells({
        CellDescription()
            .setId(1)
            .setPos({10.0f, 10.0f})
            .setEnergy(_parameters.cellNormalEnergy[0] * 2 * 3 * 4 * 3)
            .setMaxConnections(6)
            .setExecutionOrderNumber(0)
            .setCellFunction(ConstructorDescription().setGenome(genome).setActivationMode(20)),
    });

    _simController->setSimulationData(data);
    _simController->calcTimesteps(13 * 3 * 4 * 3 * 20);
    auto actualData = _simController->getSimulationData();

    ASSERT_EQ(1 + 3 * 4 * 3, actualData.cells.size());
    auto actualConstructor = getCell(actualData, 1);

    EXPECT_EQ(3, actualConstructor.connections.size());
}
