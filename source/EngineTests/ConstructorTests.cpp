#include <gtest/gtest.h>

#include <boost/range/combine.hpp>

#include "Base/Math.h"
#include "Base/NumberGenerator.h"
#include "EngineInterface/GenomeConstants.h"
#include "EngineInterface/DescriptionEditService.h"
#include "EngineInterface/Descriptions.h"
#include "EngineInterface/GenomeDescriptionConverterService.h"
#include "EngineInterface/SimulationFacade.h"
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
            result.emplace_back(static_cast<uint8_t>(NumberGenerator::get().getRandomInt(256)));
        }
        return result;
    }

    float getOffspringDistance() const
    {
        return 1.0f + 0.6f;  //0.5 = default connection distance
    }
};

TEST_F(ConstructorTests, noEnergy)
{
    auto genome = GenomeDescriptionConverterService::get().convertDescriptionToBytes(
        GenomeDescription().setHeader(GenomeHeaderDescription().setSeparateConstruction(false)).setCells({CellGenomeDescription()}));

    DataDescription data;
    data.addCell(CellDescription()
                     .setId(1)
                     .setEnergy(_parameters.cellNormalEnergy[0] * 2 - 1.0f)
                     .setCellTypeData(ConstructorDescription().setGenome(genome)));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);
    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(1, actualData.cells.size());
    auto actualHostCell = getCell(actualData, 1);

    EXPECT_EQ(0, actualHostCell.connections.size());
    EXPECT_EQ(0, std::get<ConstructorDescription>(actualHostCell.cellTypeData).genomeCurrentNodeIndex);
    EXPECT_TRUE(approxCompare(_parameters.cellNormalEnergy[0] * 2 - 1.0f, actualHostCell.energy));
    EXPECT_TRUE(approxCompare(0.0f, actualHostCell.signal->channels[0]));
}

TEST_F(ConstructorTests, alreadyFinished)
{
    DataDescription data;

    auto genome = GenomeDescriptionConverterService::get().convertDescriptionToBytes(
        GenomeDescription().setHeader(GenomeHeaderDescription().setNumBranches(1)).setCells({CellGenomeDescription()}));

    auto constructor = ConstructorDescription().setGenome(genome).setCurrentBranch(1);

    data.addCell(
        CellDescription()
            .setId(1)
            .setEnergy(_parameters.cellNormalEnergy[0] * 3)
            .setCellTypeData(constructor));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);
    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(1, actualData.cells.size());
    auto actualHostCell = getCell(actualData, 1);
    auto actualConstructor = std::get<ConstructorDescription>(actualHostCell.cellTypeData);
    EXPECT_EQ(0, actualHostCell.connections.size());
    EXPECT_EQ(0, actualConstructor.genomeCurrentNodeIndex);
    EXPECT_TRUE(approxCompare(0.0f, actualHostCell.signal->channels[0]));
}

TEST_F(ConstructorTests, notActivated)
{
    DataDescription data;

    auto genome = GenomeDescriptionConverterService::get().convertDescriptionToBytes(
        GenomeDescription().setHeader(GenomeHeaderDescription().setNumBranches(1)).setCells({CellGenomeDescription()}));
    auto constructor = ConstructorDescription().setGenome(genome);

    data.addCell(CellDescription()
                     .setId(1)
                     .setEnergy(_parameters.cellNormalEnergy[0] * 3)
                     .setCellTypeData(constructor)
                     .setActivationTime(2));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);
    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(1, actualData.cells.size());
    auto actualHostCell = getCell(actualData, 1);
    auto actualConstructor = std::get<ConstructorDescription>(actualHostCell.cellTypeData);
    EXPECT_TRUE(approxCompare(0.0f, actualHostCell.signal->channels[0]));
}

TEST_F(ConstructorTests, manualConstruction_noInputSignal)
{
    auto genome = GenomeDescriptionConverterService::get().convertDescriptionToBytes(GenomeDescription().setCells({CellGenomeDescription()}));

    DataDescription data;
    data.addCell(CellDescription()
                     .setId(1)
                     .setEnergy(_parameters.cellNormalEnergy[0] * 3)
                     .setCellTypeData(ConstructorDescription().setAutoTriggerInterval(0).setGenome(genome)));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);
    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(1, actualData.cells.size());
    auto actualHostCell = getCell(actualData, 1);

    EXPECT_EQ(0, actualHostCell.connections.size());
    EXPECT_EQ(0, std::get<ConstructorDescription>(actualHostCell.cellTypeData).genomeCurrentNodeIndex);
    EXPECT_TRUE(approxCompare(_parameters.cellNormalEnergy[0] * 3, actualHostCell.energy));
    EXPECT_TRUE(approxCompare(0.0f, actualHostCell.signal->channels[0]));
}

TEST_F(ConstructorTests, constructFirstCell_correctCycle)
{
    auto genome = GenomeDescriptionConverterService::get().convertDescriptionToBytes(GenomeDescription().setCells({CellGenomeDescription()}));

    _simulationFacade->calcTimesteps(1);

    DataDescription data;
    data.addCell(CellDescription()
                     .setId(1)
                     .setEnergy(_parameters.cellNormalEnergy[0] * 3)
                     .setCellTypeData(ConstructorDescription().setAutoTriggerInterval(3 * 6).setGenome(genome)));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(3 * 6);
    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(2, actualData.cells.size());
}

TEST_F(ConstructorTests, constructFirstCell_oneCellGenome_infiniteRepetitions)
{
    auto genome = GenomeDescriptionConverterService::get().convertDescriptionToBytes(
        GenomeDescription().setHeader(GenomeHeaderDescription().setInfiniteRepetitions()).setCells({CellGenomeDescription()}));

    _simulationFacade->calcTimesteps(1);

    DataDescription data;
    data.addCell(CellDescription()
                     .setId(1)
                     .setEnergy(_parameters.cellNormalEnergy[0] * 3)
                     .setCellTypeData(ConstructorDescription().setAutoTriggerInterval(3 * 6).setGenome(genome)));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(3 * 6);
    auto actualData = _simulationFacade->getSimulationData();
    ASSERT_EQ(2, actualData.cells.size());

    auto actualHostCell = getCell(actualData, 1);
    auto actualConstructedCell = getOtherCell(actualData, {1});
    EXPECT_EQ(0, std::get<ConstructorDescription>(actualHostCell.cellTypeData).genomeCurrentRepetition);
    EXPECT_EQ(0, std::get<ConstructorDescription>(actualHostCell.cellTypeData).currentBranch);
    EXPECT_EQ(LivingState_Activating, actualConstructedCell.livingState);
}

TEST_F(ConstructorTests, constructFirstCell_twoCellGenome_infiniteRepetitions)
{
    auto genome = GenomeDescriptionConverterService::get().convertDescriptionToBytes(
        GenomeDescription().setHeader(GenomeHeaderDescription().setInfiniteRepetitions()).setCells({CellGenomeDescription(), CellGenomeDescription()}));

    _simulationFacade->calcTimesteps(1);

    DataDescription data;
    data.addCell(CellDescription()
                     .setId(1)
                     .setEnergy(_parameters.cellNormalEnergy[0] * 3)
                     .setCellTypeData(ConstructorDescription().setAutoTriggerInterval(3 * 6).setGenome(genome)));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);
    auto actualData = _simulationFacade->getSimulationData();
    ASSERT_EQ(2, actualData.cells.size());

    auto actualHostCell = getCell(actualData, 1);
    auto actualConstructedCell = getOtherCell(actualData, {1});
    EXPECT_EQ(0, std::get<ConstructorDescription>(actualHostCell.cellTypeData).genomeCurrentRepetition);
    EXPECT_EQ(0, std::get<ConstructorDescription>(actualHostCell.cellTypeData).currentBranch);
    EXPECT_EQ(LivingState_UnderConstruction, actualConstructedCell.livingState);
}

TEST_F(ConstructorTests, constructFirstCell_wrongCycle)
{
    auto genome = GenomeDescriptionConverterService::get().convertDescriptionToBytes(
        GenomeDescription().setCells({CellGenomeDescription()}));

    _simulationFacade->calcTimesteps(1);

    DataDescription data;
    data.addCell(CellDescription()
                     .setId(1)
                     .setEnergy(_parameters.cellNormalEnergy[0] * 3)
                     .setCellTypeData(ConstructorDescription().setAutoTriggerInterval(3 * 6).setGenome(genome)));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(3 * 6 - 1);
    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(1, actualData.cells.size());
}

TEST_F(ConstructorTests, constructFirstCell_completenessCheck_constructionNotBuilt)
{
    auto constructorGenome = ConstructorGenomeDescription().setMode(0).setConstructionActivationTime(123).setMakeSelfCopy();
    auto genome =
        GenomeDescriptionConverterService::get().convertDescriptionToBytes(GenomeDescription().setCells({CellGenomeDescription().setCellTypeData(constructorGenome)}));
    auto otherGenome = GenomeDescriptionConverterService::get().convertDescriptionToBytes(GenomeDescription().setCells({CellGenomeDescription()}));

    DataDescription data;
    data.addCells({
        CellDescription()
            .setId(1)
            .setPos({10.0f, 10.0f})
            .setEnergy(_parameters.cellNormalEnergy[0] * 3)
            .setCellTypeData(ConstructorDescription().setGenome(genome).setNumInheritedGenomeNodes(4)),
        CellDescription().setId(2).setPos({11.0f, 10.0f}).setEnergy(100).setCellTypeData(OscillatorDescription()),
        CellDescription()
            .setId(3)
            .setPos({12.0f, 10.0f})
            .setEnergy(100)
            .setCellTypeData(ConstructorDescription().setGenome(otherGenome)),
    });
    data.addConnection(1, 2);
    data.addConnection(2, 3);

    _parameters.cellTypeConstructorCheckCompletenessForSelfReplication = true;
    _simulationFacade->setSimulationParameters(_parameters);
    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);
    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(3, actualData.cells.size());
}

TEST_F(ConstructorTests, constructFirstCell_completenessCheck_repeatedConstructionNotBuilt)
{
    auto constructorGenome = ConstructorGenomeDescription().setMode(0).setConstructionActivationTime(123).setMakeSelfCopy();
    auto genome =
        GenomeDescriptionConverterService::get().convertDescriptionToBytes(GenomeDescription().setCells({CellGenomeDescription().setCellTypeData(constructorGenome)}));
    auto otherGenome = GenomeDescriptionConverterService::get().convertDescriptionToBytes(
        GenomeDescription().setHeader(GenomeHeaderDescription().setNumRepetitions(2)).setCells({CellGenomeDescription()}));

    DataDescription data;
    data.addCells({
        CellDescription()
            .setId(1)
            .setPos({10.0f, 10.0f})
            .setEnergy(_parameters.cellNormalEnergy[0] * 3)
            .setCellTypeData(ConstructorDescription().setGenome(genome).setNumInheritedGenomeNodes(5)),
        CellDescription().setId(2).setPos({11.0f, 10.0f}).setEnergy(100).setCellTypeData(OscillatorDescription()),
        CellDescription()
            .setId(3)
            .setPos({12.0f, 10.0f})
            .setEnergy(100)
            .setCellTypeData(ConstructorDescription().setGenome(otherGenome).setGenomeCurrentRepetition(1)),
        CellDescription().setId(4).setPos({10.0f, 11.0f}).setEnergy(100),
    });
    data.addConnection(1, 2);
    data.addConnection(2, 3);
    data.addConnection(3, 4);

    _parameters.cellTypeConstructorCheckCompletenessForSelfReplication = true;
    _simulationFacade->setSimulationParameters(_parameters);
    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);
    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(4, actualData.cells.size());
}

TEST_F(ConstructorTests, constructFirstCell_completenessCheck_constructionBuilt)
{
    auto otherGenome = GenomeDescriptionConverterService::get().convertDescriptionToBytes(GenomeDescription().setCells({CellGenomeDescription()}));
    auto otherConstructorGenome = ConstructorGenomeDescription().setMode(0).setConstructionActivationTime(123).setGenome(otherGenome);

    auto constructorGenome = ConstructorGenomeDescription().setMode(0).setConstructionActivationTime(123).setMakeSelfCopy();
    auto genome = GenomeDescriptionConverterService::get().convertDescriptionToBytes(GenomeDescription().setCells(
        {CellGenomeDescription().setCellTypeData(constructorGenome), CellGenomeDescription(), CellGenomeDescription().setCellTypeData(otherConstructorGenome)}));

    DataDescription data;
    data.addCells({
        CellDescription()
            .setId(1)
            .setPos({10.0f, 10.0f})
            .setEnergy(_parameters.cellNormalEnergy[0] * 3)
            .setCellTypeData(ConstructorDescription().setGenome(genome).setNumInheritedGenomeNodes(4)),
        CellDescription().setId(2).setPos({11.0f, 10.0f}).setEnergy(100).setCellTypeData(OscillatorDescription()),
        CellDescription()
            .setId(3)
            .setPos({12.0f, 10.0f})
            .setEnergy(100)
            .setCellTypeData(ConstructorDescription().setGenome(otherGenome).setCurrentBranch(1)),
        CellDescription().setId(4).setPos({10.0f, 11.0f}).setEnergy(100),
    });
    data.addConnection(1, 2);
    data.addConnection(2, 3);
    data.addConnection(3, 4);

    _parameters.cellTypeConstructorCheckCompletenessForSelfReplication = true;
    _simulationFacade->setSimulationParameters(_parameters);
    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);
    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(5, actualData.cells.size());
}

TEST_F(ConstructorTests, constructFirstCell_completenessCheck_infiniteConstructionsBuilt)
{
    auto otherGenome = GenomeDescriptionConverterService::get().convertDescriptionToBytes(
        GenomeDescription().setHeader(GenomeHeaderDescription().setInfiniteRepetitions()).setCells({CellGenomeDescription()}));
    auto otherConstructorGenome = ConstructorGenomeDescription().setMode(0).setConstructionActivationTime(123).setGenome(otherGenome);

    auto constructorGenome = ConstructorGenomeDescription().setMode(0).setConstructionActivationTime(123).setMakeSelfCopy();
    auto genome = GenomeDescriptionConverterService::get().convertDescriptionToBytes(GenomeDescription().setCells(
        {CellGenomeDescription().setCellTypeData(constructorGenome), CellGenomeDescription(), CellGenomeDescription().setCellTypeData(otherConstructorGenome)}));

    DataDescription data;
    data.addCells({
        CellDescription()
            .setId(1)
            .setPos({10.0f, 10.0f})
            .setEnergy(_parameters.cellNormalEnergy[0] * 3)
            .setCellTypeData(ConstructorDescription().setGenome(genome).setNumInheritedGenomeNodes(4)),
        CellDescription().setId(2).setPos({11.0f, 10.0f}).setEnergy(100).setCellTypeData(OscillatorDescription()),
        CellDescription()
            .setId(3)
            .setPos({12.0f, 10.0f})
            .setEnergy(100)
            .setCellTypeData(ConstructorDescription().setGenome(otherGenome).setCurrentBranch(1)),
        CellDescription().setId(4).setPos({10.0f, 11.0f}).setEnergy(100),
    });
    data.addConnection(1, 2);
    data.addConnection(2, 3);
    data.addConnection(3, 4);

    _parameters.cellTypeConstructorCheckCompletenessForSelfReplication = true;
    _simulationFacade->setSimulationParameters(_parameters);
    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);
    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(5, actualData.cells.size());
}

TEST_F(ConstructorTests, constructFirstCell_completenessCheck_largeCluster)
{
    auto constexpr RectLength = 50;
    auto rect = DescriptionEditService::get().createRect(DescriptionEditService::CreateRectParameters().height(RectLength).width(RectLength));

    auto constructorGenome = ConstructorGenomeDescription().setMode(0).setConstructionActivationTime(123).setMakeSelfCopy();
    auto genome =
        GenomeDescriptionConverterService::get().convertDescriptionToBytes(GenomeDescription().setCells({CellGenomeDescription().setCellTypeData(constructorGenome)}));
    auto otherGenome = GenomeDescriptionConverterService::get().convertDescriptionToBytes(GenomeDescription().setCells({CellGenomeDescription()}));

    auto& cell1 = rect.cells.at(0);
    cell1.setEnergy(_parameters.cellNormalEnergy[0] * 3)
        .setCellTypeData(ConstructorDescription().setGenome(genome).setNumInheritedGenomeNodes(RectLength * RectLength));

    _parameters.cellTypeConstructorCheckCompletenessForSelfReplication = true;
    _simulationFacade->setSimulationParameters(_parameters);
    _simulationFacade->setSimulationData(rect);
    _simulationFacade->calcTimesteps(1);
    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(RectLength * RectLength + 1, actualData.cells.size());
}

TEST_F(ConstructorTests, constructFirstCell_completenessCheck_thinCluster)
{
    auto constructorGenome = ConstructorGenomeDescription().setMode(0).setConstructionActivationTime(123).setMakeSelfCopy();
    auto genome = GenomeDescriptionConverterService::get().convertDescriptionToBytes(GenomeDescription()
                                                                          .setHeader(GenomeHeaderDescription().setNumBranches(2))
                                                                          .setCells(
                                                                              {CellGenomeDescription(),
                                                                               CellGenomeDescription(),
                                                                               CellGenomeDescription().setCellTypeData(constructorGenome),
                                                                               CellGenomeDescription(),
                                                                               CellGenomeDescription()}));

    DataDescription data;
    data.addCells({
        CellDescription()
            .setId(1)
            .setPos({10.0f, 10.0f})
            .setEnergy(_parameters.cellNormalEnergy[0] * 3)
            .setCellTypeData(ConstructorDescription().setGenome(genome).setNumInheritedGenomeNodes(5))
            .setCreatureId(1),
        CellDescription()
            .setId(2)
            .setPos({10.0f, 9.0f})
            .setEnergy(100)
            .setCellTypeData(OscillatorDescription())
            .setCreatureId(1),
        CellDescription()
            .setId(3)
            .setPos({10.0f, 8.0f})
            .setEnergy(100)
            .setCellTypeData(OscillatorDescription())
            .setCreatureId(1),
        CellDescription()
            .setId(4)
            .setPos({10.0f, 11.0f})
            .setEnergy(100)
            .setCellTypeData(OscillatorDescription())
            .setCreatureId(1),
        CellDescription()
            .setId(5)
            .setPos({10.0f, 12.0f})
            .setEnergy(100)
            .setCellTypeData(OscillatorDescription())
            .setCreatureId(1),
        CellDescription()
            .setId(6)
            .setPos({11.0f, 10.0f})
            .setEnergy(100)
            .setCellTypeData(OscillatorDescription())
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

    _parameters.cellTypeConstructorCheckCompletenessForSelfReplication = true;
    _simulationFacade->setSimulationParameters(_parameters);
    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);
    auto actualData = _simulationFacade->getSimulationData();

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
        GenomeDescriptionConverterService::get().convertDescriptionToBytes(GenomeDescription().setCells({CellGenomeDescription().setCellTypeData(constructorGenome)}));
    auto otherGenome = GenomeDescriptionConverterService::get().convertDescriptionToBytes(GenomeDescription().setCells({CellGenomeDescription()}));

    DataDescription data;
    data.addCells({
        CellDescription()
            .setId(1)
            .setPos({10.0f, 10.0f})
            .setEnergy(_parameters.cellNormalEnergy[0] * 3)
            .setCellTypeData(ConstructorDescription().setGenome(genome)),
        CellDescription()
            .setId(2)
            .setPos({11.0f, 10.0f})
            .setEnergy(100)
            .setCellTypeData(ConstructorDescription().setGenome(otherGenome).setCurrentBranch(1)),
        CellDescription()
            .setId(3)
            .setPos({12.0f, 10.0f})
            .setEnergy(100)
            .setLivingState(LivingState_UnderConstruction)
            .setCellTypeData(ConstructorDescription().setGenome(otherGenome).setCurrentBranch(0)),
    });
    data.addConnection(1, 2);
    data.addConnection(2, 3);

    _parameters.cellTypeConstructorCheckCompletenessForSelfReplication = true;
    _simulationFacade->setSimulationParameters(_parameters);
    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);
    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(3, actualData.cells.size());
}

TEST_F(ConstructorTests, constructFirstCell_noSeparation)
{
    auto genome = GenomeDescriptionConverterService::get().convertDescriptionToBytes(
        GenomeDescription()
            .setHeader(GenomeHeaderDescription().setSeparateConstruction(false).setStiffness(0.35f))
            .setCells({CellGenomeDescription().setCellTypeData(ConstructorGenomeDescription().setMakeSelfCopy()).setColor(2)}));

    DataDescription data;
    data.addCell(
        CellDescription()
            .setId(1)
            .setPos({10.0f, 10.0f})
            .setEnergy(_parameters.cellNormalEnergy[0] * 3)
            .setCellTypeData(ConstructorDescription().setGenome(genome).setConstructionActivationTime(123)));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);
    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(2, actualData.cells.size());
    auto actualHostCell = getCell(actualData, 1);
    auto actualConstructedCell = getOtherCell(actualData, 1);

    EXPECT_EQ(1, actualHostCell.connections.size());

    auto const& actualConstructor = std::get<ConstructorDescription>(actualHostCell.cellTypeData);
    EXPECT_EQ(0, actualConstructor.genomeCurrentNodeIndex);
    EXPECT_EQ(0, actualConstructor.genomeCurrentRepetition);
    EXPECT_EQ(1, actualConstructor.currentBranch);
    EXPECT_TRUE(approxCompare(_parameters.cellNormalEnergy[0] * 2, actualHostCell.energy));
    EXPECT_TRUE(approxCompare(1.0f, actualHostCell.signal->channels[0]));
    EXPECT_EQ(LivingState_Activating, actualConstructedCell.livingState);

    EXPECT_EQ(1, actualConstructedCell.connections.size());
    EXPECT_EQ(2, actualConstructedCell.color);
    EXPECT_EQ(CellType_Constructor, actualConstructedCell.getCellType());
    EXPECT_EQ(123, actualConstructedCell.activationTime);
    EXPECT_TRUE(approxCompare(0.35f, actualConstructedCell.stiffness, 0.01f));
    EXPECT_TRUE(approxCompare(_parameters.cellNormalEnergy[0], actualConstructedCell.energy));
    EXPECT_TRUE(approxCompare(1.0f, Math::length(actualHostCell.pos - actualConstructedCell.pos)));
}

TEST_F(ConstructorTests, constructFirstCell_notFinished)
{
    auto genome = GenomeDescriptionConverterService::get().convertDescriptionToBytes(
        GenomeDescription()
            .setHeader(GenomeHeaderDescription().setSeparateConstruction(false)).setCells({CellGenomeDescription(), CellGenomeDescription()}));

    DataDescription data;
    data.addCell(CellDescription()
                     .setId(1)
                     .setPos({10.0f, 10.0f})
                     .setEnergy(_parameters.cellNormalEnergy[0] * 3)
                     .setCellTypeData(ConstructorDescription().setGenome(genome)));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);
    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(2, actualData.cells.size());
    auto actualHostCell = getCell(actualData, 1);
    auto actualConstructedCell = getOtherCell(actualData, 1);
    auto const& actualConstructor = std::get<ConstructorDescription>(actualHostCell.cellTypeData);
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
    auto genome = GenomeDescriptionConverterService::get().convertDescriptionToBytes(
        GenomeDescription().setHeader(GenomeHeaderDescription().setSeparateConstruction(true)).setCells({CellGenomeDescription()}));

    DataDescription data;
    data.addCell(CellDescription()
                     .setId(1)
                     .setEnergy(_parameters.cellNormalEnergy[0] * 3)
                     .setCellTypeData(ConstructorDescription()
                                          .setGenome(genome)));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);
    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(2, actualData.cells.size());
    auto actualHostCell = getCell(actualData, 1);
    auto actualConstructedCell = getOtherCell(actualData, 1);

    EXPECT_EQ(0, actualHostCell.connections.size());
    auto const& actualConstructor = std::get<ConstructorDescription>(actualHostCell.cellTypeData);
    EXPECT_EQ(0, actualConstructor.genomeCurrentNodeIndex);
    EXPECT_EQ(0, actualConstructor.genomeCurrentRepetition);
    EXPECT_EQ(0, actualConstructor.currentBranch);

    EXPECT_EQ(0, actualConstructedCell.connections.size());
    EXPECT_EQ(LivingState_Activating, actualConstructedCell.livingState);
}

TEST_F(ConstructorTests, constructFirstCell_manualConstruction)
{
    auto genome = GenomeDescriptionConverterService::get().convertDescriptionToBytes(
        GenomeDescription().setCells({CellGenomeDescription()}));

    DataDescription data;
    data.addCells({
       CellDescription()
            .setId(1)
            .setPos({10.0f, 10.0f})
            .setEnergy(_parameters.cellNormalEnergy[0] * 3)
            .setCellTypeData(ConstructorDescription().setAutoTriggerInterval(0).setGenome(genome)),
        CellDescription()
             .setId(2)
             .setPos({11.0f, 10.0f})
             .setEnergy(100)
             .setCellTypeData(OscillatorDescription())
             .setSignal({1, 0, 0, 0, 0, 0, 0, 0})
    });
    data.addConnection(1, 2);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);
    auto actualData = _simulationFacade->getSimulationData();

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
    auto genome = GenomeDescriptionConverterService::get().convertDescriptionToBytes(GenomeDescription().setCells({CellGenomeDescription()}));

    DataDescription data;
    data.addCells({
        CellDescription()
             .setId(1)
             .setPos({10.0f, 10.0f})
             .setEnergy(_parameters.cellNormalEnergy[0] * 3)
             .setCellTypeData(ConstructorDescription().setAutoTriggerInterval(0).setGenome(genome).setConstructionAngle1(90.0f)),
        CellDescription()
             .setId(2)
             .setPos({11.0f, 10.0f})
             .setEnergy(100)
             .setCellTypeData(OscillatorDescription())
             .setSignal({1, 0, 0, 0, 0, 0, 0, 0})});
    data.addConnection(1, 2);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);
    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(3, actualData.cells.size());
    auto actualHostCell = getCell(actualData, 1);
    auto actualConstructedCell = getOtherCell(actualData, {1, 2});

    EXPECT_TRUE(approxCompare(10.0f, actualConstructedCell.pos.x));
    EXPECT_TRUE(approxCompare(10.0f - 1.0f, actualConstructedCell.pos.y));
}

TEST_F(ConstructorTests, constructFirstCell_differentAngle2)
{
    auto genome = GenomeDescriptionConverterService::get().convertDescriptionToBytes(GenomeDescription().setCells({CellGenomeDescription().setReferenceAngle(-90.0f)}));

    DataDescription data;
    data.addCells(
        {CellDescription()
             .setId(1)
             .setPos({10.0f, 10.0f})
             .setEnergy(_parameters.cellNormalEnergy[0] * 3)
             .setCellTypeData(ConstructorDescription().setAutoTriggerInterval(0).setGenome(genome).setConstructionAngle1(-90.0f)),
         CellDescription()
             .setId(2)
             .setPos({11.0f, 10.0f})
             .setEnergy(100)
             .setCellTypeData(OscillatorDescription())
             .setSignal({1, 0, 0, 0, 0, 0, 0, 0})});
    data.addConnection(1, 2);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);
    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(3, actualData.cells.size());
    auto actualHostCell = getCell(actualData, 1);
    auto actualConstructedCell = getOtherCell(actualData, {1, 2});

    EXPECT_TRUE(approxCompare(10.0f, actualConstructedCell.pos.x));
    EXPECT_TRUE(approxCompare(10.0f + 1.0f, actualConstructedCell.pos.y));
}

TEST_F(ConstructorTests, constructNeuronCell)
{
    auto nn = NeuralNetworkGenomeDescription();
    nn.setWeight(1, 7, 3.9f);
    nn.setWeight(7, 1, -1.9f);
    nn.biases[3] = 3.8f;

    auto genome = GenomeDescriptionConverterService::get().convertDescriptionToBytes(GenomeDescription().setCells({CellGenomeDescription().setNeuralNetwork(nn)}));

    DataDescription data;
    data.addCell(
        CellDescription()
                     .setId(1)
                     .setEnergy(_parameters.cellNormalEnergy[0] * 3)
                     .setCellTypeData(ConstructorDescription().setGenome(genome)));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);
    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(2, actualData.cells.size());
    auto actualConstructedCell = getOtherCell(actualData, 1);

    EXPECT_EQ(CellType_Base, actualConstructedCell.getCellType());

    for (auto const& [weight, actualWeight] : boost::combine(nn.weights, actualConstructedCell.neuralNetwork->weights)) {
        EXPECT_TRUE(lowPrecisionCompare(weight, actualWeight));
    }
    for (int i = 0; i < MAX_CHANNELS; ++i) {
        EXPECT_TRUE(lowPrecisionCompare(nn.biases[i], actualConstructedCell.neuralNetwork->biases[i]));
    }
}

TEST_F(ConstructorTests, constructConstructorCell)
{
    auto constructorGenome = ConstructorGenomeDescription().setMode(0).setConstructionActivationTime(123).setGenome(createRandomGenome(MAX_GENOME_BYTES / 2));

    auto genome =
        GenomeDescriptionConverterService::get().convertDescriptionToBytes(GenomeDescription().setCells({CellGenomeDescription().setCellTypeData(constructorGenome)}));

    DataDescription data;

    data.addCell(CellDescription()
                     .setId(1)
                     .setEnergy(_parameters.cellNormalEnergy[0] * 3)
                     .setCellTypeData(ConstructorDescription().setGenome(genome)));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);
    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(2, actualData.cells.size());
    auto actualConstructedCell = getOtherCell(actualData, 1);

    EXPECT_EQ(CellType_Constructor, actualConstructedCell.getCellType());

    auto actualConstructor = std::get<ConstructorDescription>(actualConstructedCell.cellTypeData);
    EXPECT_EQ(constructorGenome.autoTriggerInterval, actualConstructor.autoTriggerInterval);
    EXPECT_EQ(constructorGenome.constructionActivationTime, actualConstructor.constructionActivationTime);
    EXPECT_EQ(constructorGenome.getGenomeData(), actualConstructor.genome);
}

TEST_F(ConstructorTests, constructOscillatorCell)
{
    auto oscillatorDesc = OscillatorGenomeDescription().setPulseMode(2).setAlternationMode(4);
    auto genome = GenomeDescriptionConverterService::get().convertDescriptionToBytes(GenomeDescription().setCells({CellGenomeDescription().setCellTypeData(oscillatorDesc)}));

    DataDescription data;
    data.addCell(CellDescription()
                     .setId(1)
                     .setEnergy(_parameters.cellNormalEnergy[0] * 3)
                     .setCellTypeData(ConstructorDescription().setGenome(genome)));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);
    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(2, actualData.cells.size());
    auto actualConstructedCell = getOtherCell(actualData, 1);
    auto actualOscillator = std::get<OscillatorDescription>(actualConstructedCell.cellTypeData);

    EXPECT_EQ(CellType_Oscillator, actualConstructedCell.getCellType());
    EXPECT_EQ(oscillatorDesc.pulseMode, actualOscillator.pulseMode);
    EXPECT_EQ(oscillatorDesc.alternationMode, actualOscillator.alternationMode);
}

TEST_F(ConstructorTests, constructAttackerCell)
{
    auto attackerDesc = AttackerGenomeDescription().setMode(EnergyDistributionMode_TransmittersAndConstructors);
    auto genome = GenomeDescriptionConverterService::get().convertDescriptionToBytes(GenomeDescription().setCells({CellGenomeDescription().setCellTypeData(attackerDesc)}));

    DataDescription data;
    data.addCell(CellDescription()
                     .setId(1)
                     .setEnergy(_parameters.cellNormalEnergy[0] * 3)
                     .setCellTypeData(ConstructorDescription().setGenome(genome)));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);
    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(2, actualData.cells.size());
    auto actualConstructedCell = getOtherCell(actualData, 1);

    EXPECT_EQ(CellType_Attacker, actualConstructedCell.getCellType());

    auto actualAttacker = std::get<AttackerDescription>(actualConstructedCell.cellTypeData);
    EXPECT_EQ(attackerDesc.mode, actualAttacker.mode);
}

TEST_F(ConstructorTests, constructDefenderCell)
{
    auto defenderDesc = DefenderGenomeDescription().setMode(DefenderMode_DefendAgainstInjector);
    auto genome = GenomeDescriptionConverterService::get().convertDescriptionToBytes(GenomeDescription().setCells({CellGenomeDescription().setCellTypeData(defenderDesc)}));

    DataDescription data;
    data.addCell(CellDescription()
                     .setId(1)
                     .setEnergy(_parameters.cellNormalEnergy[0] * 3)
                     .setCellTypeData(ConstructorDescription().setGenome(genome)));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);
    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(2, actualData.cells.size());
    auto actualConstructedCell = getOtherCell(actualData, 1);

    EXPECT_EQ(CellType_Defender, actualConstructedCell.getCellType());

    auto actualDefender = std::get<DefenderDescription>(actualConstructedCell.cellTypeData);
    EXPECT_EQ(defenderDesc.mode, actualDefender.mode);
}

TEST_F(ConstructorTests, constructTransmitterCell)
{
    auto transmitterDesc = DepotGenomeDescription().setMode(EnergyDistributionMode_TransmittersAndConstructors);
    auto genome =
        GenomeDescriptionConverterService::get().convertDescriptionToBytes(GenomeDescription().setCells({CellGenomeDescription().setCellTypeData(transmitterDesc)}));

    DataDescription data;
    data.addCell(CellDescription()
                     .setId(1)
                     .setEnergy(_parameters.cellNormalEnergy[0] * 3)
                     .setCellTypeData(ConstructorDescription().setGenome(genome)));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);
    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(2, actualData.cells.size());
    auto actualConstructedCell = getOtherCell(actualData, 1);

    EXPECT_EQ(CellType_Depot, actualConstructedCell.getCellType());

    auto actualTransmitter = std::get<DepotDescription>(actualConstructedCell.cellTypeData);
    EXPECT_EQ(transmitterDesc.mode, actualTransmitter.mode);
}

TEST_F(ConstructorTests, constructMuscleCell)
{
    auto muscleDesc = MuscleGenomeDescription().setMode(MuscleMode_ContractionExpansion);
    auto genome = GenomeDescriptionConverterService::get().convertDescriptionToBytes(GenomeDescription().setCells({CellGenomeDescription().setCellTypeData(muscleDesc)}));

    DataDescription data;
    data.addCell(CellDescription()
                     .setId(1)
                     .setEnergy(_parameters.cellNormalEnergy[0] * 3)
                     .setCellTypeData(ConstructorDescription().setGenome(genome)));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);
    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(2, actualData.cells.size());
    auto actualConstructedCell = getOtherCell(actualData, 1);

    EXPECT_EQ(CellType_Muscle, actualConstructedCell.getCellType());

    auto actualMuscle = std::get<MuscleDescription>(actualConstructedCell.cellTypeData);
    EXPECT_EQ(muscleDesc.mode, actualMuscle.mode);
    EXPECT_EQ(0, actualMuscle.lastBendingDirection);
    EXPECT_EQ(0.0f, actualMuscle.consecutiveBendingAngle);
}

TEST_F(ConstructorTests, constructSensorCell)
{
    auto sensorDesc = SensorGenomeDescription().setColor(2).setMinDensity(0.5f).setRestrictToMutants(SensorRestrictToMutants_RestrictToFreeCells);
    auto genome = GenomeDescriptionConverterService::get().convertDescriptionToBytes(GenomeDescription().setCells({CellGenomeDescription().setCellTypeData(sensorDesc)}));

    DataDescription data;
    data.addCell(CellDescription()
                     .setId(1)
                     .setEnergy(_parameters.cellNormalEnergy[0] * 3)
                     .setCellTypeData(ConstructorDescription().setGenome(genome)));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);
    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(2, actualData.cells.size());
    auto actualConstructedCell = getOtherCell(actualData, 1);

    EXPECT_EQ(CellType_Sensor, actualConstructedCell.getCellType());

    auto actualSensor = std::get<SensorDescription>(actualConstructedCell.cellTypeData);
    EXPECT_TRUE(lowPrecisionCompare(sensorDesc.minDensity, actualSensor.minDensity));
    EXPECT_EQ(sensorDesc.restrictToColor, actualSensor.restrictToColor);
    EXPECT_EQ(sensorDesc.restrictToMutants, actualSensor.restrictToMutants);
}

TEST_F(ConstructorTests, constructInjectorCell)
{
    auto injectorDesc = InjectorGenomeDescription().setMode(InjectorMode_InjectOnlyEmptyCells).setGenome(createRandomGenome(MAX_GENOME_BYTES / 2));
    auto genome = GenomeDescriptionConverterService::get().convertDescriptionToBytes(GenomeDescription().setCells({CellGenomeDescription().setCellTypeData(injectorDesc)}));

    DataDescription data;
    data.addCell(CellDescription()
                     .setId(1)
                     .setEnergy(_parameters.cellNormalEnergy[0] * 3)
                     .setCellTypeData(ConstructorDescription().setGenome(genome)));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);
    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(2, actualData.cells.size());
    auto actualConstructedCell = getOtherCell(actualData, 1);

    EXPECT_EQ(CellType_Injector, actualConstructedCell.getCellType());

    auto actualInjector = std::get<InjectorDescription>(actualConstructedCell.cellTypeData);
    EXPECT_EQ(injectorDesc.mode, actualInjector.mode);
    EXPECT_EQ(injectorDesc.getGenomeData(), actualInjector.genome);
}

TEST_F(ConstructorTests, constructReconnectorCell)
{
    auto reconnectorDesc = ReconnectorGenomeDescription().setRestrictToColor(2).setRestrictToMutants(ReconnectorRestrictToMutants_RestrictToSameMutants);
    auto genome =
        GenomeDescriptionConverterService::get().convertDescriptionToBytes(GenomeDescription().setCells({CellGenomeDescription().setCellTypeData(reconnectorDesc)}));

    DataDescription data;
    data.addCell(CellDescription()
                     .setId(1)
                     .setEnergy(_parameters.cellNormalEnergy[0] * 3)
                     .setCellTypeData(ConstructorDescription().setGenome(genome)));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);
    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(2, actualData.cells.size());
    auto actualConstructedCell = getOtherCell(actualData, 1);

    EXPECT_EQ(CellType_Reconnector, actualConstructedCell.getCellType());

    auto actualReconnector = std::get<ReconnectorDescription>(actualConstructedCell.cellTypeData);
    EXPECT_EQ(reconnectorDesc.restrictToColor, actualReconnector.restrictToColor);
    EXPECT_EQ(reconnectorDesc.restrictToMutants, actualReconnector.restrictToMutants);
}

TEST_F(ConstructorTests, constructDetonatorCell)
{
    auto detonatorDesc = DetonatorGenomeDescription().setCountDown(25);
    auto genome = GenomeDescriptionConverterService::get().convertDescriptionToBytes(GenomeDescription().setCells({CellGenomeDescription().setCellTypeData(detonatorDesc)}));

    DataDescription data;
    data.addCell(CellDescription()
                     .setId(1)
                     .setEnergy(_parameters.cellNormalEnergy[0] * 3)
                     .setCellTypeData(ConstructorDescription().setGenome(genome)));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);
    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(2, actualData.cells.size());
    auto actualConstructedCell = getOtherCell(actualData, 1);

    EXPECT_EQ(CellType_Detonator, actualConstructedCell.getCellType());

    auto actualDetonator = std::get<DetonatorDescription>(actualConstructedCell.cellTypeData);
    EXPECT_EQ(detonatorDesc.countdown, actualDetonator.countdown);
}

TEST_F(ConstructorTests, constructConstructorCell_nestingGenomeTooLarge)
{
    auto constructedConstructor = ConstructorGenomeDescription().setMode(0).setGenome(createRandomGenome(MAX_GENOME_BYTES));
    auto genome =
        GenomeDescriptionConverterService::get().convertDescriptionToBytes(GenomeDescription().setCells({CellGenomeDescription().setCellTypeData(constructedConstructor)}));


    DataDescription data;

    data.addCell(CellDescription()
                     .setId(1)
                     .setEnergy(_parameters.cellNormalEnergy[0] * 3)
                     .setCellTypeData(ConstructorDescription().setGenome(genome)));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);
    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(2, actualData.cells.size());
    auto actualCell = getOtherCell(actualData, 1);
    auto actualConstructedCell = getOtherCell(actualData, 1);

    EXPECT_EQ(CellType_Constructor, actualConstructedCell.getCellType());

    auto actualConstructor = std::get<ConstructorDescription>(actualCell.cellTypeData);
    auto actualConstructedConstructor = std::get<ConstructorDescription>(actualConstructedCell.cellTypeData);
    EXPECT_TRUE(actualConstructor.genome.size() <= MAX_GENOME_BYTES);
    EXPECT_TRUE(constructedConstructor.getGenomeData().size() <= MAX_GENOME_BYTES);
}

TEST_F(ConstructorTests, constructConstructorCell_copyGenome)
{
    auto constructedConstructor = ConstructorGenomeDescription().setMode(0).setMakeSelfCopy();

    auto genome =
        GenomeDescriptionConverterService::get().convertDescriptionToBytes(GenomeDescription().setCells({CellGenomeDescription().setCellTypeData(constructedConstructor)}));

    DataDescription data;
    data.addCell(CellDescription()
                     .setId(1)
                     .setEnergy(_parameters.cellNormalEnergy[0] * 3)
                     .setCellTypeData(ConstructorDescription().setGenome(genome)));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);
    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(2, actualData.cells.size());
    auto actualConstructedCell = getOtherCell(actualData, 1);

    EXPECT_EQ(CellType_Constructor, actualConstructedCell.getCellType());

    auto actualConstructor = std::get<ConstructorDescription>(actualConstructedCell.cellTypeData);
    EXPECT_EQ(genome, actualConstructor.genome);
}

TEST_F(ConstructorTests, constructSecondCell_separation)
{
    auto genome = GenomeDescriptionConverterService::get().convertDescriptionToBytes(
        GenomeDescription().setHeader(GenomeHeaderDescription().setSeparateConstruction(true)).setCells({CellGenomeDescription(), CellGenomeDescription()}));

    DataDescription data;
    data.addCells({
        CellDescription()
            .setId(1)
            .setPos({10.0f, 10.0f})
            .setEnergy(_parameters.cellNormalEnergy[0] * 3)
            .setCellTypeData(ConstructorDescription().setGenomeCurrentNodeIndex(1).setGenome(genome)),
        CellDescription()
            .setId(2)
            .setPos({10.0f - getOffspringDistance(), 10.0f})
            .setEnergy(100)
            .setCellTypeData(OscillatorDescription())
            .setLivingState(LivingState_UnderConstruction),
    });
    data.addConnection(1, 2);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);
    auto actualData = _simulationFacade->getSimulationData();

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
    auto genome = GenomeDescriptionConverterService::get().convertDescriptionToBytes(
        GenomeDescription().setHeader(GenomeHeaderDescription().setSeparateConstruction(true)).setCells({CellGenomeDescription(), CellGenomeDescription()}));

    DataDescription data;
    data.addCells({
        CellDescription()
            .setId(1)
            .setPos({10.0f, 10.0f})
            .setEnergy(_parameters.cellNormalEnergy[0] * 3)
            .setCellTypeData(ConstructorDescription().setGenomeCurrentNodeIndex(1).setGenome(genome)),
        CellDescription()
            .setId(2)
            .setPos({10.0f - getOffspringDistance(), 10.0f})
            .setEnergy(100)
            .setCellTypeData(OscillatorDescription())
            .setLivingState(LivingState_UnderConstruction),
    });
    data.addConnection(1, 2);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);
    _simulationFacade->calcTimesteps(1);
    {
        auto actualData = _simulationFacade->getSimulationData();

        ASSERT_EQ(3, actualData.cells.size());
        auto actualHostCell = getCell(actualData, 1);
        auto actualPrevConstructedCell = getCell(actualData, 2);
        auto actualConstructedCell = getOtherCell(actualData, {1, 2});

        EXPECT_EQ(LivingState_Ready, actualHostCell.livingState);
        EXPECT_EQ(LivingState_Activating, actualPrevConstructedCell.livingState);
        EXPECT_EQ(LivingState_Ready, actualConstructedCell.livingState);
    }
    _simulationFacade->calcTimesteps(1);
    {
        auto actualData = _simulationFacade->getSimulationData();

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
    auto genome = GenomeDescriptionConverterService::get().convertDescriptionToBytes(
        GenomeDescription().setHeader(GenomeHeaderDescription().setSeparateConstruction(false)).setCells({CellGenomeDescription(), CellGenomeDescription()}));

    DataDescription data;
    data.addCells({
        CellDescription()
            .setId(1)
            .setPos({10.0f, 10.0f})
            .setEnergy(_parameters.cellNormalEnergy[0] * 3)
            .setCellTypeData(ConstructorDescription().setGenomeCurrentNodeIndex(1).setGenome(genome)),
        CellDescription()
            .setId(2)
            .setPos({10.0f - getOffspringDistance(), 10.0f})
            .setEnergy(100)
            .setCellTypeData(OscillatorDescription())
            .setLivingState(LivingState_UnderConstruction),
    });
    data.addConnection(1, 2);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);
    auto actualData = _simulationFacade->getSimulationData();

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
    auto genome = GenomeDescriptionConverterService::get().convertDescriptionToBytes(
        GenomeDescription().setHeader(GenomeHeaderDescription().setSeparateConstruction(false)).setCells({CellGenomeDescription(), CellGenomeDescription()}));

    DataDescription data;
    data.addCells({
        CellDescription()
            .setId(1)
            .setPos({10.0f, 10.0f})
            .setEnergy(_parameters.cellNormalEnergy[0] * 3)
            .setCellTypeData(ConstructorDescription().setGenomeCurrentNodeIndex(1).setGenome(genome)),
        CellDescription()
            .setId(2)
            .setPos({10.0f - 1.0f - _parameters.cellMinDistance/2, 10.0f})
            .setEnergy(100)
            .setCellTypeData(OscillatorDescription())
            .setLivingState(LivingState_UnderConstruction),
    });
    data.addConnection(1, 2);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);
    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(2, actualData.cells.size());
    auto actualHostCell = getCell(actualData, 1);
    auto actualPrevConstructedCell = getCell(actualData, 2);

    EXPECT_EQ(1, actualHostCell.connections.size());
    EXPECT_TRUE(approxCompare(0.0f, actualHostCell.signal->channels[0]));
    ASSERT_EQ(1, actualPrevConstructedCell.connections.size());
    auto actualConstructor = std::get<ConstructorDescription>(actualHostCell.cellTypeData);
    EXPECT_EQ(1, actualConstructor.genomeCurrentNodeIndex);
}

TEST_F(ConstructorTests, constructSecondCell_notFinished)
{
    auto genome = GenomeDescriptionConverterService::get().convertDescriptionToBytes(GenomeDescription()
                                                                .setHeader(GenomeHeaderDescription().setSeparateConstruction(false))
                                                                .setCells({CellGenomeDescription(), CellGenomeDescription(), CellGenomeDescription()}));

    DataDescription data;
    data.addCells({
        CellDescription()
            .setId(1)
            .setPos({10.0f, 10.0f})
            .setEnergy(_parameters.cellNormalEnergy[0] * 3)
            .setCellTypeData(ConstructorDescription().setGenomeCurrentNodeIndex(1).setGenome(genome)),
        CellDescription()
            .setId(2)
            .setPos({10.0f - getOffspringDistance(), 10.0f})
            .setEnergy(100)
            .setCellTypeData(OscillatorDescription())
            .setLivingState(LivingState_UnderConstruction),
    });
    data.addConnection(1, 2);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);
    auto actualData = _simulationFacade->getSimulationData();

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
    auto genome = GenomeDescriptionConverterService::get().convertDescriptionToBytes(
        GenomeDescription().setHeader(GenomeHeaderDescription().setSeparateConstruction(false)).setCells({CellGenomeDescription(), CellGenomeDescription()}));

    DataDescription data;
    data.addCells({
        CellDescription()
            .setId(1)
            .setPos({10.0f, 10.0f})
            .setEnergy(_parameters.cellNormalEnergy[0] * 3)
            .setCellTypeData(ConstructorDescription().setGenome(genome).setConstructionAngle2(90.0f).setGenomeCurrentNodeIndex(1)),
        CellDescription()
            .setId(2)
            .setPos({10.0f - getOffspringDistance(), 10.0f})
            .setEnergy(100)
            .setCellTypeData(OscillatorDescription())
            .setLivingState(LivingState_UnderConstruction),
    });
    data.addConnection(1, 2);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);
    auto actualData = _simulationFacade->getSimulationData();

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
    auto genome = GenomeDescriptionConverterService::get().convertDescriptionToBytes(
        GenomeDescription().setHeader(GenomeHeaderDescription().setSeparateConstruction(false)).setCells({CellGenomeDescription(), CellGenomeDescription()}));

    DataDescription data;
    data.addCells({
        CellDescription()
            .setId(1)
            .setPos({10.0f, 10.0f})
            .setEnergy(_parameters.cellNormalEnergy[0] * 3)
            .setCellTypeData(ConstructorDescription().setGenome(genome).setConstructionAngle2(-90.0f).setGenomeCurrentNodeIndex(1)),
        CellDescription()
            .setId(2)
            .setPos({10.0f - getOffspringDistance(), 10.0f})
            .setEnergy(100)
            .setCellTypeData(OscillatorDescription())
            .setLivingState(LivingState_UnderConstruction),
    });
    data.addConnection(1, 2);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);
    auto actualData = _simulationFacade->getSimulationData();

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
    auto genome = GenomeDescriptionConverterService::get().convertDescriptionToBytes(
        GenomeDescription().setHeader(GenomeHeaderDescription().setInfiniteRepetitions()).setCells({CellGenomeDescription(), CellGenomeDescription()}));

    DataDescription data;
    data.addCells({
        CellDescription()
            .setId(1)
            .setPos({10.0f - getOffspringDistance(), 10.0f})
            .setEnergy(_parameters.cellNormalEnergy[0] * 3)
            .setCellTypeData(ConstructorDescription().setGenomeCurrentNodeIndex(1).setGenome(genome)),
        CellDescription().setId(2).setPos({11.0f, 10.0f}).setLivingState(LivingState_UnderConstruction),
    });
    data.addConnection(1, 2);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    ASSERT_EQ(3, actualData.cells.size());

    auto actualHostCell = getCell(actualData, 1);
    auto actualConstructedCell = getOtherCell(actualData, {1, 2});
    EXPECT_EQ(0, std::get<ConstructorDescription>(actualHostCell.cellTypeData).currentBranch);
    EXPECT_EQ(LivingState_Activating, actualConstructedCell.livingState);
}

TEST_F(ConstructorTests, constructThirdCell_multipleConnections_upperPart)
{
    auto genome = GenomeDescriptionConverterService::get().convertDescriptionToBytes(GenomeDescription()
                                                                .setHeader(GenomeHeaderDescription().setSeparateConstruction(false))
                                                                .setCells({CellGenomeDescription(), CellGenomeDescription(), CellGenomeDescription()}));

    DataDescription data;
    data.addCells({
        CellDescription()
            .setId(1)
            .setPos({10.0f, 10.0f})
            .setEnergy(_parameters.cellNormalEnergy[0] * 3)
            .setCellTypeData(ConstructorDescription().setGenomeCurrentNodeIndex(2).setGenome(genome)),
        CellDescription()
            .setId(2)
            .setPos({10.0f - getOffspringDistance(), 10.0f})
            .setEnergy(100)
            .setCellTypeData(OscillatorDescription())
            .setLivingState(LivingState_UnderConstruction),
        CellDescription()
            .setId(3)
            .setPos({10.0f - getOffspringDistance(), 9.0f})
            .setEnergy(100)
            .setCellTypeData(OscillatorDescription())
            .setLivingState(LivingState_UnderConstruction),
        CellDescription().setId(4).setPos({10.0f, 9.5f}).setEnergy(_parameters.cellNormalEnergy[0] * 3),
        CellDescription().setId(5).setPos({10.0f, 9.0f}).setEnergy(_parameters.cellNormalEnergy[0] * 3),
    });
    data.addConnection(1, 2);
    data.addConnection(2, 3);
    data.addConnection(1, 4);
    data.addConnection(4, 5);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);
    auto actualData = _simulationFacade->getSimulationData();

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
    auto genome = GenomeDescriptionConverterService::get().convertDescriptionToBytes(GenomeDescription()
                                                                .setHeader(GenomeHeaderDescription().setSeparateConstruction(false))
                                                                .setCells({CellGenomeDescription(), CellGenomeDescription(), CellGenomeDescription()}));

    DataDescription data;
    data.addCells({
        CellDescription()
            .setId(1)
            .setPos({10.0f, 10.0f})
            .setEnergy(_parameters.cellNormalEnergy[0] * 3)
            .setCellTypeData(ConstructorDescription().setGenomeCurrentNodeIndex(2).setGenome(genome)),
        CellDescription()
            .setId(2)
            .setPos({10.0f - getOffspringDistance(), 10.0f})
            .setEnergy(100)
            .setCellTypeData(OscillatorDescription())
            .setLivingState(LivingState_UnderConstruction),
        CellDescription()
            .setId(3)
            .setPos({10.0f - getOffspringDistance(), 11.0f})
            .setEnergy(100)
            .setCellTypeData(OscillatorDescription())
            .setLivingState(LivingState_UnderConstruction),
        CellDescription().setId(4).setPos({10.0f, 10.5f}).setEnergy(_parameters.cellNormalEnergy[0] * 3),
        CellDescription().setId(5).setPos({10.0f, 11.0f}).setEnergy(_parameters.cellNormalEnergy[0] * 3),
    });
    data.addConnection(1, 2);
    data.addConnection(2, 3);
    data.addConnection(1, 4);
    data.addConnection(4, 5);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);
    auto actualData = _simulationFacade->getSimulationData();

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
    auto genome = GenomeDescriptionConverterService::get().convertDescriptionToBytes(
        GenomeDescription().setHeader(GenomeHeaderDescription().setSeparateConstruction(false).setNumBranches(1)).setCells({CellGenomeDescription()}));

    DataDescription data;
    data.addCells({
        CellDescription()
            .setId(1)
            .setPos({10.0f, 10.0f})
            .setEnergy(_parameters.cellNormalEnergy[0] * 3)
            .setCellTypeData(ConstructorDescription().setGenome(genome)),
        CellDescription()
            .setId(2)
            .setPos({10.0f - getOffspringDistance(), 10.0f})
            .setEnergy(100)
            .setCellTypeData(OscillatorDescription())
            .setLivingState(LivingState_UnderConstruction),
    });
    data.addConnection(1, 2);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);
    auto actualData = _simulationFacade->getSimulationData();
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
    auto genome = GenomeDescriptionConverterService::get().convertDescriptionToBytes(GenomeDescription()
                                                                .setHeader(GenomeHeaderDescription().setSeparateConstruction(false))
            .setCells({CellGenomeDescription(), CellGenomeDescription(), CellGenomeDescription(), CellGenomeDescription(), CellGenomeDescription()}));

    DataDescription data;
    data.addCells({
        CellDescription()
            .setId(1)
            .setPos({10.0f, 10.0f})
            .setEnergy(_parameters.cellNormalEnergy[0] * 3)
            .setCellTypeData(ConstructorDescription().setGenomeCurrentNodeIndex(4).setGenome(genome)),
        CellDescription()
            .setId(2)
            .setPos({10.0f - getOffspringDistance(), 10.0f})
            .setEnergy(100)
            .setCellTypeData(OscillatorDescription())
            .setLivingState(LivingState_UnderConstruction),
        CellDescription()
            .setId(3)
            .setPos({10.0f - getOffspringDistance(), 11.0f})
            .setEnergy(100)
            .setCellTypeData(OscillatorDescription())
            .setLivingState(LivingState_UnderConstruction),
        CellDescription()
            .setId(4)
            .setPos({10.0f - getOffspringDistance() + 1.0f, 11.0f})
            .setLivingState(LivingState_UnderConstruction),
    });
    data.addConnection(1, 2);
    data.addConnection(2, 3);
    data.addConnection(3, 4);
    data.addConnection(4, 2);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);
    auto actualData = _simulationFacade->getSimulationData();

    EXPECT_EQ(5, actualData.cells.size());
    auto actualHostCell = getCell(actualData, 1);
    auto actualPrevConstructedCell = getCell(actualData, 2);
    auto actualPrevPrevConstructedCell = getCell(actualData, 3);
    auto actualPrevPrevPrevConstructedCell = getCell(actualData, 4);
    auto actualConstructedCell = getOtherCell(actualData, {1, 2, 3, 4});

    EXPECT_EQ(1, actualHostCell.connections.size());
    ASSERT_EQ(3, actualConstructedCell.connections.size());
    ASSERT_EQ(3, actualPrevConstructedCell.connections.size());
    ASSERT_EQ(2, actualPrevPrevConstructedCell.connections.size());
    ASSERT_EQ(3, actualPrevPrevPrevConstructedCell.connections.size());
    EXPECT_TRUE(hasConnection(actualData, actualConstructedCell.id, 1));
    EXPECT_TRUE(hasConnection(actualData, actualConstructedCell.id, 2));
    EXPECT_TRUE(hasConnection(actualData, actualConstructedCell.id, 4));
}

TEST_F(ConstructorTests, constructLastCellFirstRepetition)
{
    auto genome = GenomeDescriptionConverterService::get().convertDescriptionToBytes(
        GenomeDescription().setHeader(GenomeHeaderDescription().setNumRepetitions(2)).setCells({CellGenomeDescription(), CellGenomeDescription()}));

    DataDescription data;
    data.addCells({
        CellDescription()
            .setId(1)
            .setPos({10.0f, 10.0f})
            .setEnergy(_parameters.cellNormalEnergy[0] * 3)
            .setCellTypeData(ConstructorDescription().setGenome(genome).setGenomeCurrentNodeIndex(1)),
        CellDescription()
            .setId(2)
            .setPos({10.0f - getOffspringDistance(), 10.0f})
            .setEnergy(100)
            .setCellTypeData(OscillatorDescription())
            .setLivingState(LivingState_UnderConstruction),
    });
    data.addConnection(1, 2);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);
    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(3, actualData.cells.size());
    auto actualConstructedCell = getOtherCell(actualData, {1, 2});

    EXPECT_EQ(LivingState_UnderConstruction, actualConstructedCell.livingState);
}

TEST_F(ConstructorTests, constructLastCellLastRepetition)
{
    auto genome = GenomeDescriptionConverterService::get().convertDescriptionToBytes(
        GenomeDescription().setHeader(GenomeHeaderDescription().setNumRepetitions(2)).setCells({CellGenomeDescription(), CellGenomeDescription()}));

    DataDescription data;
    data.addCells({
        CellDescription()
            .setId(1)
            .setPos({10.0f, 10.0f})
            .setEnergy(_parameters.cellNormalEnergy[0] * 3)
            .setCellTypeData(ConstructorDescription().setGenome(genome).setGenomeCurrentNodeIndex(1).setGenomeCurrentRepetition(1)),
        CellDescription()
            .setId(2)
            .setPos({10.0f - getOffspringDistance(), 10.0f})
            .setEnergy(100)
            .setCellTypeData(OscillatorDescription())
            .setLivingState(LivingState_UnderConstruction),
        CellDescription()
            .setId(3)
            .setPos({9.0f - getOffspringDistance(), 10.0f})
            .setEnergy(100)
            .setCellTypeData(OscillatorDescription())
            .setLivingState(LivingState_UnderConstruction),
    });
    data.addConnection(1, 2);
    data.addConnection(2, 3);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);
    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(4, actualData.cells.size());
    auto actualConstructedCell = getOtherCell(actualData, {1, 2, 3});

    EXPECT_EQ(LivingState_Activating, actualConstructedCell.livingState);
}

TEST_F(ConstructorTests, restartIfNoLastConstructedCellFound)
{
    auto genome =
        GenomeDescriptionConverterService::get().convertDescriptionToBytes(GenomeDescription()
                                                                  .setHeader(GenomeHeaderDescription().setNumRepetitions(2))
                                                                  .setCells({CellGenomeDescription(), CellGenomeDescription(), CellGenomeDescription()}));

    DataDescription data;
    data.addCells({
        CellDescription()
            .setId(1)
            .setPos({10.0f, 10.0f})
            .setEnergy(_parameters.cellNormalEnergy[0] * 3)
            .setCellTypeData(ConstructorDescription().setGenome(genome).setGenomeCurrentNodeIndex(1).setGenomeCurrentRepetition(1)),
        CellDescription()
            .setId(2)
            .setPos({10.0f - getOffspringDistance(), 10.0f})
            .setEnergy(100)
            .setCellTypeData(OscillatorDescription())
            .setSignal({1, 0, 0, 0, 0, 0, 0, 0}),
    });
    data.addConnection(1, 2);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);
    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(3, actualData.cells.size());
    auto actualHostCell = getCell(actualData, 1);

    auto actualConstructor = std::get<ConstructorDescription>(actualHostCell.cellTypeData);
    EXPECT_EQ(1, actualConstructor.genomeCurrentNodeIndex);
    EXPECT_EQ(0, actualConstructor.genomeCurrentRepetition);
}

TEST_F(ConstructorTests, restartIfLastConstructedCellHasLowNumConnections)
{
    auto genome =
        GenomeDescriptionConverterService::get().convertDescriptionToBytes(GenomeDescription()
                                                                  .setHeader(GenomeHeaderDescription().setNumRepetitions(2))
                                                                  .setCells({CellGenomeDescription(), CellGenomeDescription(), CellGenomeDescription()}));

    DataDescription data;
    data.addCells({
        CellDescription()
            .setId(1)
            .setPos({10.0f, 10.0f})
            .setEnergy(_parameters.cellNormalEnergy[0] * 3)
            .setCellTypeData(ConstructorDescription().setGenome(genome).setGenomeCurrentNodeIndex(1).setGenomeCurrentRepetition(1).setNumInheritedGenomeNodes(3)),
        CellDescription()
            .setId(2)
            .setPos({10.0f - getOffspringDistance(), 10.0f})
            .setEnergy(100)
            .setCellTypeData(OscillatorDescription())
            .setLivingState(LivingState_UnderConstruction),
    });
    data.addConnection(1, 2);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);
    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(3, actualData.cells.size());
    auto actualHostCell = getCell(actualData, 1);

    auto actualConstructor = std::get<ConstructorDescription>(actualHostCell.cellTypeData);
    EXPECT_EQ(1, actualConstructor.genomeCurrentNodeIndex);
    EXPECT_EQ(0, actualConstructor.genomeCurrentRepetition);
}

TEST_F(ConstructorTests, allowLargeConstructionAngle1)
{
    auto genome =
        GenomeDescriptionConverterService::get().convertDescriptionToBytes(GenomeDescription().setHeader(GenomeHeaderDescription()).setCells({CellGenomeDescription()}));

    DataDescription data;
    data.addCells({
        CellDescription()
            .setId(1)
            .setPos({10.0f, 10.0f})
            .setEnergy(_parameters.cellNormalEnergy[0] * 3)
            .setCellTypeData(ConstructorDescription().setGenome(genome).setConstructionAngle1(180.0f)),
        CellDescription().setId(2).setPos({11.0f, 9.0f}).setEnergy(100),
        CellDescription().setId(3).setPos({11.0f, 11.0f}).setEnergy(100),
    });
    data.addConnection(1, 2);
    data.addConnection(1, 3);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);
    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(4, actualData.cells.size());
    auto actualConstructedCell = getOtherCell(actualData, {1, 2, 3});

    EXPECT_TRUE(approxCompare(11.0f, actualConstructedCell.pos.x));
    EXPECT_TRUE(approxCompare(10.0f, actualConstructedCell.pos.y));
}

TEST_F(ConstructorTests, allowLargeConstructionAngle2)
{
    auto genome =
        GenomeDescriptionConverterService::get().convertDescriptionToBytes(GenomeDescription().setHeader(GenomeHeaderDescription()).setCells({CellGenomeDescription()}));

    DataDescription data;
    data.addCells({
        CellDescription()
            .setId(1)
            .setPos({10.0f, 10.0f})
            .setEnergy(_parameters.cellNormalEnergy[0] * 3)
            .setCellTypeData(ConstructorDescription().setGenome(genome).setConstructionAngle1(-180.0f)),
        CellDescription().setId(2).setPos({11.0f, 9.0f}).setEnergy(100),
        CellDescription().setId(3).setPos({11.0f, 11.0f}).setEnergy(100),
    });
    data.addConnection(1, 2);
    data.addConnection(1, 3);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);
    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(4, actualData.cells.size());
    auto actualConstructedCell = getOtherCell(actualData, {1, 2, 3});

    EXPECT_TRUE(approxCompare(11.0f, actualConstructedCell.pos.x));
    EXPECT_TRUE(approxCompare(10.0f, actualConstructedCell.pos.y));
}

TEST_F(ConstructorTests, repetitionsAndBranches)
{
    auto genome = GenomeDescriptionConverterService::get().convertDescriptionToBytes(
        GenomeDescription()
            .setHeader(GenomeHeaderDescription().setNumBranches(3).setNumRepetitions(4).setSeparateConstruction(false))
            .setCells({CellGenomeDescription(), CellGenomeDescription(), CellGenomeDescription()}));

    DataDescription data;
    data.addCells({
        CellDescription()
            .setId(1)
            .setPos({10.0f, 10.0f})
            .setEnergy(_parameters.cellNormalEnergy[0] * 2 * 3 * 4 * 3)
            .setCellTypeData(ConstructorDescription().setGenome(genome).setAutoTriggerInterval(20)),
    });

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(13 * 3 * 4 * 3 * 20);
    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(1 + 3 * 4 * 3, actualData.cells.size());
    auto actualConstructor = getCell(actualData, 1);

    EXPECT_EQ(3, actualConstructor.connections.size());
}

TEST_F(ConstructorTests, severalRepetitionsOfSingleCell)
{
    auto genome = GenomeDescriptionConverterService::get().convertDescriptionToBytes(
        GenomeDescription()
            .setHeader(GenomeHeaderDescription().setNumBranches(1).setNumRepetitions(2).setSeparateConstruction(false))
            .setCells({CellGenomeDescription()}));

    DataDescription data;
    data.addCells({
        CellDescription()
            .setId(1)
            .setPos({10.0f, 10.0f})
            .setEnergy(_parameters.cellNormalEnergy[0] * 2 * 3 * 4 * 3)
            .setCellTypeData(ConstructorDescription().setGenome(genome)),
    });

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(200 * 6);
    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(3, actualData.cells.size());
    auto actualConstructor = getCell(actualData, 1);

    EXPECT_EQ(1, actualConstructor.connections.size());
    auto lastContructedCell = getCell(actualData, actualConstructor.connections.at(0).cellId);
    EXPECT_EQ(2, lastContructedCell.connections.size());
}

TEST_F(ConstructorTests, severalRepetitionsAndBranchesOfSingleCell)
{
    auto genome = GenomeDescriptionConverterService::get().convertDescriptionToBytes(
        GenomeDescription()
            .setHeader(GenomeHeaderDescription().setNumBranches(3).setNumRepetitions(2).setSeparateConstruction(false))
            .setCells({CellGenomeDescription()}));

    DataDescription data;
    data.addCells({
        CellDescription()
            .setId(1)
            .setPos({10.0f, 10.0f})
            .setEnergy(_parameters.cellNormalEnergy[0] * 2 * 3 * 4 * 3)
            .setCellTypeData(ConstructorDescription().setGenome(genome)),
    });

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(400 * 6);
    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(7, actualData.cells.size());
    auto actualConstructor = getCell(actualData, 1);

    EXPECT_EQ(3, actualConstructor.connections.size());
    for (auto const& connection : actualConstructor.connections) {
        auto lastContructedCell = getCell(actualData, connection.cellId);
        EXPECT_EQ(2, lastContructedCell.connections.size());
    }
}

TEST_F(ConstructorTests, severalRepetitionsOfSingleCell_ignoreNumRequiredConnections)
{
    auto genome = GenomeDescriptionConverterService::get().convertDescriptionToBytes(
        GenomeDescription()
            .setHeader(GenomeHeaderDescription().setNumBranches(1).setNumRepetitions(3).setSeparateConstruction(false))
            .setCells({CellGenomeDescription().setNumRequiredAdditionalConnections(1)}));

    DataDescription data;
    data.addCells({
        CellDescription()
            .setId(1)
            .setPos({10.0f, 10.0f})
            .setEnergy(_parameters.cellNormalEnergy[0] * 2 * 3 * 4 * 3)
            .setCellTypeData(ConstructorDescription().setGenome(genome).setNumInheritedGenomeNodes(3)),
    });

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(400 * 6);
    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(4, actualData.cells.size());
    auto actualConstructor = getCell(actualData, 1);

    EXPECT_EQ(1, actualConstructor.connections.size());
    for (auto const& connection : actualConstructor.connections) {
        auto lastContructedCell = getCell(actualData, connection.cellId);
        EXPECT_EQ(2, lastContructedCell.connections.size());
    }
}
