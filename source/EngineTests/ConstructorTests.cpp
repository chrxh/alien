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
            result.radiationType1_strength.baseValue[i] = 0;
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
        GenomeDescription().header(GenomeHeaderDescription().separateConstruction(false)).cells({CellGenomeDescription()}));

    CollectionDescription data;
    data.addCell(CellDescription()
                     .id(1)
                     .energy(_parameters.normalCellEnergy.value[0] * 2 - 1.0f)
                     .cellType(ConstructorDescription().genome(genome)));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);
    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(1, actualData._cells.size());
    auto actualHostCell = getCell(actualData, 1);

    EXPECT_EQ(0, actualHostCell._connections.size());
    EXPECT_EQ(0, std::get<ConstructorDescription>(actualHostCell._cellTypeData)._genomeCurrentNodeIndex);
    EXPECT_TRUE(approxCompare(_parameters.normalCellEnergy.value[0] * 2 - 1.0f, actualHostCell._energy));
    EXPECT_TRUE(approxCompare(0.0f, actualHostCell._signal->_channels[0]));
}

TEST_F(ConstructorTests, alreadyFinished)
{
    CollectionDescription data;

    auto genome = GenomeDescriptionConverterService::get().convertDescriptionToBytes(
        GenomeDescription().header(GenomeHeaderDescription().numBranches(1)).cells({CellGenomeDescription()}));

    auto constructor = ConstructorDescription().genome(genome).genomeCurrentBranch(1);

    data.addCell(
        CellDescription()
            .id(1)
            .energy(_parameters.normalCellEnergy.value[0] * 3)
            .cellType(constructor));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);
    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(1, actualData._cells.size());
    auto actualHostCell = getCell(actualData, 1);
    auto actualConstructor = std::get<ConstructorDescription>(actualHostCell._cellTypeData);
    EXPECT_EQ(0, actualHostCell._connections.size());
    EXPECT_EQ(0, actualConstructor._genomeCurrentNodeIndex);
    EXPECT_TRUE(approxCompare(0.0f, actualHostCell._signal->_channels[0]));
}

TEST_F(ConstructorTests, notActivated)
{
    CollectionDescription data;

    auto genome = GenomeDescriptionConverterService::get().convertDescriptionToBytes(
        GenomeDescription().header(GenomeHeaderDescription().numBranches(1)).cells({CellGenomeDescription()}));
    auto constructor = ConstructorDescription().genome(genome);

    data.addCell(CellDescription()
                     .id(1)
                     .energy(_parameters.normalCellEnergy.value[0] * 3)
                     .cellType(constructor)
                     .activationTime(2));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);
    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(1, actualData._cells.size());
    auto actualHostCell = getCell(actualData, 1);
    auto actualConstructor = std::get<ConstructorDescription>(actualHostCell._cellTypeData);
    EXPECT_TRUE(approxCompare(0.0f, actualHostCell._signal->_channels[0]));
}

TEST_F(ConstructorTests, manualConstruction_noInputSignal)
{
    auto genome = GenomeDescriptionConverterService::get().convertDescriptionToBytes(GenomeDescription().cells({CellGenomeDescription()}));

    CollectionDescription data;
    data.addCell(CellDescription()
                     .id(1)
                     .energy(_parameters.normalCellEnergy.value[0] * 3)
                     .cellType(ConstructorDescription().autoTriggerInterval(0).genome(genome)));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);
    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(1, actualData._cells.size());
    auto actualHostCell = getCell(actualData, 1);

    EXPECT_EQ(0, actualHostCell._connections.size());
    EXPECT_EQ(0, std::get<ConstructorDescription>(actualHostCell._cellTypeData)._genomeCurrentNodeIndex);
    EXPECT_TRUE(approxCompare(_parameters.normalCellEnergy.value[0] * 3, actualHostCell._energy));
    EXPECT_TRUE(approxCompare(0.0f, actualHostCell._signal->_channels[0]));
}

TEST_F(ConstructorTests, constructFirstCell_correctCycle)
{
    auto genome = GenomeDescriptionConverterService::get().convertDescriptionToBytes(GenomeDescription().cells({CellGenomeDescription()}));

    _simulationFacade->calcTimesteps(1);

    CollectionDescription data;
    data.addCell(CellDescription()
                     .id(1)
                     .energy(_parameters.normalCellEnergy.value[0] * 3)
                     .cellType(ConstructorDescription().autoTriggerInterval(3 * 6).genome(genome)));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(3 * 6);
    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(2, actualData._cells.size());
}

TEST_F(ConstructorTests, constructFirstCell_oneCellGenome_infiniteRepetitions)
{
    auto genome = GenomeDescriptionConverterService::get().convertDescriptionToBytes(
        GenomeDescription().header(GenomeHeaderDescription().infiniteRepetitions()).cells({CellGenomeDescription()}));

    _simulationFacade->calcTimesteps(1);

    CollectionDescription data;
    data.addCell(CellDescription()
                     .id(1)
                     .energy(_parameters.normalCellEnergy.value[0] * 3)
                     .cellType(ConstructorDescription().autoTriggerInterval(3 * 6).genome(genome)));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(3 * 6);
    auto actualData = _simulationFacade->getSimulationData();
    ASSERT_EQ(2, actualData._cells.size());

    auto actualHostCell = getCell(actualData, 1);
    auto actualConstructedCell = getOtherCell(actualData, {1});
    EXPECT_EQ(0, std::get<ConstructorDescription>(actualHostCell._cellTypeData)._genomeCurrentRepetition);
    EXPECT_EQ(0, std::get<ConstructorDescription>(actualHostCell._cellTypeData)._genomeCurrentBranch);
    EXPECT_EQ(LivingState_Activating, actualConstructedCell._livingState);
}

TEST_F(ConstructorTests, constructFirstCell_twoCellGenome_infiniteRepetitions)
{
    auto genome = GenomeDescriptionConverterService::get().convertDescriptionToBytes(
        GenomeDescription().header(GenomeHeaderDescription().infiniteRepetitions()).cells({CellGenomeDescription(), CellGenomeDescription()}));

    _simulationFacade->calcTimesteps(1);

    CollectionDescription data;
    data.addCell(CellDescription()
                     .id(1)
                     .energy(_parameters.normalCellEnergy.value[0] * 3)
                     .cellType(ConstructorDescription().autoTriggerInterval(3 * 6).genome(genome)));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);
    auto actualData = _simulationFacade->getSimulationData();
    ASSERT_EQ(2, actualData._cells.size());

    auto actualHostCell = getCell(actualData, 1);
    auto actualConstructedCell = getOtherCell(actualData, {1});
    EXPECT_EQ(0, std::get<ConstructorDescription>(actualHostCell._cellTypeData)._genomeCurrentRepetition);
    EXPECT_EQ(0, std::get<ConstructorDescription>(actualHostCell._cellTypeData)._genomeCurrentBranch);
    EXPECT_EQ(LivingState_UnderConstruction, actualConstructedCell._livingState);
}

TEST_F(ConstructorTests, constructFirstCell_wrongCycle)
{
    auto genome = GenomeDescriptionConverterService::get().convertDescriptionToBytes(
        GenomeDescription().cells({CellGenomeDescription()}));

    _simulationFacade->calcTimesteps(1);

    CollectionDescription data;
    data.addCell(CellDescription()
                     .id(1)
                     .energy(_parameters.normalCellEnergy.value[0] * 3)
                     .cellType(ConstructorDescription().autoTriggerInterval(3 * 6).genome(genome)));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(3 * 6 - 1);
    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(1, actualData._cells.size());
}

TEST_F(ConstructorTests, constructFirstCell_completenessCheck_constructionNotBuilt)
{
    auto constructorGenome = ConstructorGenomeDescription().mode(0).constructionActivationTime(123).makeSelfCopy();
    auto genome =
        GenomeDescriptionConverterService::get().convertDescriptionToBytes(GenomeDescription().cells({CellGenomeDescription().cellType(constructorGenome)}));
    auto otherGenome = GenomeDescriptionConverterService::get().convertDescriptionToBytes(GenomeDescription().cells({CellGenomeDescription()}));

    CollectionDescription data;
    data.addCells({
        CellDescription()
            .id(1)
            .pos({10.0f, 10.0f})
            .energy(_parameters.normalCellEnergy.value[0] * 3)
            .cellType(ConstructorDescription().genome(genome).numInheritedGenomeNodes(4)),
        CellDescription().id(2).pos({11.0f, 10.0f}).energy(100).cellType(OscillatorDescription()),
        CellDescription()
            .id(3)
            .pos({12.0f, 10.0f})
            .energy(100)
            .cellType(ConstructorDescription().genome(otherGenome)),
    });
    data.addConnection(1, 2);
    data.addConnection(2, 3);

    _parameters.constructorCompletenessCheck.value = true;
    _simulationFacade->setSimulationParameters(_parameters);
    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);
    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(3, actualData._cells.size());
}

TEST_F(ConstructorTests, constructFirstCell_completenessCheck_repeatedConstructionNotBuilt)
{
    auto constructorGenome = ConstructorGenomeDescription().mode(0).constructionActivationTime(123).makeSelfCopy();
    auto genome =
        GenomeDescriptionConverterService::get().convertDescriptionToBytes(GenomeDescription().cells({CellGenomeDescription().cellType(constructorGenome)}));
    auto otherGenome = GenomeDescriptionConverterService::get().convertDescriptionToBytes(
        GenomeDescription().header(GenomeHeaderDescription().numRepetitions(2)).cells({CellGenomeDescription()}));

    CollectionDescription data;
    data.addCells({
        CellDescription()
            .id(1)
            .pos({10.0f, 10.0f})
            .energy(_parameters.normalCellEnergy.value[0] * 3)
            .cellType(ConstructorDescription().genome(genome).numInheritedGenomeNodes(5)),
        CellDescription().id(2).pos({11.0f, 10.0f}).energy(100).cellType(OscillatorDescription()),
        CellDescription()
            .id(3)
            .pos({12.0f, 10.0f})
            .energy(100)
            .cellType(ConstructorDescription().genome(otherGenome).genomeCurrentRepetition(1)),
        CellDescription().id(4).pos({10.0f, 11.0f}).energy(100),
    });
    data.addConnection(1, 2);
    data.addConnection(2, 3);
    data.addConnection(3, 4);

    _parameters.constructorCompletenessCheck.value = true;
    _simulationFacade->setSimulationParameters(_parameters);
    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);
    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(4, actualData._cells.size());
}

TEST_F(ConstructorTests, constructFirstCell_completenessCheck_constructionBuilt)
{
    auto otherGenome = GenomeDescriptionConverterService::get().convertDescriptionToBytes(GenomeDescription().cells({CellGenomeDescription()}));
    auto otherConstructorGenome = ConstructorGenomeDescription().mode(0).constructionActivationTime(123).genome(otherGenome);

    auto constructorGenome = ConstructorGenomeDescription().mode(0).constructionActivationTime(123).makeSelfCopy();
    auto genome = GenomeDescriptionConverterService::get().convertDescriptionToBytes(GenomeDescription().cells(
        {CellGenomeDescription().cellType(constructorGenome), CellGenomeDescription(), CellGenomeDescription().cellType(otherConstructorGenome)}));

    CollectionDescription data;
    data.addCells({
        CellDescription()
            .id(1)
            .pos({10.0f, 10.0f})
            .energy(_parameters.normalCellEnergy.value[0] * 3)
            .cellType(ConstructorDescription().genome(genome).numInheritedGenomeNodes(4)),
        CellDescription().id(2).pos({11.0f, 10.0f}).energy(100).cellType(OscillatorDescription()),
        CellDescription()
            .id(3)
            .pos({12.0f, 10.0f})
            .energy(100)
            .cellType(ConstructorDescription().genome(otherGenome).genomeCurrentBranch(1)),
        CellDescription().id(4).pos({10.0f, 11.0f}).energy(100),
    });
    data.addConnection(1, 2);
    data.addConnection(2, 3);
    data.addConnection(3, 4);

    _parameters.constructorCompletenessCheck.value = true;
    _simulationFacade->setSimulationParameters(_parameters);
    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);
    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(5, actualData._cells.size());
}

TEST_F(ConstructorTests, constructFirstCell_completenessCheck_infiniteConstructionsBuilt)
{
    auto otherGenome = GenomeDescriptionConverterService::get().convertDescriptionToBytes(
        GenomeDescription().header(GenomeHeaderDescription().infiniteRepetitions()).cells({CellGenomeDescription()}));
    auto otherConstructorGenome = ConstructorGenomeDescription().mode(0).constructionActivationTime(123).genome(otherGenome);

    auto constructorGenome = ConstructorGenomeDescription().mode(0).constructionActivationTime(123).makeSelfCopy();
    auto genome = GenomeDescriptionConverterService::get().convertDescriptionToBytes(GenomeDescription().cells(
        {CellGenomeDescription().cellType(constructorGenome), CellGenomeDescription(), CellGenomeDescription().cellType(otherConstructorGenome)}));

    CollectionDescription data;
    data.addCells({
        CellDescription()
            .id(1)
            .pos({10.0f, 10.0f})
            .energy(_parameters.normalCellEnergy.value[0] * 3)
            .cellType(ConstructorDescription().genome(genome).numInheritedGenomeNodes(4)),
        CellDescription().id(2).pos({11.0f, 10.0f}).energy(100).cellType(OscillatorDescription()),
        CellDescription()
            .id(3)
            .pos({12.0f, 10.0f})
            .energy(100)
            .cellType(ConstructorDescription().genome(otherGenome).genomeCurrentBranch(1)),
        CellDescription().id(4).pos({10.0f, 11.0f}).energy(100),
    });
    data.addConnection(1, 2);
    data.addConnection(2, 3);
    data.addConnection(3, 4);

    _parameters.constructorCompletenessCheck.value = true;
    _simulationFacade->setSimulationParameters(_parameters);
    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);
    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(5, actualData._cells.size());
}

TEST_F(ConstructorTests, constructFirstCell_completenessCheck_largeCluster)
{
    auto constexpr RectLength = 50;
    auto rect = DescriptionEditService::get().createRect(DescriptionEditService::CreateRectParameters().height(RectLength).width(RectLength));

    auto constructorGenome = ConstructorGenomeDescription().mode(0).constructionActivationTime(123).makeSelfCopy();
    auto genome =
        GenomeDescriptionConverterService::get().convertDescriptionToBytes(GenomeDescription().cells({CellGenomeDescription().cellType(constructorGenome)}));
    auto otherGenome = GenomeDescriptionConverterService::get().convertDescriptionToBytes(GenomeDescription().cells({CellGenomeDescription()}));

    auto& cell1 = rect._cells.at(0);
    cell1.energy(_parameters.normalCellEnergy.value[0] * 3)
        .cellType(ConstructorDescription().genome(genome).numInheritedGenomeNodes(RectLength * RectLength));

    _parameters.constructorCompletenessCheck.value = true;
    _simulationFacade->setSimulationParameters(_parameters);
    _simulationFacade->setSimulationData(rect);
    _simulationFacade->calcTimesteps(1);
    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(RectLength * RectLength + 1, actualData._cells.size());
}

TEST_F(ConstructorTests, constructFirstCell_completenessCheck_thinCluster)
{
    auto constructorGenome = ConstructorGenomeDescription().mode(0).constructionActivationTime(123).makeSelfCopy();
    auto genome = GenomeDescriptionConverterService::get().convertDescriptionToBytes(GenomeDescription()
                                                                          .header(GenomeHeaderDescription().numBranches(2))
                                                                          .cells(
                                                                              {CellGenomeDescription(),
                                                                               CellGenomeDescription(),
                                                                               CellGenomeDescription().cellType(constructorGenome),
                                                                               CellGenomeDescription(),
                                                                               CellGenomeDescription()}));

    CollectionDescription data;
    data.addCells({
        CellDescription()
            .id(1)
            .pos({10.0f, 10.0f})
            .energy(_parameters.normalCellEnergy.value[0] * 3)
            .cellType(ConstructorDescription().genome(genome).numInheritedGenomeNodes(5))
            .creatureId(1),
        CellDescription()
            .id(2)
            .pos({10.0f, 9.0f})
            .energy(100)
            .cellType(OscillatorDescription())
            .creatureId(1),
        CellDescription()
            .id(3)
            .pos({10.0f, 8.0f})
            .energy(100)
            .cellType(OscillatorDescription())
            .creatureId(1),
        CellDescription()
            .id(4)
            .pos({10.0f, 11.0f})
            .energy(100)
            .cellType(OscillatorDescription())
            .creatureId(1),
        CellDescription()
            .id(5)
            .pos({10.0f, 12.0f})
            .energy(100)
            .cellType(OscillatorDescription())
            .creatureId(1),
        CellDescription()
            .id(6)
            .pos({11.0f, 10.0f})
            .energy(100)
            .cellType(OscillatorDescription())
            .creatureId(2),
    });
    data.addConnection(1, 2);
    data.addConnection(1, 6);
    data.addConnection(1, 4);
    data.addConnection(2, 3);
    data.addConnection(4, 5);

    auto& firstCell = data._cells.at(0);
    while (true) {
        if (firstCell._connections.at(0)._cellId != 2) {
            std::vector<ConnectionDescription> newConnections;
            newConnections.emplace_back(firstCell._connections.back());
            for (int j = 0; j < firstCell._connections.size() - 1; ++j) {
                newConnections.emplace_back(firstCell._connections.at(j));
            }
            firstCell._connections = newConnections;
        } else {
            break;
        }
    }

    _parameters.constructorCompletenessCheck.value = true;
    _simulationFacade->setSimulationParameters(_parameters);
    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);
    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(7, actualData._cells.size());
}

/**
 * Completeness check needs to inspect cells under construction because when a constructor is finished its construction
 * is still in state "under construction" for some time steps but needs to be inspected
 *
 * UPDATE: Test does not make sense with new completeness check
 */
TEST_F(ConstructorTests, DISABLED_constructFirstCell_completenessCheck_underConstruction)
{
    auto constructorGenome = ConstructorGenomeDescription().mode(0).constructionActivationTime(123).makeSelfCopy();
    auto genome =
        GenomeDescriptionConverterService::get().convertDescriptionToBytes(GenomeDescription().cells({CellGenomeDescription().cellType(constructorGenome)}));
    auto otherGenome = GenomeDescriptionConverterService::get().convertDescriptionToBytes(GenomeDescription().cells({CellGenomeDescription()}));

    CollectionDescription data;
    data.addCells({
        CellDescription()
            .id(1)
            .pos({10.0f, 10.0f})
            .energy(_parameters.normalCellEnergy.value[0] * 3)
            .cellType(ConstructorDescription().genome(genome)),
        CellDescription()
            .id(2)
            .pos({11.0f, 10.0f})
            .energy(100)
            .cellType(ConstructorDescription().genome(otherGenome).genomeCurrentBranch(1)),
        CellDescription()
            .id(3)
            .pos({12.0f, 10.0f})
            .energy(100)
            .livingState(LivingState_UnderConstruction)
            .cellType(ConstructorDescription().genome(otherGenome).genomeCurrentBranch(0)),
    });
    data.addConnection(1, 2);
    data.addConnection(2, 3);

    _parameters.constructorCompletenessCheck.value = true;
    _simulationFacade->setSimulationParameters(_parameters);
    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);
    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(3, actualData._cells.size());
}

TEST_F(ConstructorTests, constructFirstCell_noSeparation)
{
    auto genome = GenomeDescriptionConverterService::get().convertDescriptionToBytes(
        GenomeDescription()
            .header(GenomeHeaderDescription().separateConstruction(false).stiffness(0.35f))
            .cells({CellGenomeDescription().cellType(ConstructorGenomeDescription().makeSelfCopy()).color(2)}));

    CollectionDescription data;
    data.addCell(
        CellDescription()
            .id(1)
            .pos({10.0f, 10.0f})
            .energy(_parameters.normalCellEnergy.value[0] * 3)
            .cellType(ConstructorDescription().genome(genome).constructionActivationTime(123)));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);
    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(2, actualData._cells.size());
    auto actualHostCell = getCell(actualData, 1);
    auto actualConstructedCell = getOtherCell(actualData, 1);

    EXPECT_EQ(1, actualHostCell._connections.size());

    auto const& actualConstructor = std::get<ConstructorDescription>(actualHostCell._cellTypeData);
    EXPECT_EQ(0, actualConstructor._genomeCurrentNodeIndex);
    EXPECT_EQ(0, actualConstructor._genomeCurrentRepetition);
    EXPECT_EQ(1, actualConstructor._genomeCurrentBranch);
    EXPECT_TRUE(approxCompare(_parameters.normalCellEnergy.value[0] * 2, actualHostCell._energy));
    EXPECT_TRUE(approxCompare(1.0f, actualHostCell._signal->_channels[0]));
    EXPECT_EQ(LivingState_Activating, actualConstructedCell._livingState);

    EXPECT_EQ(1, actualConstructedCell._connections.size());
    EXPECT_EQ(2, actualConstructedCell._color);
    EXPECT_EQ(CellType_Constructor, actualConstructedCell.getCellType());
    EXPECT_EQ(123, actualConstructedCell._activationTime);
    EXPECT_TRUE(approxCompare(0.35f, actualConstructedCell._stiffness, 0.01f));
    EXPECT_TRUE(approxCompare(_parameters.normalCellEnergy.value[0], actualConstructedCell._energy));
    EXPECT_TRUE(approxCompare(1.0f, Math::length(actualHostCell._pos - actualConstructedCell._pos)));
}

TEST_F(ConstructorTests, constructFirstCell_notFinished)
{
    auto genome = GenomeDescriptionConverterService::get().convertDescriptionToBytes(
        GenomeDescription()
            .header(GenomeHeaderDescription().separateConstruction(false)).cells({CellGenomeDescription(), CellGenomeDescription()}));

    CollectionDescription data;
    data.addCell(CellDescription()
                     .id(1)
                     .pos({10.0f, 10.0f})
                     .energy(_parameters.normalCellEnergy.value[0] * 3)
                     .cellType(ConstructorDescription().genome(genome)));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);
    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(2, actualData._cells.size());
    auto actualHostCell = getCell(actualData, 1);
    auto actualConstructedCell = getOtherCell(actualData, 1);
    auto const& actualConstructor = std::get<ConstructorDescription>(actualHostCell._cellTypeData);
    EXPECT_EQ(1, actualConstructor._genomeCurrentNodeIndex);
    EXPECT_EQ(0, actualConstructor._genomeCurrentRepetition);
    EXPECT_EQ(0, actualConstructor._genomeCurrentBranch);

    EXPECT_EQ(1, actualHostCell._connections.size());
    EXPECT_EQ(LivingState_Ready, actualHostCell._livingState);

    EXPECT_EQ(1, actualConstructedCell._connections.size());
    EXPECT_EQ(LivingState_UnderConstruction, actualConstructedCell._livingState);
}

TEST_F(ConstructorTests, constructFirstCell_separation)
{
    auto genome = GenomeDescriptionConverterService::get().convertDescriptionToBytes(
        GenomeDescription().header(GenomeHeaderDescription().separateConstruction(true)).cells({CellGenomeDescription()}));

    CollectionDescription data;
    data.addCell(CellDescription()
                     .id(1)
                     .energy(_parameters.normalCellEnergy.value[0] * 3)
                     .cellType(ConstructorDescription()
                                          .genome(genome)));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);
    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(2, actualData._cells.size());
    auto actualHostCell = getCell(actualData, 1);
    auto actualConstructedCell = getOtherCell(actualData, 1);

    EXPECT_EQ(0, actualHostCell._connections.size());
    auto const& actualConstructor = std::get<ConstructorDescription>(actualHostCell._cellTypeData);
    EXPECT_EQ(0, actualConstructor._genomeCurrentNodeIndex);
    EXPECT_EQ(0, actualConstructor._genomeCurrentRepetition);
    EXPECT_EQ(0, actualConstructor._genomeCurrentBranch);

    EXPECT_EQ(0, actualConstructedCell._connections.size());
    EXPECT_EQ(LivingState_Activating, actualConstructedCell._livingState);
}

TEST_F(ConstructorTests, constructFirstCell_manualConstruction)
{
    auto genome = GenomeDescriptionConverterService::get().convertDescriptionToBytes(
        GenomeDescription().cells({CellGenomeDescription()}));

    CollectionDescription data;
    data.addCells({
       CellDescription()
            .id(1)
            .pos({10.0f, 10.0f})
            .energy(_parameters.normalCellEnergy.value[0] * 3)
            .cellType(ConstructorDescription().autoTriggerInterval(0).genome(genome)),
        CellDescription()
             .id(2)
             .pos({11.0f, 10.0f})
             .energy(100)
             .cellType(OscillatorDescription())
             .signal({1, 0, 0, 0, 0, 0, 0, 0})
    });
    data.addConnection(1, 2);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);
    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(3, actualData._cells.size());
    auto actualHostCell = getCell(actualData, 1);
    auto actualConstructedCell = getOtherCell(actualData, {1, 2});

    EXPECT_EQ(1, actualHostCell._connections.size());

    EXPECT_EQ(0, actualConstructedCell._connections.size());
    EXPECT_EQ(LivingState_Activating, actualConstructedCell._livingState);

    EXPECT_TRUE(approxCompare(10.0f - 1.0f, actualConstructedCell._pos.x));
    EXPECT_TRUE(approxCompare(10.0f, actualConstructedCell._pos.y));
}

TEST_F(ConstructorTests, constructFirstCell_differentAngle1)
{
    auto genome = GenomeDescriptionConverterService::get().convertDescriptionToBytes(GenomeDescription().cells({CellGenomeDescription()}));

    CollectionDescription data;
    data.addCells({
        CellDescription()
             .id(1)
             .pos({10.0f, 10.0f})
             .energy(_parameters.normalCellEnergy.value[0] * 3)
             .cellType(ConstructorDescription().autoTriggerInterval(0).genome(genome).constructionAngle1(90.0f)),
        CellDescription()
             .id(2)
             .pos({11.0f, 10.0f})
             .energy(100)
             .cellType(OscillatorDescription())
             .signal({1, 0, 0, 0, 0, 0, 0, 0})});
    data.addConnection(1, 2);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);
    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(3, actualData._cells.size());
    auto actualHostCell = getCell(actualData, 1);
    auto actualConstructedCell = getOtherCell(actualData, {1, 2});

    EXPECT_TRUE(approxCompare(10.0f, actualConstructedCell._pos.x));
    EXPECT_TRUE(approxCompare(10.0f - 1.0f, actualConstructedCell._pos.y));
}

TEST_F(ConstructorTests, constructFirstCell_differentAngle2)
{
    auto genome = GenomeDescriptionConverterService::get().convertDescriptionToBytes(GenomeDescription().cells({CellGenomeDescription().referenceAngle(-90.0f)}));

    CollectionDescription data;
    data.addCells(
        {CellDescription()
             .id(1)
             .pos({10.0f, 10.0f})
             .energy(_parameters.normalCellEnergy.value[0] * 3)
             .cellType(ConstructorDescription().autoTriggerInterval(0).genome(genome).constructionAngle1(-90.0f)),
         CellDescription()
             .id(2)
             .pos({11.0f, 10.0f})
             .energy(100)
             .cellType(OscillatorDescription())
             .signal({1, 0, 0, 0, 0, 0, 0, 0})});
    data.addConnection(1, 2);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);
    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(3, actualData._cells.size());
    auto actualHostCell = getCell(actualData, 1);
    auto actualConstructedCell = getOtherCell(actualData, {1, 2});

    EXPECT_TRUE(approxCompare(10.0f, actualConstructedCell._pos.x));
    EXPECT_TRUE(approxCompare(10.0f + 1.0f, actualConstructedCell._pos.y));
}

TEST_F(ConstructorTests, constructNeuronCell)
{
    auto nn = NeuralNetworkGenomeDescription();
    nn.weight(1, 7, 3.9f);
    nn.weight(7, 1, -1.9f);
    nn._biases[3] = 3.8f;

    auto genome = GenomeDescriptionConverterService::get().convertDescriptionToBytes(GenomeDescription().cells({CellGenomeDescription().neuralNetwork(nn)}));

    CollectionDescription data;
    data.addCell(
        CellDescription()
                     .id(1)
                     .energy(_parameters.normalCellEnergy.value[0] * 3)
                     .cellType(ConstructorDescription().genome(genome)));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);
    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(2, actualData._cells.size());
    auto actualConstructedCell = getOtherCell(actualData, 1);

    EXPECT_EQ(CellType_Base, actualConstructedCell.getCellType());

    for (auto const& [weight, actualWeight] : boost::combine(nn._weights, actualConstructedCell._neuralNetwork->_weights)) {
        EXPECT_TRUE(lowPrecisionCompare(weight, actualWeight));
    }
    for (int i = 0; i < MAX_CHANNELS; ++i) {
        EXPECT_TRUE(lowPrecisionCompare(nn._biases[i], actualConstructedCell._neuralNetwork->_biases[i]));
    }
}

TEST_F(ConstructorTests, constructConstructorCell)
{
    auto constructorGenome = ConstructorGenomeDescription().mode(0).constructionActivationTime(123).genome(createRandomGenome(MAX_GENOME_BYTES / 2));

    auto genome =
        GenomeDescriptionConverterService::get().convertDescriptionToBytes(GenomeDescription().cells({CellGenomeDescription().cellType(constructorGenome)}));

    CollectionDescription data;

    data.addCell(CellDescription()
                     .id(1)
                     .energy(_parameters.normalCellEnergy.value[0] * 3)
                     .cellType(ConstructorDescription().genome(genome)));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);
    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(2, actualData._cells.size());
    auto actualConstructedCell = getOtherCell(actualData, 1);

    EXPECT_EQ(CellType_Constructor, actualConstructedCell.getCellType());

    auto actualConstructor = std::get<ConstructorDescription>(actualConstructedCell._cellTypeData);
    EXPECT_EQ(constructorGenome._autoTriggerInterval, actualConstructor._autoTriggerInterval);
    EXPECT_EQ(constructorGenome._constructionActivationTime, actualConstructor._constructionActivationTime);
    EXPECT_EQ(constructorGenome.getGenomeData(), actualConstructor._genome);
}

TEST_F(ConstructorTests, constructOscillatorCell)
{
    auto oscillatorDesc = OscillatorGenomeDescription().autoTriggerInterval(2).alternationInterval(4);
    auto genome = GenomeDescriptionConverterService::get().convertDescriptionToBytes(GenomeDescription().cells({CellGenomeDescription().cellType(oscillatorDesc)}));

    CollectionDescription data;
    data.addCell(CellDescription()
                     .id(1)
                     .energy(_parameters.normalCellEnergy.value[0] * 3)
                     .cellType(ConstructorDescription().genome(genome)));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);
    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(2, actualData._cells.size());
    auto actualConstructedCell = getOtherCell(actualData, 1);
    auto actualOscillator = std::get<OscillatorDescription>(actualConstructedCell._cellTypeData);

    EXPECT_EQ(CellType_Oscillator, actualConstructedCell.getCellType());
    EXPECT_EQ(oscillatorDesc._autoTriggerInterval, actualOscillator._autoTriggerInterval);
    EXPECT_EQ(oscillatorDesc._alternationInterval, actualOscillator._alternationInterval);
}

TEST_F(ConstructorTests, constructAttackerCell)
{
    auto attackerDesc = AttackerGenomeDescription();
    auto genome = GenomeDescriptionConverterService::get().convertDescriptionToBytes(GenomeDescription().cells({CellGenomeDescription().cellType(attackerDesc)}));

    CollectionDescription data;
    data.addCell(CellDescription()
                     .id(1)
                     .energy(_parameters.normalCellEnergy.value[0] * 3)
                     .cellType(ConstructorDescription().genome(genome)));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);
    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(2, actualData._cells.size());
    auto actualConstructedCell = getOtherCell(actualData, 1);

    EXPECT_EQ(CellType_Attacker, actualConstructedCell.getCellType());
}

TEST_F(ConstructorTests, constructDefenderCell)
{
    auto defenderDesc = DefenderGenomeDescription().mode(DefenderMode_DefendAgainstInjector);
    auto genome = GenomeDescriptionConverterService::get().convertDescriptionToBytes(GenomeDescription().cells({CellGenomeDescription().cellType(defenderDesc)}));

    CollectionDescription data;
    data.addCell(CellDescription()
                     .id(1)
                     .energy(_parameters.normalCellEnergy.value[0] * 3)
                     .cellType(ConstructorDescription().genome(genome)));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);
    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(2, actualData._cells.size());
    auto actualConstructedCell = getOtherCell(actualData, 1);

    EXPECT_EQ(CellType_Defender, actualConstructedCell.getCellType());

    auto actualDefender = std::get<DefenderDescription>(actualConstructedCell._cellTypeData);
    EXPECT_EQ(defenderDesc._mode, actualDefender._mode);
}

TEST_F(ConstructorTests, constructTransmitterCell)
{
    auto transmitterDesc = DepotGenomeDescription().mode(EnergyDistributionMode_TransmittersAndConstructors);
    auto genome =
        GenomeDescriptionConverterService::get().convertDescriptionToBytes(GenomeDescription().cells({CellGenomeDescription().cellType(transmitterDesc)}));

    CollectionDescription data;
    data.addCell(CellDescription()
                     .id(1)
                     .energy(_parameters.normalCellEnergy.value[0] * 3)
                     .cellType(ConstructorDescription().genome(genome)));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);
    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(2, actualData._cells.size());
    auto actualConstructedCell = getOtherCell(actualData, 1);

    EXPECT_EQ(CellType_Depot, actualConstructedCell.getCellType());

    auto actualTransmitter = std::get<DepotDescription>(actualConstructedCell._cellTypeData);
    EXPECT_EQ(transmitterDesc._mode, actualTransmitter._mode);
}

//TEST_F(ConstructorTests, constructMuscleCell)
//{
//    auto muscleDesc = MuscleGenomeDescription().mode(BendingGenomeDescription().autoTriggerInterval(3));
//    auto genome = GenomeDescriptionConverterService::get().convertDescriptionToBytes(GenomeDescription().cells({CellGenomeDescription().cellType(muscleDesc)}));
//
//    CollectionDescription data;
//    data.addCell(CellDescription()
//                     .id(1)
//                     .energy(_parameters.cellNormalEnergy[0] * 3)
//                     .cellType(ConstructorDescription().genome(genome)));
//
//    _simulationFacade->setSimulationData(data);
//    _simulationFacade->calcTimesteps(1);
//    auto actualData = _simulationFacade->getSimulationData();
//
//    ASSERT_EQ(2, actualData._cells.size());
//    auto actualConstructedCell = getOtherCell(actualData, 1);
//
//    EXPECT_EQ(CellType_Muscle, actualConstructedCell.getCellType());
//
//    auto actualMuscle = std::get<MuscleDescription>(actualConstructedCell._cellTypeData);
//    EXPECT_EQ(muscleDesc._mode, actualMuscle._mode);
//    EXPECT_EQ(0, actualMuscle._lastBendingDirection);
//    EXPECT_EQ(0.0f, actualMuscle._consecutiveBendingAngle);
//}

TEST_F(ConstructorTests, constructSensorCell)
{
    auto sensorDesc = SensorGenomeDescription().restrictToColor(2).minDensity(0.5f).restrictToMutants(SensorRestrictToMutants_RestrictToFreeCells);
    auto genome = GenomeDescriptionConverterService::get().convertDescriptionToBytes(GenomeDescription().cells({CellGenomeDescription().cellType(sensorDesc)}));

    CollectionDescription data;
    data.addCell(CellDescription()
                     .id(1)
                     .energy(_parameters.normalCellEnergy.value[0] * 3)
                     .cellType(ConstructorDescription().genome(genome)));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);
    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(2, actualData._cells.size());
    auto actualConstructedCell = getOtherCell(actualData, 1);

    EXPECT_EQ(CellType_Sensor, actualConstructedCell.getCellType());

    auto actualSensor = std::get<SensorDescription>(actualConstructedCell._cellTypeData);
    EXPECT_TRUE(lowPrecisionCompare(sensorDesc._minDensity, actualSensor._minDensity));
    EXPECT_EQ(sensorDesc._restrictToColor, actualSensor._restrictToColor);
    EXPECT_EQ(sensorDesc._restrictToMutants, actualSensor._restrictToMutants);
}

TEST_F(ConstructorTests, constructInjectorCell)
{
    auto injectorDesc = InjectorGenomeDescription().mode(InjectorMode_InjectOnlyEmptyCells).genome(createRandomGenome(MAX_GENOME_BYTES / 2));
    auto genome = GenomeDescriptionConverterService::get().convertDescriptionToBytes(GenomeDescription().cells({CellGenomeDescription().cellType(injectorDesc)}));

    CollectionDescription data;
    data.addCell(CellDescription()
                     .id(1)
                     .energy(_parameters.normalCellEnergy.value[0] * 3)
                     .cellType(ConstructorDescription().genome(genome)));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);
    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(2, actualData._cells.size());
    auto actualConstructedCell = getOtherCell(actualData, 1);

    EXPECT_EQ(CellType_Injector, actualConstructedCell.getCellType());

    auto actualInjector = std::get<InjectorDescription>(actualConstructedCell._cellTypeData);
    EXPECT_EQ(injectorDesc._mode, actualInjector._mode);
    EXPECT_EQ(injectorDesc.getGenomeData(), actualInjector._genome);
}

TEST_F(ConstructorTests, constructReconnectorCell)
{
    auto reconnectorDesc = ReconnectorGenomeDescription().restrictToColor(2).restrictToMutants(ReconnectorRestrictToMutants_RestrictToSameMutants);
    auto genome =
        GenomeDescriptionConverterService::get().convertDescriptionToBytes(GenomeDescription().cells({CellGenomeDescription().cellType(reconnectorDesc)}));

    CollectionDescription data;
    data.addCell(CellDescription()
                     .id(1)
                     .energy(_parameters.normalCellEnergy.value[0] * 3)
                     .cellType(ConstructorDescription().genome(genome)));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);
    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(2, actualData._cells.size());
    auto actualConstructedCell = getOtherCell(actualData, 1);

    EXPECT_EQ(CellType_Reconnector, actualConstructedCell.getCellType());

    auto actualReconnector = std::get<ReconnectorDescription>(actualConstructedCell._cellTypeData);
    EXPECT_EQ(reconnectorDesc._restrictToColor, actualReconnector._restrictToColor);
    EXPECT_EQ(reconnectorDesc._restrictToMutants, actualReconnector._restrictToMutants);
}

TEST_F(ConstructorTests, constructDetonatorCell)
{
    auto detonatorDesc = DetonatorGenomeDescription().countdown(25);
    auto genome = GenomeDescriptionConverterService::get().convertDescriptionToBytes(GenomeDescription().cells({CellGenomeDescription().cellType(detonatorDesc)}));

    CollectionDescription data;
    data.addCell(CellDescription()
                     .id(1)
                     .energy(_parameters.normalCellEnergy.value[0] * 3)
                     .cellType(ConstructorDescription().genome(genome)));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);
    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(2, actualData._cells.size());
    auto actualConstructedCell = getOtherCell(actualData, 1);

    EXPECT_EQ(CellType_Detonator, actualConstructedCell.getCellType());

    auto actualDetonator = std::get<DetonatorDescription>(actualConstructedCell._cellTypeData);
    EXPECT_EQ(detonatorDesc._countdown, actualDetonator._countdown);
}

TEST_F(ConstructorTests, constructConstructorCell_nestingGenomeTooLarge)
{
    auto constructedConstructor = ConstructorGenomeDescription().mode(0).genome(createRandomGenome(MAX_GENOME_BYTES));
    auto genome =
        GenomeDescriptionConverterService::get().convertDescriptionToBytes(GenomeDescription().cells({CellGenomeDescription().cellType(constructedConstructor)}));


    CollectionDescription data;

    data.addCell(CellDescription()
                     .id(1)
                     .energy(_parameters.normalCellEnergy.value[0] * 3)
                     .cellType(ConstructorDescription().genome(genome)));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);
    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(2, actualData._cells.size());
    auto actualCell = getOtherCell(actualData, 1);
    auto actualConstructedCell = getOtherCell(actualData, 1);

    EXPECT_EQ(CellType_Constructor, actualConstructedCell.getCellType());

    auto actualConstructor = std::get<ConstructorDescription>(actualCell._cellTypeData);
    auto actualConstructedConstructor = std::get<ConstructorDescription>(actualConstructedCell._cellTypeData);
    EXPECT_TRUE(actualConstructor._genome.size() <= MAX_GENOME_BYTES);
    EXPECT_TRUE(constructedConstructor.getGenomeData().size() <= MAX_GENOME_BYTES);
}

TEST_F(ConstructorTests, constructConstructorCell_copyGenome)
{
    auto constructedConstructor = ConstructorGenomeDescription().mode(0).makeSelfCopy();

    auto genome =
        GenomeDescriptionConverterService::get().convertDescriptionToBytes(GenomeDescription().cells({CellGenomeDescription().cellType(constructedConstructor)}));

    CollectionDescription data;
    data.addCell(CellDescription()
                     .id(1)
                     .energy(_parameters.normalCellEnergy.value[0] * 3)
                     .cellType(ConstructorDescription().genome(genome)));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);
    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(2, actualData._cells.size());
    auto actualConstructedCell = getOtherCell(actualData, 1);

    EXPECT_EQ(CellType_Constructor, actualConstructedCell.getCellType());

    auto actualConstructor = std::get<ConstructorDescription>(actualConstructedCell._cellTypeData);
    EXPECT_EQ(genome, actualConstructor._genome);
}

TEST_F(ConstructorTests, constructSecondCell_separation)
{
    auto genome = GenomeDescriptionConverterService::get().convertDescriptionToBytes(
        GenomeDescription().header(GenomeHeaderDescription().separateConstruction(true)).cells({CellGenomeDescription(), CellGenomeDescription()}));

    CollectionDescription data;
    data.addCells({
        CellDescription()
            .id(1)
            .pos({10.0f, 10.0f})
            .energy(_parameters.normalCellEnergy.value[0] * 3)
            .cellType(ConstructorDescription().genomeCurrentNodeIndex(1).genome(genome)),
        CellDescription()
            .id(2)
            .pos({10.0f - getOffspringDistance(), 10.0f})
            .energy(100)
            .cellType(OscillatorDescription())
            .livingState(LivingState_UnderConstruction),
    });
    data.addConnection(1, 2);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);
    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(3, actualData._cells.size());
    auto actualHostCell = getCell(actualData, 1);
    auto actualPrevConstructedCell = getCell(actualData, 2);
    auto actualConstructedCell = getOtherCell(actualData, {1, 2});

    EXPECT_EQ(0, actualHostCell._connections.size());
    EXPECT_EQ(LivingState_Ready, actualHostCell._livingState);

    ASSERT_EQ(1, actualPrevConstructedCell._connections.size());
    EXPECT_EQ(LivingState_UnderConstruction, actualPrevConstructedCell._livingState);
    EXPECT_TRUE(lowPrecisionCompare(1.0f, actualPrevConstructedCell._connections[0]._distance));

    ASSERT_EQ(1, actualConstructedCell._connections.size());
    EXPECT_EQ(LivingState_Activating, actualConstructedCell._livingState);
    EXPECT_TRUE(lowPrecisionCompare(1.0f, actualConstructedCell._connections[0]._distance));
}

TEST_F(ConstructorTests, constructSecondCell_constructionStateTransitions)
{
    auto genome = GenomeDescriptionConverterService::get().convertDescriptionToBytes(
        GenomeDescription().header(GenomeHeaderDescription().separateConstruction(true)).cells({CellGenomeDescription(), CellGenomeDescription()}));

    CollectionDescription data;
    data.addCells({
        CellDescription()
            .id(1)
            .pos({10.0f, 10.0f})
            .energy(_parameters.normalCellEnergy.value[0] * 3)
            .cellType(ConstructorDescription().genomeCurrentNodeIndex(1).genome(genome)),
        CellDescription()
            .id(2)
            .pos({10.0f - getOffspringDistance(), 10.0f})
            .energy(100)
            .cellType(OscillatorDescription())
            .livingState(LivingState_UnderConstruction),
    });
    data.addConnection(1, 2);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);
    _simulationFacade->calcTimesteps(1);
    {
        auto actualData = _simulationFacade->getSimulationData();

        ASSERT_EQ(3, actualData._cells.size());
        auto actualHostCell = getCell(actualData, 1);
        auto actualPrevConstructedCell = getCell(actualData, 2);
        auto actualConstructedCell = getOtherCell(actualData, {1, 2});

        EXPECT_EQ(LivingState_Ready, actualHostCell._livingState);
        EXPECT_EQ(LivingState_Activating, actualPrevConstructedCell._livingState);
        EXPECT_EQ(LivingState_Ready, actualConstructedCell._livingState);
    }
    _simulationFacade->calcTimesteps(1);
    {
        auto actualData = _simulationFacade->getSimulationData();

        ASSERT_EQ(3, actualData._cells.size());
        auto actualHostCell = getCell(actualData, 1);
        auto actualPrevConstructedCell = getCell(actualData, 2);
        auto actualConstructedCell = getOtherCell(actualData, {1, 2});

        EXPECT_EQ(LivingState_Ready, actualHostCell._livingState);
        EXPECT_EQ(LivingState_Ready, actualPrevConstructedCell._livingState);
        EXPECT_EQ(LivingState_Ready, actualConstructedCell._livingState);
    }
}

TEST_F(ConstructorTests, constructSecondCell_noSeparation)
{
    auto genome = GenomeDescriptionConverterService::get().convertDescriptionToBytes(
        GenomeDescription().header(GenomeHeaderDescription().separateConstruction(false)).cells({CellGenomeDescription(), CellGenomeDescription()}));

    CollectionDescription data;
    data.addCells({
        CellDescription()
            .id(1)
            .pos({10.0f, 10.0f})
            .energy(_parameters.normalCellEnergy.value[0] * 3)
            .cellType(ConstructorDescription().genomeCurrentNodeIndex(1).genome(genome)),
        CellDescription()
            .id(2)
            .pos({10.0f - getOffspringDistance(), 10.0f})
            .energy(100)
            .cellType(OscillatorDescription())
            .livingState(LivingState_UnderConstruction),
    });
    data.addConnection(1, 2);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);
    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(3, actualData._cells.size());
    auto actualHostCell = getCell(actualData, 1);
    auto actualPrevConstructedCell = getCell(actualData, 2);
    auto actualConstructedCell = getOtherCell(actualData, {1, 2});

    EXPECT_EQ(1, actualHostCell._connections.size());
    EXPECT_EQ(LivingState_Ready, actualHostCell._livingState);
    EXPECT_TRUE(lowPrecisionCompare(1.0f, actualPrevConstructedCell._connections[0]._distance));

    ASSERT_EQ(2, actualConstructedCell._connections.size());
    EXPECT_EQ(LivingState_Activating, actualConstructedCell._livingState);
    std::map<uint64_t, ConnectionDescription> connectionById;
    for (auto const& connection : actualConstructedCell._connections) {
        connectionById.emplace(connection._cellId, connection);
    }
    EXPECT_TRUE(lowPrecisionCompare(1.0f, connectionById.at(1)._distance));
    EXPECT_TRUE(approxCompare(180.0f, connectionById.at(1)._angleFromPrevious));
    EXPECT_TRUE(lowPrecisionCompare(1.0f, connectionById.at(2)._distance));
    EXPECT_TRUE(approxCompare(180.0f, connectionById.at(2)._angleFromPrevious));

    ASSERT_EQ(1, actualPrevConstructedCell._connections.size());
    EXPECT_EQ(LivingState_UnderConstruction, actualPrevConstructedCell._livingState);
    EXPECT_TRUE(lowPrecisionCompare(1.0f, actualPrevConstructedCell._connections[0]._distance));
}

TEST_F(ConstructorTests, constructSecondCell_noSpace)
{
    auto genome = GenomeDescriptionConverterService::get().convertDescriptionToBytes(
        GenomeDescription().header(GenomeHeaderDescription().separateConstruction(false)).cells({CellGenomeDescription(), CellGenomeDescription()}));

    CollectionDescription data;
    data.addCells({
        CellDescription()
            .id(1)
            .pos({10.0f, 10.0f})
            .energy(_parameters.normalCellEnergy.value[0] * 3)
            .cellType(ConstructorDescription().genomeCurrentNodeIndex(1).genome(genome)),
        CellDescription()
            .id(2)
            .pos({10.0f - 1.0f - _parameters.minCellDistance.value / 2, 10.0f})
            .energy(100)
            .cellType(OscillatorDescription())
            .livingState(LivingState_UnderConstruction),
    });
    data.addConnection(1, 2);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);
    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(2, actualData._cells.size());
    auto actualHostCell = getCell(actualData, 1);
    auto actualPrevConstructedCell = getCell(actualData, 2);

    EXPECT_EQ(1, actualHostCell._connections.size());
    EXPECT_TRUE(approxCompare(0.0f, actualHostCell._signal->_channels[0]));
    ASSERT_EQ(1, actualPrevConstructedCell._connections.size());
    auto actualConstructor = std::get<ConstructorDescription>(actualHostCell._cellTypeData);
    EXPECT_EQ(1, actualConstructor._genomeCurrentNodeIndex);
}

TEST_F(ConstructorTests, constructSecondCell_notFinished)
{
    auto genome = GenomeDescriptionConverterService::get().convertDescriptionToBytes(GenomeDescription()
                                                                .header(GenomeHeaderDescription().separateConstruction(false))
                                                                .cells({CellGenomeDescription(), CellGenomeDescription(), CellGenomeDescription()}));

    CollectionDescription data;
    data.addCells({
        CellDescription()
            .id(1)
            .pos({10.0f, 10.0f})
            .energy(_parameters.normalCellEnergy.value[0] * 3)
            .cellType(ConstructorDescription().genomeCurrentNodeIndex(1).genome(genome)),
        CellDescription()
            .id(2)
            .pos({10.0f - getOffspringDistance(), 10.0f})
            .energy(100)
            .cellType(OscillatorDescription())
            .livingState(LivingState_UnderConstruction),
    });
    data.addConnection(1, 2);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);
    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(3, actualData._cells.size());
    auto actualHostCell = getCell(actualData, 1);
    auto actualPrevConstructedCell = getCell(actualData, 2);
    auto actualConstructedCell = getOtherCell(actualData, {1, 2});

    EXPECT_EQ(1, actualHostCell._connections.size());
    EXPECT_EQ(LivingState_Ready, actualHostCell._livingState);

    ASSERT_EQ(2, actualConstructedCell._connections.size());
    EXPECT_EQ(LivingState_UnderConstruction, actualConstructedCell._livingState);

    ASSERT_EQ(1, actualPrevConstructedCell._connections.size());
    EXPECT_EQ(LivingState_UnderConstruction, actualPrevConstructedCell._livingState);
}

TEST_F(ConstructorTests, constructSecondCell_differentAngle1)
{
    auto genome = GenomeDescriptionConverterService::get().convertDescriptionToBytes(
        GenomeDescription().header(GenomeHeaderDescription().separateConstruction(false)).cells({CellGenomeDescription(), CellGenomeDescription()}));

    CollectionDescription data;
    data.addCells({
        CellDescription()
            .id(1)
            .pos({10.0f, 10.0f})
            .energy(_parameters.normalCellEnergy.value[0] * 3)
            .cellType(ConstructorDescription().genome(genome).constructionAngle2(90.0f).genomeCurrentNodeIndex(1)),
        CellDescription()
            .id(2)
            .pos({10.0f - getOffspringDistance(), 10.0f})
            .energy(100)
            .cellType(OscillatorDescription())
            .livingState(LivingState_UnderConstruction),
    });
    data.addConnection(1, 2);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);
    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(3, actualData._cells.size());
    auto actualHostCell = getCell(actualData, 1);
    auto actualPrevConstructedCell = getCell(actualData, 2);
    auto actualConstructedCell = getOtherCell(actualData, {1, 2});

    EXPECT_EQ(1, actualHostCell._connections.size());

    ASSERT_EQ(2, actualConstructedCell._connections.size());
    std::map<uint64_t, ConnectionDescription> connectionById;
    for (auto const& connection : actualConstructedCell._connections) {
        connectionById.emplace(connection._cellId, connection);
    }
    EXPECT_TRUE(lowPrecisionCompare(1.0f, connectionById.at(1)._distance));
    EXPECT_TRUE(lowPrecisionCompare(270.0f, connectionById.at(1)._angleFromPrevious));
    EXPECT_TRUE(lowPrecisionCompare(1.0f, connectionById.at(2)._distance));
    EXPECT_TRUE(lowPrecisionCompare(90.0f, connectionById.at(2)._angleFromPrevious));

    ASSERT_EQ(1, actualPrevConstructedCell._connections.size());
}

TEST_F(ConstructorTests, constructSecondCell_differentAngle2)
{
    auto genome = GenomeDescriptionConverterService::get().convertDescriptionToBytes(
        GenomeDescription().header(GenomeHeaderDescription().separateConstruction(false)).cells({CellGenomeDescription(), CellGenomeDescription()}));

    CollectionDescription data;
    data.addCells({
        CellDescription()
            .id(1)
            .pos({10.0f, 10.0f})
            .energy(_parameters.normalCellEnergy.value[0] * 3)
            .cellType(ConstructorDescription().genome(genome).constructionAngle2(-90.0f).genomeCurrentNodeIndex(1)),
        CellDescription()
            .id(2)
            .pos({10.0f - getOffspringDistance(), 10.0f})
            .energy(100)
            .cellType(OscillatorDescription())
            .livingState(LivingState_UnderConstruction),
    });
    data.addConnection(1, 2);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);
    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(3, actualData._cells.size());
    auto actualHostCell = getCell(actualData, 1);
    auto actualPrevConstructedCell = getCell(actualData, 2);
    auto actualConstructedCell = getOtherCell(actualData, {1, 2});

    EXPECT_EQ(1, actualHostCell._connections.size());

    ASSERT_EQ(2, actualConstructedCell._connections.size());
    std::map<uint64_t, ConnectionDescription> connectionById;
    for (auto const& connection : actualConstructedCell._connections) {
        connectionById.emplace(connection._cellId, connection);
    }
    EXPECT_TRUE(lowPrecisionCompare(1.0f, connectionById.at(1)._distance));
    EXPECT_TRUE(lowPrecisionCompare(90.0f, connectionById.at(1)._angleFromPrevious));
    EXPECT_TRUE(lowPrecisionCompare(1.0f, connectionById.at(2)._distance));
    EXPECT_TRUE(lowPrecisionCompare(270.0f, connectionById.at(2)._angleFromPrevious));

    ASSERT_EQ(1, actualPrevConstructedCell._connections.size());
}

TEST_F(ConstructorTests, constructSecondCell_twoCellGenome_infiniteRepetitions)
{
    auto genome = GenomeDescriptionConverterService::get().convertDescriptionToBytes(
        GenomeDescription().header(GenomeHeaderDescription().infiniteRepetitions()).cells({CellGenomeDescription(), CellGenomeDescription()}));

    CollectionDescription data;
    data.addCells({
        CellDescription()
            .id(1)
            .pos({10.0f - getOffspringDistance(), 10.0f})
            .energy(_parameters.normalCellEnergy.value[0] * 3)
            .cellType(ConstructorDescription().genomeCurrentNodeIndex(1).genome(genome)),
        CellDescription().id(2).pos({11.0f, 10.0f}).livingState(LivingState_UnderConstruction),
    });
    data.addConnection(1, 2);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    ASSERT_EQ(3, actualData._cells.size());

    auto actualHostCell = getCell(actualData, 1);
    auto actualConstructedCell = getOtherCell(actualData, {1, 2});
    EXPECT_EQ(0, std::get<ConstructorDescription>(actualHostCell._cellTypeData)._genomeCurrentBranch);
    EXPECT_EQ(LivingState_Activating, actualConstructedCell._livingState);
}

TEST_F(ConstructorTests, constructThirdCell_multipleConnections_upperPart)
{
    auto genome = GenomeDescriptionConverterService::get().convertDescriptionToBytes(GenomeDescription()
                                                                .header(GenomeHeaderDescription().separateConstruction(false))
                                                                .cells({CellGenomeDescription(), CellGenomeDescription(), CellGenomeDescription()}));

    CollectionDescription data;
    data.addCells({
        CellDescription()
            .id(1)
            .pos({10.0f, 10.0f})
            .energy(_parameters.normalCellEnergy.value[0] * 3)
            .cellType(ConstructorDescription().genomeCurrentNodeIndex(2).genome(genome)),
        CellDescription()
            .id(2)
            .pos({10.0f - getOffspringDistance(), 10.0f})
            .energy(100)
            .cellType(OscillatorDescription())
            .livingState(LivingState_UnderConstruction),
        CellDescription()
            .id(3)
            .pos({10.0f - getOffspringDistance(), 9.0f})
            .energy(100)
            .cellType(OscillatorDescription())
            .livingState(LivingState_UnderConstruction),
        CellDescription().id(4).pos({10.0f, 9.5f}).energy(_parameters.normalCellEnergy.value[0] * 3),
        CellDescription().id(5).pos({10.0f, 9.0f}).energy(_parameters.normalCellEnergy.value[0] * 3),
    });
    data.addConnection(1, 2);
    data.addConnection(2, 3);
    data.addConnection(1, 4);
    data.addConnection(4, 5);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);
    auto actualData = _simulationFacade->getSimulationData();

    EXPECT_EQ(6, actualData._cells.size());
    auto actualHostCell = getCell(actualData, 1);
    auto uninvolvedCell1 = getCell(actualData, 4);
    auto uninvolvedCell2 = getCell(actualData, 5);
    auto actualPrevConstructedCell = getCell(actualData, 2);
    auto actualPrevPrevConstructedCell = getCell(actualData, 3);
    auto actualConstructedCell = getOtherCell(actualData, {1, 2, 3, 4, 5});

    EXPECT_EQ(2, uninvolvedCell1._connections.size());
    EXPECT_EQ(1, uninvolvedCell2._connections.size());
    EXPECT_EQ(2, actualHostCell._connections.size());
    ASSERT_EQ(3, actualConstructedCell._connections.size());
    ASSERT_EQ(2, actualPrevPrevConstructedCell._connections.size());
    ASSERT_EQ(2, actualPrevConstructedCell._connections.size());
}

TEST_F(ConstructorTests, constructThirdCell_multipleConnections_bottomPart)
{
    auto genome = GenomeDescriptionConverterService::get().convertDescriptionToBytes(GenomeDescription()
                                                                .header(GenomeHeaderDescription().separateConstruction(false))
                                                                .cells({CellGenomeDescription(), CellGenomeDescription(), CellGenomeDescription()}));

    CollectionDescription data;
    data.addCells({
        CellDescription()
            .id(1)
            .pos({10.0f, 10.0f})
            .energy(_parameters.normalCellEnergy.value[0] * 3)
            .cellType(ConstructorDescription().genomeCurrentNodeIndex(2).genome(genome)),
        CellDescription()
            .id(2)
            .pos({10.0f - getOffspringDistance(), 10.0f})
            .energy(100)
            .cellType(OscillatorDescription())
            .livingState(LivingState_UnderConstruction),
        CellDescription()
            .id(3)
            .pos({10.0f - getOffspringDistance(), 11.0f})
            .energy(100)
            .cellType(OscillatorDescription())
            .livingState(LivingState_UnderConstruction),
        CellDescription().id(4).pos({10.0f, 10.5f}).energy(_parameters.normalCellEnergy.value[0] * 3),
        CellDescription().id(5).pos({10.0f, 11.0f}).energy(_parameters.normalCellEnergy.value[0] * 3),
    });
    data.addConnection(1, 2);
    data.addConnection(2, 3);
    data.addConnection(1, 4);
    data.addConnection(4, 5);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);
    auto actualData = _simulationFacade->getSimulationData();

    EXPECT_EQ(6, actualData._cells.size());
    auto actualHostCell = getCell(actualData, 1);
    auto uninvolvedCell1 = getCell(actualData, 4);
    auto uninvolvedCell2 = getCell(actualData, 5);
    auto actualPrevConstructedCell = getCell(actualData, 2);
    auto actualPrevPrevConstructedCell = getCell(actualData, 3);
    auto actualConstructedCell = getOtherCell(actualData, {1, 2, 3, 4, 5});

    EXPECT_EQ(2, uninvolvedCell1._connections.size());
    EXPECT_EQ(1, uninvolvedCell2._connections.size());
    EXPECT_EQ(2, actualHostCell._connections.size());
    ASSERT_EQ(3, actualConstructedCell._connections.size());
    ASSERT_EQ(2, actualPrevPrevConstructedCell._connections.size());
    ASSERT_EQ(2, actualPrevConstructedCell._connections.size());
}

TEST_F(ConstructorTests, constructSecondCell_noSeparation_singleConstruction)
{
    auto genome = GenomeDescriptionConverterService::get().convertDescriptionToBytes(
        GenomeDescription().header(GenomeHeaderDescription().separateConstruction(false).numBranches(1)).cells({CellGenomeDescription()}));

    CollectionDescription data;
    data.addCells({
        CellDescription()
            .id(1)
            .pos({10.0f, 10.0f})
            .energy(_parameters.normalCellEnergy.value[0] * 3)
            .cellType(ConstructorDescription().genome(genome)),
        CellDescription()
            .id(2)
            .pos({10.0f - getOffspringDistance(), 10.0f})
            .energy(100)
            .cellType(OscillatorDescription())
            .livingState(LivingState_UnderConstruction),
    });
    data.addConnection(1, 2);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);
    auto actualData = _simulationFacade->getSimulationData();
    auto actualHostCell = getCell(actualData, 1);

    bool found = false;
    for (auto const& connection : actualHostCell._connections) {
        if (connection._cellId != 1 && connection._cellId != 2) {
            EXPECT_TRUE(lowPrecisionCompare(1.0f, connection._distance));
            found = true;
        }
    }
    EXPECT_TRUE(found);
}

TEST_F(ConstructorTests, constructFourthCell_noOverlappingConnection)
{
    auto genome = GenomeDescriptionConverterService::get().convertDescriptionToBytes(GenomeDescription()
                                                                .header(GenomeHeaderDescription().separateConstruction(false))
            .cells({CellGenomeDescription(), CellGenomeDescription(), CellGenomeDescription(), CellGenomeDescription(), CellGenomeDescription()}));

    CollectionDescription data;
    data.addCells({
        CellDescription()
            .id(1)
            .pos({10.0f, 10.0f})
            .energy(_parameters.normalCellEnergy.value[0] * 3)
            .cellType(ConstructorDescription().genomeCurrentNodeIndex(4).genome(genome)),
        CellDescription()
            .id(2)
            .pos({10.0f - getOffspringDistance(), 10.0f})
            .energy(100)
            .cellType(OscillatorDescription())
            .livingState(LivingState_UnderConstruction),
        CellDescription()
            .id(3)
            .pos({10.0f - getOffspringDistance(), 11.0f})
            .energy(100)
            .cellType(OscillatorDescription())
            .livingState(LivingState_UnderConstruction),
        CellDescription()
            .id(4)
            .pos({10.0f - getOffspringDistance() + 1.0f, 11.0f})
            .livingState(LivingState_UnderConstruction),
    });
    data.addConnection(1, 2);
    data.addConnection(2, 3);
    data.addConnection(3, 4);
    data.addConnection(4, 2);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);
    auto actualData = _simulationFacade->getSimulationData();

    EXPECT_EQ(5, actualData._cells.size());
    auto actualHostCell = getCell(actualData, 1);
    auto actualPrevConstructedCell = getCell(actualData, 2);
    auto actualPrevPrevConstructedCell = getCell(actualData, 3);
    auto actualPrevPrevPrevConstructedCell = getCell(actualData, 4);
    auto actualConstructedCell = getOtherCell(actualData, {1, 2, 3, 4});

    EXPECT_EQ(1, actualHostCell._connections.size());
    ASSERT_EQ(3, actualConstructedCell._connections.size());
    ASSERT_EQ(3, actualPrevConstructedCell._connections.size());
    ASSERT_EQ(2, actualPrevPrevConstructedCell._connections.size());
    ASSERT_EQ(3, actualPrevPrevPrevConstructedCell._connections.size());
    EXPECT_TRUE(hasConnection(actualData, actualConstructedCell._id, 1));
    EXPECT_TRUE(hasConnection(actualData, actualConstructedCell._id, 2));
    EXPECT_TRUE(hasConnection(actualData, actualConstructedCell._id, 4));
}

TEST_F(ConstructorTests, constructLastCellFirstRepetition)
{
    auto genome = GenomeDescriptionConverterService::get().convertDescriptionToBytes(
        GenomeDescription().header(GenomeHeaderDescription().numRepetitions(2)).cells({CellGenomeDescription(), CellGenomeDescription()}));

    CollectionDescription data;
    data.addCells({
        CellDescription()
            .id(1)
            .pos({10.0f, 10.0f})
            .energy(_parameters.normalCellEnergy.value[0] * 3)
            .cellType(ConstructorDescription().genome(genome).genomeCurrentNodeIndex(1)),
        CellDescription()
            .id(2)
            .pos({10.0f - getOffspringDistance(), 10.0f})
            .energy(100)
            .cellType(OscillatorDescription())
            .livingState(LivingState_UnderConstruction),
    });
    data.addConnection(1, 2);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);
    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(3, actualData._cells.size());
    auto actualConstructedCell = getOtherCell(actualData, {1, 2});

    EXPECT_EQ(LivingState_UnderConstruction, actualConstructedCell._livingState);
}

TEST_F(ConstructorTests, constructLastCellLastRepetition)
{
    auto genome = GenomeDescriptionConverterService::get().convertDescriptionToBytes(
        GenomeDescription().header(GenomeHeaderDescription().numRepetitions(2)).cells({CellGenomeDescription(), CellGenomeDescription()}));

    CollectionDescription data;
    data.addCells({
        CellDescription()
            .id(1)
            .pos({10.0f, 10.0f})
            .energy(_parameters.normalCellEnergy.value[0] * 3)
            .cellType(ConstructorDescription().genome(genome).genomeCurrentNodeIndex(1).genomeCurrentRepetition(1)),
        CellDescription()
            .id(2)
            .pos({10.0f - getOffspringDistance(), 10.0f})
            .energy(100)
            .cellType(OscillatorDescription())
            .livingState(LivingState_UnderConstruction),
        CellDescription()
            .id(3)
            .pos({9.0f - getOffspringDistance(), 10.0f})
            .energy(100)
            .cellType(OscillatorDescription())
            .livingState(LivingState_UnderConstruction),
    });
    data.addConnection(1, 2);
    data.addConnection(2, 3);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);
    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(4, actualData._cells.size());
    auto actualConstructedCell = getOtherCell(actualData, {1, 2, 3});

    EXPECT_EQ(LivingState_Activating, actualConstructedCell._livingState);
}

TEST_F(ConstructorTests, restartIfNoLastConstructedCellFound)
{
    auto genome =
        GenomeDescriptionConverterService::get().convertDescriptionToBytes(GenomeDescription()
                                                                  .header(GenomeHeaderDescription().numRepetitions(2))
                                                                  .cells({CellGenomeDescription(), CellGenomeDescription(), CellGenomeDescription()}));

    CollectionDescription data;
    data.addCells({
        CellDescription()
            .id(1)
            .pos({10.0f, 10.0f})
            .energy(_parameters.normalCellEnergy.value[0] * 3)
            .cellType(ConstructorDescription().genome(genome).genomeCurrentNodeIndex(1).genomeCurrentRepetition(1)),
        CellDescription()
            .id(2)
            .pos({10.0f - getOffspringDistance(), 10.0f})
            .energy(100)
            .cellType(OscillatorDescription())
            .signal({1, 0, 0, 0, 0, 0, 0, 0}),
    });
    data.addConnection(1, 2);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);
    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(3, actualData._cells.size());
    auto actualHostCell = getCell(actualData, 1);

    auto actualConstructor = std::get<ConstructorDescription>(actualHostCell._cellTypeData);
    EXPECT_EQ(1, actualConstructor._genomeCurrentNodeIndex);
    EXPECT_EQ(0, actualConstructor._genomeCurrentRepetition);
}

TEST_F(ConstructorTests, restartIfLastConstructedCellHasLowNumConnections)
{
    auto genome =
        GenomeDescriptionConverterService::get().convertDescriptionToBytes(GenomeDescription()
                                                                  .header(GenomeHeaderDescription().numRepetitions(2))
                                                                  .cells({CellGenomeDescription(), CellGenomeDescription(), CellGenomeDescription()}));

    CollectionDescription data;
    data.addCells({
        CellDescription()
            .id(1)
            .pos({10.0f, 10.0f})
            .energy(_parameters.normalCellEnergy.value[0] * 3)
            .cellType(ConstructorDescription().genome(genome).genomeCurrentNodeIndex(1).genomeCurrentRepetition(1).numInheritedGenomeNodes(3)),
        CellDescription()
            .id(2)
            .pos({10.0f - getOffspringDistance(), 10.0f})
            .energy(100)
            .cellType(OscillatorDescription())
            .livingState(LivingState_UnderConstruction),
    });
    data.addConnection(1, 2);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);
    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(3, actualData._cells.size());
    auto actualHostCell = getCell(actualData, 1);

    auto actualConstructor = std::get<ConstructorDescription>(actualHostCell._cellTypeData);
    EXPECT_EQ(1, actualConstructor._genomeCurrentNodeIndex);
    EXPECT_EQ(0, actualConstructor._genomeCurrentRepetition);
}

TEST_F(ConstructorTests, allowLargeConstructionAngle1)
{
    auto genome =
        GenomeDescriptionConverterService::get().convertDescriptionToBytes(GenomeDescription().header(GenomeHeaderDescription()).cells({CellGenomeDescription()}));

    CollectionDescription data;
    data.addCells({
        CellDescription()
            .id(1)
            .pos({10.0f, 10.0f})
            .energy(_parameters.normalCellEnergy.value[0] * 3)
            .cellType(ConstructorDescription().genome(genome).constructionAngle1(180.0f)),
        CellDescription().id(2).pos({11.0f, 9.0f}).energy(100),
        CellDescription().id(3).pos({11.0f, 11.0f}).energy(100),
    });
    data.addConnection(1, 2);
    data.addConnection(1, 3);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);
    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(4, actualData._cells.size());
    auto actualConstructedCell = getOtherCell(actualData, {1, 2, 3});

    EXPECT_TRUE(approxCompare(11.0f, actualConstructedCell._pos.x));
    EXPECT_TRUE(approxCompare(10.0f, actualConstructedCell._pos.y));
}

TEST_F(ConstructorTests, allowLargeConstructionAngle2)
{
    auto genome =
        GenomeDescriptionConverterService::get().convertDescriptionToBytes(GenomeDescription().header(GenomeHeaderDescription()).cells({CellGenomeDescription()}));

    CollectionDescription data;
    data.addCells({
        CellDescription()
            .id(1)
            .pos({10.0f, 10.0f})
            .energy(_parameters.normalCellEnergy.value[0] * 3)
            .cellType(ConstructorDescription().genome(genome).constructionAngle1(-180.0f)),
        CellDescription().id(2).pos({11.0f, 9.0f}).energy(100),
        CellDescription().id(3).pos({11.0f, 11.0f}).energy(100),
    });
    data.addConnection(1, 2);
    data.addConnection(1, 3);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);
    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(4, actualData._cells.size());
    auto actualConstructedCell = getOtherCell(actualData, {1, 2, 3});

    EXPECT_TRUE(approxCompare(11.0f, actualConstructedCell._pos.x));
    EXPECT_TRUE(approxCompare(10.0f, actualConstructedCell._pos.y));
}

TEST_F(ConstructorTests, repetitionsAndBranches)
{
    auto genome = GenomeDescriptionConverterService::get().convertDescriptionToBytes(
        GenomeDescription()
            .header(GenomeHeaderDescription().numBranches(3).numRepetitions(4).separateConstruction(false))
            .cells({CellGenomeDescription(), CellGenomeDescription(), CellGenomeDescription()}));

    CollectionDescription data;
    data.addCells({
        CellDescription()
            .id(1)
            .pos({10.0f, 10.0f})
            .energy(_parameters.normalCellEnergy.value[0] * 2 * 3 * 4 * 3)
            .cellType(ConstructorDescription().genome(genome).autoTriggerInterval(20)),
    });

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(13 * 3 * 4 * 3 * 20);
    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(1 + 3 * 4 * 3, actualData._cells.size());
    auto actualConstructor = getCell(actualData, 1);

    EXPECT_EQ(3, actualConstructor._connections.size());
}

TEST_F(ConstructorTests, severalRepetitionsOfSingleCell)
{
    auto genome = GenomeDescriptionConverterService::get().convertDescriptionToBytes(
        GenomeDescription()
            .header(GenomeHeaderDescription().numBranches(1).numRepetitions(2).separateConstruction(false))
            .cells({CellGenomeDescription()}));

    CollectionDescription data;
    data.addCells({
        CellDescription()
            .id(1)
            .pos({10.0f, 10.0f})
            .energy(_parameters.normalCellEnergy.value[0] * 2 * 3 * 4 * 3)
            .cellType(ConstructorDescription().genome(genome)),
    });

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(200 * 6);
    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(3, actualData._cells.size());
    auto actualConstructor = getCell(actualData, 1);

    EXPECT_EQ(1, actualConstructor._connections.size());
    auto lastContructedCell = getCell(actualData, actualConstructor._connections.at(0)._cellId);
    EXPECT_EQ(2, lastContructedCell._connections.size());
}

TEST_F(ConstructorTests, severalRepetitionsAndBranchesOfSingleCell)
{
    auto genome = GenomeDescriptionConverterService::get().convertDescriptionToBytes(
        GenomeDescription()
            .header(GenomeHeaderDescription().numBranches(3).numRepetitions(2).separateConstruction(false))
            .cells({CellGenomeDescription()}));

    CollectionDescription data;
    data.addCells({
        CellDescription()
            .id(1)
            .pos({10.0f, 10.0f})
            .energy(_parameters.normalCellEnergy.value[0] * 2 * 3 * 4 * 3)
            .cellType(ConstructorDescription().genome(genome)),
    });

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(400 * 6);
    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(7, actualData._cells.size());
    auto actualConstructor = getCell(actualData, 1);

    EXPECT_EQ(3, actualConstructor._connections.size());
    for (auto const& connection : actualConstructor._connections) {
        auto lastContructedCell = getCell(actualData, connection._cellId);
        EXPECT_EQ(2, lastContructedCell._connections.size());
    }
}

TEST_F(ConstructorTests, severalRepetitionsOfSingleCell_ignoreNumRequiredConnections)
{
    auto genome = GenomeDescriptionConverterService::get().convertDescriptionToBytes(
        GenomeDescription()
            .header(GenomeHeaderDescription().numBranches(1).numRepetitions(3).separateConstruction(false))
            .cells({CellGenomeDescription().numRequiredAdditionalConnections(1)}));

    CollectionDescription data;
    data.addCells({
        CellDescription()
            .id(1)
            .pos({10.0f, 10.0f})
            .energy(_parameters.normalCellEnergy.value[0] * 2 * 3 * 4 * 3)
            .cellType(ConstructorDescription().genome(genome).numInheritedGenomeNodes(3)),
    });

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(400 * 6);
    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(4, actualData._cells.size());
    auto actualConstructor = getCell(actualData, 1);

    EXPECT_EQ(1, actualConstructor._connections.size());
    for (auto const& connection : actualConstructor._connections) {
        auto lastContructedCell = getCell(actualData, connection._cellId);
        EXPECT_EQ(2, lastContructedCell._connections.size());
    }
}
