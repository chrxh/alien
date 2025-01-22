#include <gtest/gtest.h>

#include "EngineInterface/DescriptionEditService.h"
#include "EngineInterface/Descriptions.h"
#include "EngineInterface/SimulationFacade.h"
#include "EngineInterface/GenomeDescriptionConverterService.h"

#include "IntegrationTestFramework.h"

class InjectorTests : public IntegrationTestFramework
{
public:
    static SimulationParameters getParameters()
    {
        SimulationParameters result;
        result.innerFriction = 0;
        result.baseValues.friction = 1;
        for (int i = 0; i < MAX_COLORS; ++i) {
            result.baseValues.radiationCellAgeStrength[i] = 0;
            result.highRadiationFactor[i] = 0;
            for (int j = 0; j < MAX_COLORS; ++j) {
                result.cellTypeInjectorDurationColorMatrix[i][j] = 3;
            }
        }
        return result;
    }
    InjectorTests()
        : IntegrationTestFramework(getParameters())
    {}

    ~InjectorTests() = default;
};

TEST_F(InjectorTests, nothingFound)
{
    DataDescription data;
    data.addCells(
        {CellDescription()
             .id(1)
             .pos({10.0f, 10.0f})
             .cellType(InjectorDescription().mode(InjectorMode_InjectAll)),
         CellDescription()
             .id(2)
             .pos({11.0f, 10.0f})
             .cellType(OscillatorDescription().autoTriggerInterval(1))
             .signal({1, 0, 0, 0, 0, 0, 0, 0})});
    data.addConnection(1, 2);

    _simulationFacade->setSimulationData(data);
    for (int i = 0; i < 6 * 4; ++i) {
        _simulationFacade->calcTimesteps(1);
    }

    auto actualData = _simulationFacade->getSimulationData();
    auto actualCell = getCell(actualData, 1);
    auto actualInjector = std::get<InjectorDescription>(actualCell._cellTypeData);

    EXPECT_EQ(2, actualData._cells.size());
    EXPECT_TRUE(approxCompare(0.0f, actualCell._signal->_channels[0]));
    EXPECT_EQ(0, actualInjector._counter);
}

TEST_F(InjectorTests, matchButNoInjection)
{
    auto genome = GenomeDescriptionConverterService::get().convertDescriptionToBytes(GenomeDescription().cells({CellGenomeDescription()}));

    DataDescription data;
    data.addCells(
        {CellDescription()
             .id(1)
             .pos({10.0f, 10.0f})
             .cellType(InjectorDescription().mode(InjectorMode_InjectAll).genome(genome)),
         CellDescription()
             .id(2)
             .pos({11.0f, 10.0f})
             .cellType(OscillatorDescription().autoTriggerInterval(1))
             .signal({1, 0, 0, 0, 0, 0, 0, 0}),
        CellDescription()
            .id(3)
            .pos({9.0f, 10.0f})
            .cellType(ConstructorDescription().numInheritedGenomeNodes(1)),
    });
    data.addConnection(1, 2);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualCell = getCell(actualData, 1);
    auto actualInjector = std::get<InjectorDescription>(actualCell._cellTypeData);
    auto actualTargetCell = getCell(actualData, 3);
    auto actualTargetConstructor = std::get<ConstructorDescription>(actualTargetCell._cellTypeData);
    auto origTargetCell = getCell(data, 3);
    auto origTargetConstructor = std::get<ConstructorDescription>(origTargetCell._cellTypeData);

    EXPECT_EQ(3, actualData._cells.size());
    EXPECT_TRUE(approxCompare(1.0f, actualCell._signal->_channels[0]));
    EXPECT_EQ(1, actualInjector._counter);
    EXPECT_EQ(origTargetConstructor._genome, actualTargetConstructor._genome);
    EXPECT_TRUE(actualTargetConstructor.isGenomeInherited());
}

TEST_F(InjectorTests, injection)
{
    auto genome = GenomeDescriptionConverterService::get().convertDescriptionToBytes(GenomeDescription().cells({CellGenomeDescription()}));

    DataDescription data;
    data.addCells({
        CellDescription()
            .id(1)
            .pos({10.0f, 10.0f})
            .cellType(InjectorDescription().mode(InjectorMode_InjectAll).genome(genome)),
        CellDescription()
            .id(2)
            .pos({11.0f, 10.0f})
            .cellType(OscillatorDescription().autoTriggerInterval(1))
            .signal({1, 0, 0, 0, 0, 0, 0, 0}),
        CellDescription().id(3).pos({9.0f, 10.0f}).cellType(ConstructorDescription().numInheritedGenomeNodes(1)),
    });
    data.addConnection(1, 2);

    _simulationFacade->setSimulationData(data);
    for (int i = 0; i < 1 + 6*3; ++i) {
        _simulationFacade->calcTimesteps(1);
    }

    auto actualData = _simulationFacade->getSimulationData();
    auto actualCell = getCell(actualData, 1);
    auto actualInjector = std::get<InjectorDescription>(actualCell._cellTypeData);
    auto actualTargetCell = getCell(actualData, 3);
    auto actualTargetConstructor = std::get<ConstructorDescription>(actualTargetCell._cellTypeData);
    auto origCell = getCell(data, 1);
    auto origInjector = std::get<InjectorDescription>(origCell._cellTypeData);

    EXPECT_EQ(3, actualData._cells.size());
    EXPECT_TRUE(approxCompare(1.0f, actualCell._signal->_channels[0]));
    EXPECT_EQ(0, actualInjector._counter);
    EXPECT_EQ(origInjector._genome, actualTargetConstructor._genome);
    EXPECT_FALSE(actualTargetConstructor.isGenomeInherited());
}

TEST_F(InjectorTests, injectOnlyEmptyCells_failed)
{
    auto genome = GenomeDescriptionConverterService::get().convertDescriptionToBytes(GenomeDescription().cells({CellGenomeDescription()}));
    auto otherGenome = GenomeDescriptionConverterService::get().convertDescriptionToBytes(GenomeDescription().cells({CellGenomeDescription(), CellGenomeDescription()}));

    DataDescription data;
    data.addCells({
        CellDescription()
            .id(1)
            .pos({10.0f, 10.0f})
            .cellType(InjectorDescription().mode(InjectorMode_InjectOnlyEmptyCells).genome(genome)),
        CellDescription()
            .id(2)
            .pos({11.0f, 10.0f})
            .cellType(OscillatorDescription().autoTriggerInterval(1))
            .signal({1, 0, 0, 0, 0, 0, 0, 0}),
        CellDescription()
            .id(3)
            .pos({9.0f, 10.0f})
            .cellType(ConstructorDescription().genome(otherGenome).numInheritedGenomeNodes(2)),
    });
    data.addConnection(1, 2);

    _simulationFacade->setSimulationData(data);
    for (int i = 0; i < 1 + 6 * 3; ++i) {
        _simulationFacade->calcTimesteps(1);
    }

    auto actualData = _simulationFacade->getSimulationData();
    auto actualCell = getCell(actualData, 1);
    auto actualInjector = std::get<InjectorDescription>(actualCell._cellTypeData);
    auto actualTargetCell = getCell(actualData, 3);
    auto actualTargetConstructor = std::get<ConstructorDescription>(actualTargetCell._cellTypeData);
    auto origTargetCell = getCell(data, 3);
    auto origTargetConstructor = std::get<ConstructorDescription>(origTargetCell._cellTypeData);

    EXPECT_EQ(3, actualData._cells.size());
    EXPECT_TRUE(approxCompare(0.0f, actualCell._signal->_channels[0]));
    EXPECT_EQ(0, actualInjector._counter);
    EXPECT_EQ(origTargetConstructor._genome, actualTargetConstructor._genome);
    EXPECT_TRUE(actualTargetConstructor.isGenomeInherited());
}

TEST_F(InjectorTests, injectOnlyEmptyCells_success)
{
    auto genome = GenomeDescriptionConverterService::get().convertDescriptionToBytes(GenomeDescription().cells({CellGenomeDescription()}));
    auto otherGenome = GenomeDescriptionConverterService::get().convertDescriptionToBytes(GenomeDescription().cells({CellGenomeDescription(), CellGenomeDescription()}));

    DataDescription data;
    data.addCells({
        CellDescription()
            .id(1)
            .pos({10.0f, 10.0f})
            .cellType(InjectorDescription().mode(InjectorMode_InjectOnlyEmptyCells).genome(genome)),
        CellDescription()
            .id(2)
            .pos({11.0f, 10.0f})
            .cellType(OscillatorDescription().autoTriggerInterval(1))
            .signal({1, 0, 0, 0, 0, 0, 0, 0}),
        CellDescription()
            .id(3)
            .pos({9.0f, 10.0f})
            .cellType(ConstructorDescription().genome(otherGenome)),
        CellDescription()
            .id(4)
            .pos({7.0f, 10.0f})
            .cellType(ConstructorDescription().numInheritedGenomeNodes(2)),
    });
    data.addConnection(1, 2);
    data.addConnection(1, 3);
    data.addConnection(3, 4);

    _simulationFacade->setSimulationData(data);
    for (int i = 0; i < 1; ++i) {
        _simulationFacade->calcTimesteps(1);
    }

    auto actualData = _simulationFacade->getSimulationData();
    auto actualCell = getCell(actualData, 1);
    auto actualInjector = std::get<InjectorDescription>(actualCell._cellTypeData);

    auto actualTargetConstructor = std::get<ConstructorDescription>(getCell(actualData, 4)._cellTypeData);

    auto origOtherConstructor = std::get<ConstructorDescription>(getCell(data, 3)._cellTypeData);
    auto actualOtherConstructor = std::get<ConstructorDescription>(getCell(actualData, 3)._cellTypeData);


    EXPECT_EQ(4, actualData._cells.size());
    EXPECT_TRUE(approxCompare(1.0f, actualCell._signal->_channels[0]));
    EXPECT_EQ(0, actualInjector._counter);
    EXPECT_EQ(actualInjector._genome, actualTargetConstructor._genome);
    EXPECT_EQ(origOtherConstructor._genome, actualOtherConstructor._genome);
    EXPECT_FALSE(actualTargetConstructor.isGenomeInherited());
}
