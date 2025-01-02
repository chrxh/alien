#include <gtest/gtest.h>

#include "EngineInterface/DescriptionEditService.h"
#include "EngineInterface/Descriptions.h"
#include "EngineInterface/SimulationFacade.h"
#include "EngineInterface/GenomeDescriptionService.h"

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
                result.cellFunctionInjectorDurationColorMatrix[i][j] = 3;
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
             .setId(1)
             .setPos({10.0f, 10.0f})
             .setCellFunction(InjectorDescription().setMode(InjectorMode_InjectAll)),
         CellDescription()
             .setId(2)
             .setPos({11.0f, 10.0f})
             .setCellFunction(OscillatorDescription().setPulseMode(1))
             .setSignal({1, 0, 0, 0, 0, 0, 0, 0})});
    data.addConnection(1, 2);

    _simulationFacade->setSimulationData(data);
    for (int i = 0; i < 6 * 4; ++i) {
        _simulationFacade->calcTimesteps(1);
    }

    auto actualData = _simulationFacade->getSimulationData();
    auto actualCell = getCell(actualData, 1);
    auto actualInjector = std::get<InjectorDescription>(*actualCell.cellFunction);

    EXPECT_EQ(2, actualData.cells.size());
    EXPECT_TRUE(approxCompare(0.0f, actualCell.signal->channels[0]));
    EXPECT_EQ(0, actualInjector.counter);
}

TEST_F(InjectorTests, matchButNoInjection)
{
    auto genome = GenomeDescriptionService::get().convertDescriptionToBytes(GenomeDescription().setCells({CellGenomeDescription()}));

    DataDescription data;
    data.addCells(
        {CellDescription()
             .setId(1)
             .setPos({10.0f, 10.0f})
             .setCellFunction(InjectorDescription().setMode(InjectorMode_InjectAll).setGenome(genome)),
         CellDescription()
             .setId(2)
             .setPos({11.0f, 10.0f})
             .setCellFunction(OscillatorDescription().setPulseMode(1))
             .setSignal({1, 0, 0, 0, 0, 0, 0, 0}),
        CellDescription()
            .setId(3)
            .setPos({9.0f, 10.0f})
            .setCellFunction(ConstructorDescription().setNumInheritedGenomeNodes(1)),
    });
    data.addConnection(1, 2);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualCell = getCell(actualData, 1);
    auto actualInjector = std::get<InjectorDescription>(*actualCell.cellFunction);
    auto actualTargetCell = getCell(actualData, 3);
    auto actualTargetConstructor = std::get<ConstructorDescription>(*actualTargetCell.cellFunction);
    auto origTargetCell = getCell(data, 3);
    auto origTargetConstructor = std::get<ConstructorDescription>(*origTargetCell.cellFunction);

    EXPECT_EQ(3, actualData.cells.size());
    EXPECT_TRUE(approxCompare(1.0f, actualCell.signal->channels[0]));
    EXPECT_EQ(1, actualInjector.counter);
    EXPECT_EQ(origTargetConstructor.genome, actualTargetConstructor.genome);
    EXPECT_TRUE(actualTargetConstructor.isGenomeInherited());
}

TEST_F(InjectorTests, injection)
{
    auto genome = GenomeDescriptionService::get().convertDescriptionToBytes(GenomeDescription().setCells({CellGenomeDescription()}));

    DataDescription data;
    data.addCells({
        CellDescription()
            .setId(1)
            .setPos({10.0f, 10.0f})
            .setCellFunction(InjectorDescription().setMode(InjectorMode_InjectAll).setGenome(genome)),
        CellDescription()
            .setId(2)
            .setPos({11.0f, 10.0f})
            .setCellFunction(OscillatorDescription().setPulseMode(1))
            .setSignal({1, 0, 0, 0, 0, 0, 0, 0}),
        CellDescription().setId(3).setPos({9.0f, 10.0f}).setCellFunction(ConstructorDescription().setNumInheritedGenomeNodes(1)),
    });
    data.addConnection(1, 2);

    _simulationFacade->setSimulationData(data);
    for (int i = 0; i < 1 + 6*3; ++i) {
        _simulationFacade->calcTimesteps(1);
    }

    auto actualData = _simulationFacade->getSimulationData();
    auto actualCell = getCell(actualData, 1);
    auto actualInjector = std::get<InjectorDescription>(*actualCell.cellFunction);
    auto actualTargetCell = getCell(actualData, 3);
    auto actualTargetConstructor = std::get<ConstructorDescription>(*actualTargetCell.cellFunction);
    auto origCell = getCell(data, 1);
    auto origInjector = std::get<InjectorDescription>(*origCell.cellFunction);

    EXPECT_EQ(3, actualData.cells.size());
    EXPECT_TRUE(approxCompare(1.0f, actualCell.signal->channels[0]));
    EXPECT_EQ(0, actualInjector.counter);
    EXPECT_EQ(origInjector.genome, actualTargetConstructor.genome);
    EXPECT_FALSE(actualTargetConstructor.isGenomeInherited());
}

TEST_F(InjectorTests, injectOnlyEmptyCells_failed)
{
    auto genome = GenomeDescriptionService::get().convertDescriptionToBytes(GenomeDescription().setCells({CellGenomeDescription()}));
    auto otherGenome = GenomeDescriptionService::get().convertDescriptionToBytes(GenomeDescription().setCells({CellGenomeDescription(), CellGenomeDescription()}));

    DataDescription data;
    data.addCells({
        CellDescription()
            .setId(1)
            .setPos({10.0f, 10.0f})
            .setCellFunction(InjectorDescription().setMode(InjectorMode_InjectOnlyEmptyCells).setGenome(genome)),
        CellDescription()
            .setId(2)
            .setPos({11.0f, 10.0f})
            .setCellFunction(OscillatorDescription().setPulseMode(1))
            .setSignal({1, 0, 0, 0, 0, 0, 0, 0}),
        CellDescription()
            .setId(3)
            .setPos({9.0f, 10.0f})
            .setCellFunction(ConstructorDescription().setGenome(otherGenome).setNumInheritedGenomeNodes(2)),
    });
    data.addConnection(1, 2);

    _simulationFacade->setSimulationData(data);
    for (int i = 0; i < 1 + 6 * 3; ++i) {
        _simulationFacade->calcTimesteps(1);
    }

    auto actualData = _simulationFacade->getSimulationData();
    auto actualCell = getCell(actualData, 1);
    auto actualInjector = std::get<InjectorDescription>(*actualCell.cellFunction);
    auto actualTargetCell = getCell(actualData, 3);
    auto actualTargetConstructor = std::get<ConstructorDescription>(*actualTargetCell.cellFunction);
    auto origTargetCell = getCell(data, 3);
    auto origTargetConstructor = std::get<ConstructorDescription>(*origTargetCell.cellFunction);

    EXPECT_EQ(3, actualData.cells.size());
    EXPECT_TRUE(approxCompare(0.0f, actualCell.signal->channels[0]));
    EXPECT_EQ(0, actualInjector.counter);
    EXPECT_EQ(origTargetConstructor.genome, actualTargetConstructor.genome);
    EXPECT_TRUE(actualTargetConstructor.isGenomeInherited());
}

TEST_F(InjectorTests, injectOnlyEmptyCells_success)
{
    auto genome = GenomeDescriptionService::get().convertDescriptionToBytes(GenomeDescription().setCells({CellGenomeDescription()}));
    auto otherGenome = GenomeDescriptionService::get().convertDescriptionToBytes(GenomeDescription().setCells({CellGenomeDescription(), CellGenomeDescription()}));

    DataDescription data;
    data.addCells({
        CellDescription()
            .setId(1)
            .setPos({10.0f, 10.0f})
            .setCellFunction(InjectorDescription().setMode(InjectorMode_InjectOnlyEmptyCells).setGenome(genome)),
        CellDescription()
            .setId(2)
            .setPos({11.0f, 10.0f})
            .setCellFunction(OscillatorDescription().setPulseMode(1))
            .setSignal({1, 0, 0, 0, 0, 0, 0, 0}),
        CellDescription()
            .setId(3)
            .setPos({9.0f, 10.0f})
            .setCellFunction(ConstructorDescription().setGenome(otherGenome)),
        CellDescription()
            .setId(4)
            .setPos({7.0f, 10.0f})
            .setCellFunction(ConstructorDescription().setNumInheritedGenomeNodes(2)),
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
    auto actualInjector = std::get<InjectorDescription>(*actualCell.cellFunction);

    auto actualTargetConstructor = std::get<ConstructorDescription>(*getCell(actualData, 4).cellFunction);

    auto origOtherConstructor = std::get<ConstructorDescription>(*getCell(data, 3).cellFunction);
    auto actualOtherConstructor = std::get<ConstructorDescription>(*getCell(actualData, 3).cellFunction);


    EXPECT_EQ(4, actualData.cells.size());
    EXPECT_TRUE(approxCompare(1.0f, actualCell.signal->channels[0]));
    EXPECT_EQ(0, actualInjector.counter);
    EXPECT_EQ(actualInjector.genome, actualTargetConstructor.genome);
    EXPECT_EQ(origOtherConstructor.genome, actualOtherConstructor.genome);
    EXPECT_FALSE(actualTargetConstructor.isGenomeInherited());
}
