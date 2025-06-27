#include <gtest/gtest.h>

#include "EngineInterface/DescriptionEditService.h"
#include "EngineInterface/Descriptions.h"
#include "EngineInterface/SimulationFacade.h"

#include "IntegrationTestFramework.h"

class SensorTests_New : public IntegrationTestFramework
{
public:
    SensorTests_New()
        : IntegrationTestFramework()
    {}

    ~SensorTests_New() = default;
};

TEST_F(SensorTests_New, autoTriggered)
{
    CollectionDescription data;
    data.addCells({
        CellDescription().id(1).pos({100.0f, 100.0f}).cellTypeData(SensorDescription().autoTriggerInterval(15)),
    });
    _simulationFacade->setSimulationData(data);

    {
        _simulationFacade->calcTimesteps(1);
        auto actualSensor = getCell(_simulationFacade->getSimulationData(), 1);
        EXPECT_TRUE(actualSensor._signal.has_value());
    }
    {
        _simulationFacade->calcTimesteps(1);
        auto actualSensor = getCell(_simulationFacade->getSimulationData(), 1);
        EXPECT_FALSE(actualSensor._signal.has_value());
    }
    {
        _simulationFacade->calcTimesteps(14);
        auto actualSensor = getCell(_simulationFacade->getSimulationData(), 1);
        EXPECT_TRUE(actualSensor._signal.has_value());
    }
    {
        _simulationFacade->calcTimesteps(1);
        auto actualSensor = getCell(_simulationFacade->getSimulationData(), 1);
        EXPECT_FALSE(actualSensor._signal.has_value());
    }
}

TEST_F(SensorTests_New, manuallyTriggered_noSignal)
{
    CollectionDescription data;
    data.addCells({
        CellDescription().id(1).pos({100.0f, 100.0f}).cellTypeData(SensorDescription().autoTriggerInterval(0)),
    });
    _simulationFacade->setSimulationData(data);

    for (int i = 0; i < 100; ++i) {
        _simulationFacade->calcTimesteps(1);
        auto actualSensor = getCell(_simulationFacade->getSimulationData(), 1);
        EXPECT_FALSE(actualSensor._signal.has_value());
    }
}

TEST_F(SensorTests_New, manuallyTriggered_signal)
{
    CollectionDescription data;
    data.addCells({
        CellDescription().id(1).pos({100.0f, 100.0f}).cellTypeData(SensorDescription().autoTriggerInterval(0)),
        CellDescription().id(2).pos({101.0f, 100.0f}).signalAndRelaxTime({1, 0, 0, 0, 0, 0, 0, 0}),
    });
    data.addConnection(1, 2);
    _simulationFacade->setSimulationData(data);

    _simulationFacade->calcTimesteps(1);
    auto actualSensor = getCell(_simulationFacade->getSimulationData(), 1);
    EXPECT_TRUE(actualSensor._signal.has_value());
}

TEST_F(SensorTests_New, aboveMinDensity)
{
    CollectionDescription data;
    data.addCells({
        CellDescription().id(1).pos({100.0f, 100.0f}).cellTypeData(SensorDescription().autoTriggerInterval(3).minDensity(0.2f)),
        CellDescription().id(2).pos({101.0f, 100.0f}),
    });
    data.addConnection(1, 2);
    data.add(DescriptionEditService::get().get().createRect(
        DescriptionEditService::CreateRectParameters().center({100.0f, 10.0f}).width(16).height(16).cellDistance(1.0f)));

    _simulationFacade->setSimulationData(data);

    _simulationFacade->calcTimesteps(1);
    auto actualSensor = getCell(_simulationFacade->getSimulationData(), 1);
    EXPECT_TRUE(actualSensor._signal.has_value());
    EXPECT_TRUE(approxCompare(1.0f, actualSensor._signal->_channels[Channels::SensorFoundResult]));
    EXPECT_TRUE(actualSensor._signal->_channels[Channels::SensorDensity] > 0.2f);
    EXPECT_TRUE(actualSensor._signal->_channels[Channels::SensorDistance] < 1.0f - (100.0f - 10.0f - 16.0f) / 256);
    EXPECT_TRUE(actualSensor._signal->_channels[Channels::SensorDistance] > 1.0f - (100.0f - 10.0f + 16.0f) / 256);
    EXPECT_TRUE(actualSensor._signal->_channels[Channels::SensorAngle] > (-90.0f - 15.0f) / 180);
    EXPECT_TRUE(actualSensor._signal->_channels[Channels::SensorAngle] < (-90.0f + 15.0f) / 180);
}

TEST_F(SensorTests_New, belowMinDensity)
{
    CollectionDescription data;
    data.addCells({
        CellDescription().id(1).pos({100.0f, 100.0f}).cellTypeData(SensorDescription().autoTriggerInterval(3).minDensity(0.1f)),
        CellDescription().id(2).pos({101.0f, 100.0f}),
    });
    data.addConnection(1, 2);
    data.add(DescriptionEditService::get().get().createRect(
        DescriptionEditService::CreateRectParameters().center({100.0f, 10.0f}).width(16).height(16).cellDistance(2.5f)));

    _simulationFacade->setSimulationData(data);

    _simulationFacade->calcTimesteps(1);
    auto actualSensor = getCell(_simulationFacade->getSimulationData(), 1);
    EXPECT_TRUE(actualSensor._signal.has_value());
    EXPECT_TRUE(approxCompare(1.0f, actualSensor._signal->_channels[Channels::SensorFoundResult]));
}

TEST_F(SensorTests_New, targetAbove)
{
    CollectionDescription data;
    data.addCells({
        CellDescription().id(1).pos({100.0f, 100.0f}).cellTypeData(SensorDescription().autoTriggerInterval(3).minDensity(0.2f)),
        CellDescription().id(2).pos({101.0f, 100.0f}),
    });
    data.addConnection(1, 2);
    data.add(DescriptionEditService::get().get().createRect(
        DescriptionEditService::CreateRectParameters().center({100.0f, 10.0f}).width(16).height(16).cellDistance(1.0f)));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualSensor = getCell(actualData, 1);

    EXPECT_TRUE(approxCompare(1.0f, actualSensor._signal->_channels[Channels::SensorFoundResult]));
    EXPECT_TRUE(actualSensor._signal->_channels[Channels::SensorDensity] > 0.2f);
    EXPECT_TRUE(actualSensor._signal->_channels[Channels::SensorDistance] < 1.0f - (100.0f - 10.0f - 16.0f) / 256);
    EXPECT_TRUE(actualSensor._signal->_channels[Channels::SensorDistance] > 1.0f - (100.0f - 10.0f + 16.0f) / 256);
    EXPECT_TRUE(actualSensor._signal->_channels[Channels::SensorAngle] > (-90.0f - 15.0f) / 180);
    EXPECT_TRUE(actualSensor._signal->_channels[Channels::SensorAngle] < (-90.0f + 15.0f) / 180);
}

TEST_F(SensorTests_New, targetBelow)
{
    CollectionDescription data;
    data.addCells({
        CellDescription().id(1).pos({100.0f, 100.0f}).cellTypeData(SensorDescription().autoTriggerInterval(3).minDensity(0.2f)),
        CellDescription().id(2).pos({101.0f, 100.0f}),
    });
    data.addConnection(1, 2);
    data.add(DescriptionEditService::get().get().createRect(
        DescriptionEditService::CreateRectParameters().center({100.0f, 190.0f}).width(16).height(16).cellDistance(1.0f)));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualSensor = getCell(actualData, 1);

    EXPECT_TRUE(approxCompare(1.0f, actualSensor._signal->_channels[Channels::SensorFoundResult]));
    EXPECT_TRUE(actualSensor._signal->_channels[Channels::SensorDensity] > 0.2f);
    EXPECT_TRUE(actualSensor._signal->_channels[Channels::SensorDistance] < 1.0f - (100.0f - 10.0f - 16.0f) / 256);
    EXPECT_TRUE(actualSensor._signal->_channels[Channels::SensorDistance] > 1.0f - (100.0f - 10.0f + 16.0f) / 256);
    EXPECT_TRUE(actualSensor._signal->_channels[Channels::SensorAngle] > (90.0f - 15.0f) / 180);
    EXPECT_TRUE(actualSensor._signal->_channels[Channels::SensorAngle] < (90.0f + 15.0f) / 180);
}

TEST_F(SensorTests_New, targetConcealed)
{
    CollectionDescription data;
    data.addCells({
        CellDescription().id(1).pos({100.0f, 100.0f}).cellTypeData(SensorDescription().autoTriggerInterval(3).minDensity(0.2f)),
        CellDescription().id(2).pos({101.0f, 101.0f}),
        CellDescription().id(3).pos({101.0f, 99.0f}),
    });
    data.addConnection(1, 2);
    data.addConnection(1, 3);
    data.addConnection(2, 3);
    data.add(DescriptionEditService::get().get().createRect(
        DescriptionEditService::CreateRectParameters().center({190.0f, 100.0f}).width(16).height(16).cellDistance(1.0f)));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualSensor = getCell(actualData, 1);

    EXPECT_TRUE(approxCompare(0.0f, actualSensor._signal->_channels[Channels::SensorFoundResult]));
}

TEST_F(SensorTests_New, targetNotConcealed)
{
    CollectionDescription data;
    data.addCells({
        CellDescription().id(1).pos({100.0f, 100.0f}).cellTypeData(SensorDescription().autoTriggerInterval(3).minDensity(0.2f)),
        CellDescription().id(2).pos({101.0f, 101.0f}),
        CellDescription().id(3).pos({101.0f, 99.0f}),
    });
    data.addConnection(1, 2);
    data.addConnection(1, 3);
    data.addConnection(2, 3);
    data.add(DescriptionEditService::get().get().createRect(
        DescriptionEditService::CreateRectParameters().center({100.0f, 190.0f}).width(16).height(16).cellDistance(1.0f)));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualSensor = getCell(actualData, 1);

    EXPECT_TRUE(approxCompare(1.0f, actualSensor._signal->_channels[Channels::SensorFoundResult]));
}

TEST_F(SensorTests_New, foundMassWithMatchingDensity)
{
    CollectionDescription data;
    data.addCells({
        CellDescription().id(1).pos({100.0f, 100.0f}).cellTypeData(SensorDescription().autoTriggerInterval(3).minDensity(0.7f)),
        CellDescription().id(2).pos({101.0f, 100.0f}),
    });
    data.addConnection(1, 2);

    data.add(DescriptionEditService::get().createRect(
        DescriptionEditService::CreateRectParameters().center({100.0f, 10.0f}).width(16).height(16).cellDistance(1.5f)));
    data.add(DescriptionEditService::get().createRect(
        DescriptionEditService::CreateRectParameters().center({100.0f, 200.0f}).width(16).height(16).cellDistance(1.0f)));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualSensor = getCell(actualData, 1);

    EXPECT_TRUE(approxCompare(1.0f, actualSensor._signal->_channels[Channels::SensorFoundResult]));
    EXPECT_TRUE(actualSensor._signal->_channels[Channels::SensorDensity] > 0.7f);
    EXPECT_TRUE(actualSensor._signal->_channels[Channels::SensorDistance] < 1.0f - 80.0f / 256);
    EXPECT_TRUE(actualSensor._signal->_channels[Channels::SensorDistance] > 1.0f - 105.0f / 256);
    EXPECT_TRUE(actualSensor._signal->_channels[Channels::SensorAngle] > (90.0f - 15.0f) / 180);
    EXPECT_TRUE(actualSensor._signal->_channels[Channels::SensorAngle] < (90.0f + 15.0f) / 180);
}

TEST_F(SensorTests_New, scanForOtherMutants_found)
{
    CollectionDescription data;
    data.addCreature(
        CreatureDescription().id(0).mutationId(6),
        {
            CellDescription()
                .id(1)
                .pos({100.0f, 100.0f})
                .cellTypeData(SensorDescription().restrictToMutants(SensorRestrictToMutants_RestrictToOtherMutants)),
            CellDescription().id(2).pos({101.0f, 100.0f}),
        });
    data.addConnection(1, 2);

    data.addCreature(
        CreatureDescription().mutationId(7),
        DescriptionEditService::get()
            .createRect(DescriptionEditService::CreateRectParameters()
                            .center({100.0f, 10.0f})
                            .width(16)
                            .height(16)
                            .cellDistance(0.5f)
                            .creatureId(2)
                            .cellType(BaseDescription()))
            ._cells);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualSensorCell = getCell(actualData, 1);

    EXPECT_TRUE(approxCompare(1.0f, actualSensorCell._signal->_channels[Channels::SensorFoundResult]));
    EXPECT_TRUE(actualSensorCell._signal->_channels[Channels::SensorDensity] > 0.3f);
    EXPECT_TRUE(actualSensorCell._signal->_channels[Channels::SensorDistance] < 1.0f - 80.0f / 256);
    EXPECT_TRUE(actualSensorCell._signal->_channels[Channels::SensorDistance] > 1.0f - 105.0f / 256);
    EXPECT_TRUE(actualSensorCell._signal->_channels[Channels::SensorAngle] > (-90.0f - 15.0f) / 180);
    EXPECT_TRUE(actualSensorCell._signal->_channels[Channels::SensorAngle] < (-90.0f + 15.0f) / 180);
}

TEST_F(SensorTests_New, scanForOtherMutants_found_wallBehind)
{
    CollectionDescription data;
    data.addCreature(
        CreatureDescription().id(0).mutationId(6),
        {
            CellDescription().id(1).pos({100.0f, 100.0f}).cellTypeData(SensorDescription().restrictToMutants(SensorRestrictToMutants_RestrictToOtherMutants)),
            CellDescription().id(2).pos({101.0f, 100.0f}),
        });
    data.addConnection(1, 2);

    data.add(DescriptionEditService::get().createRect(
        DescriptionEditService::CreateRectParameters().center({220.0f, 100.0f}).width(1).height(16).cellDistance(0.5f).cellType(StructureCellDescription())));

    data.addCreature(
        CreatureDescription().mutationId(7),
        DescriptionEditService::get()
            .createRect(
                DescriptionEditService::CreateRectParameters().center({200.0f, 100.0f}).width(16).height(16).cellDistance(0.5f).cellType(BaseDescription()))
            ._cells);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualSensorCell = getCell(actualData, 1);

    EXPECT_TRUE(approxCompare(1.0f, actualSensorCell._signal->_channels[Channels::SensorFoundResult]));
}

TEST_F(SensorTests_New, scanForOtherMutants_notFound_wallInBetween)
{
    CollectionDescription data;
    data.addCreature(
        CreatureDescription().id(0).mutationId(7),
        {
            CellDescription().id(1).pos({100.0f, 100.0f}).cellTypeData(SensorDescription().restrictToMutants(SensorRestrictToMutants_RestrictToOtherMutants)),
            CellDescription().id(2).pos({101.0f, 100.0f}),
        });
    data.addConnection(1, 2);

    data.add(DescriptionEditService::get().createRect(
        DescriptionEditService::CreateRectParameters().center({150.0f, 100.0f}).width(1).height(16).cellDistance(0.5f).cellType(StructureCellDescription())));

    data.addCreature(
        CreatureDescription().mutationId(7),
        DescriptionEditService::get()
            .createRect(
                DescriptionEditService::CreateRectParameters().center({200.0f, 100.0f}).width(16).height(16).cellDistance(0.5f).cellType(BaseDescription()))
            ._cells);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualSensorCell = getCell(actualData, 1);

    EXPECT_TRUE(approxCompare(0.0f, actualSensorCell._signal->_channels[Channels::SensorFoundResult]));
}

TEST_F(SensorTests_New, scanForOtherMutants_notFound_sameMutationId)
{
    CollectionDescription data;
    data.addCreature(
        CreatureDescription().id(0).mutationId(7),
        {
            CellDescription().id(1).pos({100.0f, 100.0f}).cellTypeData(SensorDescription().restrictToMutants(SensorRestrictToMutants_RestrictToOtherMutants)),
            CellDescription().id(2).pos({101.0f, 100.0f}),
        });
    data.addConnection(1, 2);

    data.addCreature(
        CreatureDescription().mutationId(7),
        DescriptionEditService::get()
            .createRect(
                DescriptionEditService::CreateRectParameters().center({10.0f, 100.0f}).width(16).height(16).cellDistance(0.5f).cellType(BaseDescription()))
            ._cells);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualSensorCell = getCell(actualData, 1);

    EXPECT_TRUE(approxCompare(0.0f, actualSensorCell._signal->_channels[Channels::SensorFoundResult]));
}

TEST_F(SensorTests_New, scanForOtherMutants_notFound_structure)
{
    CollectionDescription data;
    data.addCreature(
        CreatureDescription().id(0).mutationId(7),
        {
            CellDescription().id(1).pos({100.0f, 100.0f}).cellTypeData(SensorDescription().restrictToMutants(SensorRestrictToMutants_RestrictToOtherMutants)),
            CellDescription().id(2).pos({101.0f, 100.0f}),
        });
    data.addConnection(1, 2);

    data.addCreature(
        CreatureDescription().mutationId(6),
        DescriptionEditService::get()
            .createRect(DescriptionEditService::CreateRectParameters()
                            .center({10.0f, 100.0f})
                            .width(16)
                            .height(16)
                            .cellDistance(0.5f)
                            .cellType(StructureCellDescription()))
            ._cells);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualSensorCell = getCell(actualData, 1);

    EXPECT_TRUE(approxCompare(0.0f, actualSensorCell._signal->_channels[Channels::SensorFoundResult]));
}

TEST_F(SensorTests_New, scanForOtherMutants_notFound_freeCell)
{
    CollectionDescription data;
    data.addCreature(
        CreatureDescription().id(0).mutationId(7),
        {
            CellDescription().id(1).pos({100.0f, 100.0f}).cellTypeData(SensorDescription().restrictToMutants(SensorRestrictToMutants_RestrictToOtherMutants)),
            CellDescription().id(2).pos({101.0f, 100.0f}),
        });
    data.addConnection(1, 2);

    data.addCreature(
        CreatureDescription().mutationId(6),
        DescriptionEditService::get()
            .createRect(
                DescriptionEditService::CreateRectParameters().center({10.0f, 100.0f}).width(16).height(16).cellDistance(0.5f).cellType(FreeCellDescription()))
            ._cells);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualSensorCell = getCell(actualData, 1);

    EXPECT_TRUE(approxCompare(0.0f, actualSensorCell._signal->_channels[Channels::SensorFoundResult]));
}

TEST_F(SensorTests_New, scanForSameMutants_found)
{
    CollectionDescription data;
    data.addCreature(
        CreatureDescription().id(0).mutationId(6),
        {
            CellDescription().id(1).pos({100.0f, 100.0f}).cellTypeData(SensorDescription().restrictToMutants(SensorRestrictToMutants_RestrictToSameMutants)),
            CellDescription().id(2).pos({101.0f, 100.0f}),
        });
    data.addConnection(1, 2);

    data.addCreature(
        CreatureDescription().mutationId(6),
        DescriptionEditService::get()
            .createRect(
                DescriptionEditService::CreateRectParameters().center({10.0f, 100.0f}).width(16).height(16).cellDistance(0.5f).cellType(BaseDescription()))
            ._cells);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualSensorCell = getCell(actualData, 1);

    EXPECT_TRUE(approxCompare(1.0f, actualSensorCell._signal->_channels[Channels::SensorFoundResult]));
}

TEST_F(SensorTests_New, scanForSameMutants_notFound_otherMutationId)
{
    auto const MutantId = 6;
    for (int otherMutantId = 0; otherMutantId < 100; ++otherMutantId) {
        if (otherMutantId == MutantId) {
            continue;
        }
        CollectionDescription data;
        data.addCreature(
            CreatureDescription().id(0).mutationId(MutantId),
            {
                CellDescription().id(1).pos({100.0f, 100.0f}).cellTypeData(SensorDescription().restrictToMutants(SensorRestrictToMutants_RestrictToSameMutants)),
                CellDescription().id(2).pos({101.0f, 100.0f}),
            });
        data.addConnection(1, 2);

        data.addCreature(
            CreatureDescription().mutationId(otherMutantId),
            DescriptionEditService::get()
                .createRect(
                    DescriptionEditService::CreateRectParameters().center({10.0f, 100.0f}).width(16).height(16).cellDistance(0.5f).cellType(BaseDescription()))
                ._cells);

        _simulationFacade->clear();
        _simulationFacade->setCurrentTimestep(0ull);
        _simulationFacade->setSimulationData(data);
        _simulationFacade->calcTimesteps(1);

        auto actualData = _simulationFacade->getSimulationData();
        auto actualSensorCell = getCell(actualData, 1);

        EXPECT_TRUE(approxCompare(0.0f, actualSensorCell._signal->_channels[Channels::SensorFoundResult]));
    }
}

TEST_F(SensorTests_New, scanForSameMutants_notFound_structure)
{
    auto const MutantId = 6;
    CollectionDescription data;
    data.addCreature(
        CreatureDescription().id(0).mutationId(MutantId),
        {
            CellDescription().id(1).pos({100.0f, 100.0f}).cellTypeData(SensorDescription().restrictToMutants(SensorRestrictToMutants_RestrictToSameMutants)),
            CellDescription().id(2).pos({101.0f, 100.0f}),
        });
    data.addConnection(1, 2);

    data.addCreature(
        CreatureDescription().mutationId(MutantId),
        DescriptionEditService::get()
            .createRect(DescriptionEditService::CreateRectParameters()
                            .center({10.0f, 100.0f})
                            .width(16)
                            .height(16)
                            .cellDistance(0.5f)
                            .cellType(StructureCellDescription()))
            ._cells);

    _simulationFacade->clear();
    _simulationFacade->setCurrentTimestep(0ull);
    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualSensorCell = getCell(actualData, 1);

    EXPECT_TRUE(approxCompare(0.0f, actualSensorCell._signal->_channels[Channels::SensorFoundResult]));
}

TEST_F(SensorTests_New, scanForSameMutants_notFound_freeCell)
{
    auto const MutantId = 6;
    CollectionDescription data;
    data.addCreature(
        CreatureDescription().id(0).mutationId(MutantId),
        {
            CellDescription().id(1).pos({100.0f, 100.0f}).cellTypeData(SensorDescription().restrictToMutants(SensorRestrictToMutants_RestrictToSameMutants)),
            CellDescription().id(2).pos({101.0f, 100.0f}),
        });
    data.addConnection(1, 2);

    data.addCreature(
        CreatureDescription().mutationId(MutantId),
        DescriptionEditService::get()
            .createRect(
                DescriptionEditService::CreateRectParameters().center({10.0f, 100.0f}).width(16).height(16).cellDistance(0.5f).cellType(FreeCellDescription()))
            ._cells);

    _simulationFacade->clear();
    _simulationFacade->setCurrentTimestep(0ull);
    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualSensorCell = getCell(actualData, 1);

    EXPECT_TRUE(approxCompare(0.0f, actualSensorCell._signal->_channels[Channels::SensorFoundResult]));
}

TEST_F(SensorTests_New, scanForStructures_found)
{
    CollectionDescription data;
    data.addCells({
        CellDescription().id(1).pos({100.0f, 100.0f}).cellTypeData(SensorDescription().restrictToMutants(SensorRestrictToMutants_RestrictToStructures)),
        CellDescription().id(2).pos({101.0f, 100.0f}),
    });
    data.addConnection(1, 2);

    data.add(DescriptionEditService::get().createRect(
        DescriptionEditService::CreateRectParameters().center({10.0f, 100.0f}).width(16).height(16).cellDistance(0.5f).cellType(StructureCellDescription())));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualSensorCell = getCell(actualData, 1);

    EXPECT_TRUE(approxCompare(1.0f, actualSensorCell._signal->_channels[Channels::SensorFoundResult]));
}

TEST_F(SensorTests_New, scanForStructures_notFound)
{
    CollectionDescription data;
    data.addCells({
        CellDescription().id(1).pos({100.0f, 100.0f}).cellTypeData(SensorDescription().restrictToMutants(SensorRestrictToMutants_RestrictToStructures)),
        CellDescription().id(2).pos({101.0f, 100.0f}),
    });
    data.addConnection(1, 2);

    data.add(DescriptionEditService::get().createRect(
        DescriptionEditService::CreateRectParameters().center({10.0f, 100.0f}).width(16).height(16).cellDistance(0.5f).cellType(BaseDescription())));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualSensorCell = getCell(actualData, 1);

    EXPECT_TRUE(approxCompare(0.0f, actualSensorCell._signal->_channels[Channels::SensorFoundResult]));
}

TEST_F(SensorTests_New, scanForFreeCells_found)
{
    CollectionDescription data;
    data.addCells({
        CellDescription().id(1).pos({100.0f, 100.0f}).cellTypeData(SensorDescription().restrictToMutants(SensorRestrictToMutants_RestrictToFreeCells)),
        CellDescription().id(2).pos({101.0f, 100.0f}),
    });
    data.addConnection(1, 2);

    data.add(DescriptionEditService::get().createRect(
        DescriptionEditService::CreateRectParameters().center({10.0f, 100.0f}).width(16).height(16).cellDistance(0.5f).cellType(FreeCellDescription())));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualSensorCell = getCell(actualData, 1);

    EXPECT_TRUE(approxCompare(1.0f, actualSensorCell._signal->_channels[Channels::SensorFoundResult]));
}

TEST_F(SensorTests_New, scanForFreeCells_notFound)
{
    CollectionDescription data;
    data.addCells({
        CellDescription().id(1).pos({100.0f, 100.0f}).cellTypeData(SensorDescription().restrictToMutants(SensorRestrictToMutants_RestrictToFreeCells)),
        CellDescription().id(2).pos({101.0f, 100.0f}),
    });
    data.addConnection(1, 2);

    data.add(DescriptionEditService::get().createRect(
        DescriptionEditService::CreateRectParameters().center({10.0f, 100.0f}).width(16).height(16).cellDistance(0.5f).cellType(StructureCellDescription())));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualSensorCell = getCell(actualData, 1);

    EXPECT_TRUE(approxCompare(0.0f, actualSensorCell._signal->_channels[Channels::SensorFoundResult]));
}

TEST_F(SensorTests_New, scanForLessComplexMutants_found)
{
    for (int otherGenomeComplexity = 0; otherGenomeComplexity < 500; ++otherGenomeComplexity) {
        CollectionDescription data;
        data.addCreature(
            CreatureDescription().id(0).genomeComplexity(1000.0f),
            {
                CellDescription()
                    .id(1)
                    .pos({100.0f, 100.0f})
                    .cellTypeData(SensorDescription().restrictToMutants(SensorRestrictToMutants_RestrictToLessComplexMutants)),
                CellDescription().id(2).pos({101.0f, 100.0f}),
            });
        data.addConnection(1, 2);

        data.addCreature(
            CreatureDescription().genomeComplexity(toFloat(otherGenomeComplexity)),
            DescriptionEditService::get()
                .createRect(
                    DescriptionEditService::CreateRectParameters().center({10.0f, 100.0f}).cellType(BaseDescription()).width(16).height(16).cellDistance(0.5f))
                ._cells);

        _simulationFacade->clear();
        _simulationFacade->setCurrentTimestep(0ull);
        _simulationFacade->setSimulationData(data);
        _simulationFacade->calcTimesteps(1);

        auto actualData = _simulationFacade->getSimulationData();
        auto actualSensorCell = getCell(actualData, 1);

        EXPECT_TRUE(approxCompare(1.0f, actualSensorCell._signal->_channels[Channels::SensorFoundResult]));
    }
}

TEST_F(SensorTests_New, scanForLessComplexMutants_notFound_otherMoreComplex)
{
    for (int otherGenomeComplexity = 1000; otherGenomeComplexity < 2001; ++otherGenomeComplexity) {
        CollectionDescription data;
        data.addCreature(
            CreatureDescription().id(0).genomeComplexity(1000.0f),
            {
                CellDescription()
                    .id(1)
                    .pos({100.0f, 100.0f})
                    .cellTypeData(SensorDescription().restrictToMutants(SensorRestrictToMutants_RestrictToLessComplexMutants)),
                CellDescription().id(2).pos({101.0f, 100.0f}),
            });
        data.addConnection(1, 2);

        data.addCreature(
            CreatureDescription().genomeComplexity(toFloat(otherGenomeComplexity)),
            DescriptionEditService::get()
                .createRect(
                    DescriptionEditService::CreateRectParameters().center({10.0f, 100.0f}).width(16).height(16).cellDistance(0.5f).cellType(BaseDescription()))
                ._cells);

        _simulationFacade->clear();
        _simulationFacade->setCurrentTimestep(0ull);
        _simulationFacade->setSimulationData(data);
        _simulationFacade->calcTimesteps(1);

        auto actualData = _simulationFacade->getSimulationData();
        auto actualSensorCell = getCell(actualData, 1);

        EXPECT_TRUE(approxCompare(0.0f, actualSensorCell._signal->_channels[Channels::SensorFoundResult]));
    }
}

TEST_F(SensorTests_New, scanForLessComplexMutants_notFound_structure)
{
    CollectionDescription data;
    data.addCreature(
        CreatureDescription().id(0).genomeComplexity(1000.0f),
        {CellDescription().id(1).pos({100.0f, 100.0f}).cellTypeData(SensorDescription().restrictToMutants(SensorRestrictToMutants_RestrictToLessComplexMutants)),
         CellDescription().id(2).pos({101.0f, 100.0f}).cellTypeData(OscillatorDescription()).signalAndRelaxTime({1, 0, 0, 0, 0, 0, 0, 0})});
    data.addConnection(1, 2);

    data.add(DescriptionEditService::get().createRect(
        DescriptionEditService::CreateRectParameters().center({10.0f, 100.0f}).width(16).height(16).cellDistance(0.5f).cellType(StructureCellDescription())));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualSensorCell = getCell(actualData, 1);

    EXPECT_TRUE(approxCompare(0.0f, actualSensorCell._signal->_channels[Channels::SensorFoundResult]));
}

TEST_F(SensorTests_New, scanForLessComplexMutants_notFound_freeCell)
{
    CollectionDescription data;
    data.addCreature(
        CreatureDescription().id(0).genomeComplexity(1000.0f),
        {
            CellDescription()
                .id(1)
                .pos({100.0f, 100.0f})
                .cellTypeData(SensorDescription().restrictToMutants(SensorRestrictToMutants_RestrictToLessComplexMutants)),
            CellDescription().id(2).pos({101.0f, 100.0f}),
        });
    data.addConnection(1, 2);

    data.add(DescriptionEditService::get().createRect(
        DescriptionEditService::CreateRectParameters().center({10.0f, 100.0f}).width(16).height(16).cellDistance(0.5f).cellType(FreeCellDescription())));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualSensorCell = getCell(actualData, 1);

    EXPECT_TRUE(approxCompare(0.0f, actualSensorCell._signal->_channels[Channels::SensorFoundResult]));
}

TEST_F(SensorTests_New, scanForMoreComplexMutants_found)
{
    for (int otherGenomeComplexity = 1000; otherGenomeComplexity < 2001; ++otherGenomeComplexity) {
        CollectionDescription data;
        data.addCreature(
            CreatureDescription().id(0).genomeComplexity(500.0f),
            {
                CellDescription()
                    .id(1)
                    .pos({100.0f, 100.0f})
                    .cellTypeData(SensorDescription().restrictToMutants(SensorRestrictToMutants_RestrictToMoreComplexMutants)),
                CellDescription().id(2).pos({101.0f, 100.0f}),
            });
        data.addConnection(1, 2);

        data.addCreature(
            CreatureDescription().genomeComplexity(toFloat(otherGenomeComplexity)),
            DescriptionEditService::get()
                .createRect(
                    DescriptionEditService::CreateRectParameters().center({10.0f, 100.0f}).width(16).height(16).cellDistance(0.5f).cellType(BaseDescription()))
                ._cells);

        _simulationFacade->clear();
        _simulationFacade->setCurrentTimestep(0ull);
        _simulationFacade->setSimulationData(data);
        _simulationFacade->calcTimesteps(1);

        auto actualData = _simulationFacade->getSimulationData();
        auto actualSensorCell = getCell(actualData, 1);

        EXPECT_TRUE(approxCompare(1.0f, actualSensorCell._signal->_channels[Channels::SensorFoundResult]));
    }
}

TEST_F(SensorTests_New, scanForMoreComplexMutants_notFound_otherLessComplex)
{
    for (int otherGenomeComplexity = 0; otherGenomeComplexity < 500; ++otherGenomeComplexity) {
        CollectionDescription data;
        data.addCreature(
            CreatureDescription().id(0).genomeComplexity(500.0f),
            {
                CellDescription()
                    .id(1)
                    .pos({100.0f, 100.0f})
                    .cellTypeData(SensorDescription().restrictToMutants(SensorRestrictToMutants_RestrictToMoreComplexMutants)),
                CellDescription().id(2).pos({101.0f, 100.0f}),
            });
        data.addConnection(1, 2);

        data.addCreature(
            CreatureDescription().genomeComplexity(toFloat(otherGenomeComplexity)),
            DescriptionEditService::get()
                .createRect(
                    DescriptionEditService::CreateRectParameters().center({10.0f, 100.0f}).width(16).height(16).cellDistance(0.5f).cellType(BaseDescription()))
                ._cells);

        _simulationFacade->clear();
        _simulationFacade->setCurrentTimestep(0ull);
        _simulationFacade->setSimulationData(data);
        _simulationFacade->calcTimesteps(1);

        auto actualData = _simulationFacade->getSimulationData();
        auto actualSensorCell = getCell(actualData, 1);

        EXPECT_TRUE(approxCompare(0.0f, actualSensorCell._signal->_channels[Channels::SensorFoundResult]));
    }
}

TEST_F(SensorTests_New, scanForMoreComplexMutants_notFound_structure)
{
    CollectionDescription data;
    data.addCreature(
        CreatureDescription().id(0).genomeComplexity(100.0f),
        {
            CellDescription()
                .id(1)
                .pos({100.0f, 100.0f})
                .cellTypeData(SensorDescription().restrictToMutants(SensorRestrictToMutants_RestrictToMoreComplexMutants)),
            CellDescription().id(2).pos({101.0f, 100.0f}),
        });
    data.addConnection(1, 2);

    data.add(DescriptionEditService::get().createRect(
        DescriptionEditService::CreateRectParameters().center({10.0f, 100.0f}).width(16).height(16).cellDistance(0.5f).cellType(StructureCellDescription())));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualSensorCell = getCell(actualData, 1);

    EXPECT_TRUE(approxCompare(0.0f, actualSensorCell._signal->_channels[Channels::SensorFoundResult]));
}

TEST_F(SensorTests_New, scanForMoreComplexMutants_notFound_freeCell)
{
    _parameters.attackerSensorDetectionFactor.value[0] = 1.0f;
    _simulationFacade->setSimulationParameters(_parameters);
    CollectionDescription data;
    data.addCreature(
        CreatureDescription().id(0).genomeComplexity(100.0f),
        {
            CellDescription()
                .id(1)
                .pos({100.0f, 100.0f})
                .cellTypeData(SensorDescription().restrictToMutants(SensorRestrictToMutants_RestrictToMoreComplexMutants)),
            CellDescription().id(2).pos({101.0f, 100.0f}),
        });
    data.addConnection(1, 2);

    data.add(DescriptionEditService::get().createRect(
        DescriptionEditService::CreateRectParameters().center({10.0f, 100.0f}).width(16).height(16).cellDistance(0.5f).cellType(FreeCellDescription())));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualSensorCell = getCell(actualData, 1);

    EXPECT_TRUE(approxCompare(0.0f, actualSensorCell._signal->_channels[Channels::SensorFoundResult]));
}

TEST_F(SensorTests_New, minRange_found)
{
    CollectionDescription data;
    data.addCells({
        CellDescription().id(1).pos({100.0f, 100.0f}).cellTypeData(SensorDescription().minRange(50)),
        CellDescription().id(2).pos({101.0f, 100.0f}),
    });
    data.addConnection(1, 2);

    data.add(DescriptionEditService::get().createRect(
        DescriptionEditService::CreateRectParameters().center({10.0f, 100.0f}).width(16).height(16).cellDistance(0.5f).cellType(BaseDescription())));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualSensorCell = getCell(actualData, 1);

    EXPECT_TRUE(approxCompare(1.0f, actualSensorCell._signal->_channels[Channels::SensorFoundResult]));
}

TEST_F(SensorTests_New, minRange_notFound)
{
    CollectionDescription data;
    data.addCells({
        CellDescription().id(1).pos({100.0f, 100.0f}).cellTypeData(SensorDescription().minRange(120)),
        CellDescription().id(2).pos({101.0f, 100.0f}),
    });
    data.addConnection(1, 2);

    data.add(DescriptionEditService::get().createRect(
        DescriptionEditService::CreateRectParameters().center({10.0f, 100.0f}).width(16).height(16).cellDistance(0.5f).cellType(BaseDescription())));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualSensorCell = getCell(actualData, 1);

    EXPECT_TRUE(approxCompare(0.0f, actualSensorCell._signal->_channels[Channels::SensorFoundResult]));
}

TEST_F(SensorTests_New, maxRange_found)
{
    CollectionDescription data;
    data.addCells({
        CellDescription().id(1).pos({100.0f, 100.0f}).cellTypeData(SensorDescription().maxRange(120)),
        CellDescription().id(2).pos({101.0f, 100.0f}),
    });
    data.addConnection(1, 2);

    data.add(DescriptionEditService::get().createRect(
        DescriptionEditService::CreateRectParameters().center({10.0f, 100.0f}).width(16).height(16).cellDistance(0.5f).cellType(BaseDescription())));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualSensorCell = getCell(actualData, 1);

    EXPECT_TRUE(approxCompare(1.0f, actualSensorCell._signal->_channels[Channels::SensorFoundResult]));
}

TEST_F(SensorTests_New, maxRange_notFound)
{
    CollectionDescription data;
    data.addCells({
        CellDescription().id(1).pos({100.0f, 100.0f}).cellTypeData(SensorDescription().maxRange(50)),
        CellDescription().id(2).pos({101.0f, 100.0f}),
    });
    data.addConnection(1, 2);

    data.add(DescriptionEditService::get().createRect(
        DescriptionEditService::CreateRectParameters().center({10.0f, 100.0f}).width(16).height(16).cellDistance(0.5f).cellType(BaseDescription())));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualSensorCell = getCell(actualData, 1);

    EXPECT_TRUE(approxCompare(0.0f, actualSensorCell._signal->_channels[Channels::SensorFoundResult]));
}
