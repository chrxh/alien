#include <gtest/gtest.h>

#include "EngineInterface/DescriptionEditService.h"
#include "EngineInterface/Descriptions.h"
#include "EngineInterface/SimulationFacade.h"

#include "IntegrationTestFramework.h"

class SensorTests : public IntegrationTestFramework
{
public:
    SensorTests()
        : IntegrationTestFramework()
    {}

    ~SensorTests() = default;
};

TEST_F(SensorTests, autoTriggered)
{
    CollectionDescription data;
    data._cells = {
        CellDescription().id(1).pos({100.0f, 100.0f}).cellTypeData(SensorDescription().autoTriggerInterval(15)),
    };
    _simulationFacade->setSimulationData(data);

    {
        _simulationFacade->calcTimesteps(1);
        auto actualSensor = _simulationFacade->getSimulationData().getCellRef(1);
        EXPECT_TRUE(actualSensor._signal.has_value());
    }
    {
        _simulationFacade->calcTimesteps(1);
        auto actualSensor = _simulationFacade->getSimulationData().getCellRef(1);
        EXPECT_FALSE(actualSensor._signal.has_value());
    }
    {
        _simulationFacade->calcTimesteps(14);
        auto actualSensor = _simulationFacade->getSimulationData().getCellRef(1);
        EXPECT_TRUE(actualSensor._signal.has_value());
    }
    {
        _simulationFacade->calcTimesteps(1);
        auto actualSensor = _simulationFacade->getSimulationData().getCellRef(1);
        EXPECT_FALSE(actualSensor._signal.has_value());
    }
}

TEST_F(SensorTests, manuallyTriggered_noSignal)
{
    CollectionDescription data;
    data._cells = {
        CellDescription().id(1).pos({100.0f, 100.0f}).cellTypeData(SensorDescription().autoTriggerInterval(0)),
    };
    _simulationFacade->setSimulationData(data);

    for (int i = 0; i < 100; ++i) {
        _simulationFacade->calcTimesteps(1);
        auto actualSensor = _simulationFacade->getSimulationData().getCellRef(1);
        EXPECT_FALSE(actualSensor._signal.has_value());
    }
}

TEST_F(SensorTests, manuallyTriggered_signal)
{
    CollectionDescription data;
    data._cells = {
        CellDescription().id(1).pos({100.0f, 100.0f}).cellTypeData(SensorDescription().autoTriggerInterval(0)),
        CellDescription().id(2).pos({101.0f, 100.0f}).signalAndRelaxTime({1, 0, 0, 0, 0, 0, 0, 0}),
    };
    data.addConnection(1, 2);
    _simulationFacade->setSimulationData(data);

    _simulationFacade->calcTimesteps(1);
    auto actualSensor = _simulationFacade->getSimulationData().getCellRef(1);
    EXPECT_TRUE(actualSensor._signal.has_value());
}

TEST_F(SensorTests, aboveMinDensity)
{
    CollectionDescription data;
    data._cells = {
        CellDescription().id(1).pos({100.0f, 100.0f}).cellTypeData(SensorDescription().autoTriggerInterval(3).minDensity(0.2f)),
        CellDescription().id(2).pos({101.0f, 100.0f}),
    };
    data.addConnection(1, 2);
    data.add(DescriptionEditService::get().get().createRect(
        DescriptionEditService::CreateRectParameters().center({100.0f, 10.0f}).width(16).height(16).cellDistance(1.0f)));

    _simulationFacade->setSimulationData(data);

    _simulationFacade->calcTimesteps(1);
    auto actualSensor = _simulationFacade->getSimulationData().getCellRef(1);
    EXPECT_TRUE(actualSensor._signal.has_value());
    EXPECT_TRUE(approxCompare(1.0f, actualSensor._signal->_channels[Channels::SensorFoundResult]));
    EXPECT_TRUE(actualSensor._signal->_channels[Channels::SensorDensity] > 0.2f);
    EXPECT_TRUE(actualSensor._signal->_channels[Channels::SensorDistance] < 1.0f - (100.0f - 10.0f - 16.0f) / 256);
    EXPECT_TRUE(actualSensor._signal->_channels[Channels::SensorDistance] > 1.0f - (100.0f - 10.0f + 16.0f) / 256);
    EXPECT_TRUE(actualSensor._signal->_channels[Channels::SensorAngle] > (-90.0f - 15.0f) / 180);
    EXPECT_TRUE(actualSensor._signal->_channels[Channels::SensorAngle] < (-90.0f + 15.0f) / 180);
}

TEST_F(SensorTests, belowMinDensity)
{
    CollectionDescription data;
    data._cells = {
        CellDescription().id(1).pos({100.0f, 100.0f}).cellTypeData(SensorDescription().autoTriggerInterval(3).minDensity(0.1f)),
        CellDescription().id(2).pos({101.0f, 100.0f}),
    };
    data.addConnection(1, 2);
    data.add(DescriptionEditService::get().get().createRect(
        DescriptionEditService::CreateRectParameters().center({100.0f, 10.0f}).width(16).height(16).cellDistance(2.5f)));

    _simulationFacade->setSimulationData(data);

    _simulationFacade->calcTimesteps(1);
    auto actualSensor = _simulationFacade->getSimulationData().getCellRef(1);
    EXPECT_TRUE(actualSensor._signal.has_value());
    EXPECT_TRUE(approxCompare(1.0f, actualSensor._signal->_channels[Channels::SensorFoundResult]));
}

TEST_F(SensorTests, targetAbove)
{
    CollectionDescription data;
    data._cells = {
        CellDescription().id(1).pos({100.0f, 100.0f}).cellTypeData(SensorDescription().autoTriggerInterval(3).minDensity(0.2f)),
        CellDescription().id(2).pos({101.0f, 100.0f}),
    };
    data.addConnection(1, 2);
    data.add(DescriptionEditService::get().get().createRect(
        DescriptionEditService::CreateRectParameters().center({100.0f, 10.0f}).width(16).height(16).cellDistance(1.0f)));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualSensor = actualData.getCellRef(1);

    EXPECT_TRUE(approxCompare(1.0f, actualSensor._signal->_channels[Channels::SensorFoundResult]));
    EXPECT_TRUE(actualSensor._signal->_channels[Channels::SensorDensity] > 0.2f);
    EXPECT_TRUE(actualSensor._signal->_channels[Channels::SensorDistance] < 1.0f - (100.0f - 10.0f - 16.0f) / 256);
    EXPECT_TRUE(actualSensor._signal->_channels[Channels::SensorDistance] > 1.0f - (100.0f - 10.0f + 16.0f) / 256);
    EXPECT_TRUE(actualSensor._signal->_channels[Channels::SensorAngle] > (-90.0f - 15.0f) / 180);
    EXPECT_TRUE(actualSensor._signal->_channels[Channels::SensorAngle] < (-90.0f + 15.0f) / 180);
}

TEST_F(SensorTests, targetBelow)
{
    CollectionDescription data;
    data._cells = {
        CellDescription().id(1).pos({100.0f, 100.0f}).cellTypeData(SensorDescription().autoTriggerInterval(3).minDensity(0.2f)),
        CellDescription().id(2).pos({101.0f, 100.0f}),
    };
    data.addConnection(1, 2);
    data.add(DescriptionEditService::get().get().createRect(
        DescriptionEditService::CreateRectParameters().center({100.0f, 190.0f}).width(16).height(16).cellDistance(1.0f)));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualSensor = actualData.getCellRef(1);

    EXPECT_TRUE(approxCompare(1.0f, actualSensor._signal->_channels[Channels::SensorFoundResult]));
    EXPECT_TRUE(actualSensor._signal->_channels[Channels::SensorDensity] > 0.2f);
    EXPECT_TRUE(actualSensor._signal->_channels[Channels::SensorDistance] < 1.0f - (100.0f - 10.0f - 16.0f) / 256);
    EXPECT_TRUE(actualSensor._signal->_channels[Channels::SensorDistance] > 1.0f - (100.0f - 10.0f + 16.0f) / 256);
    EXPECT_TRUE(actualSensor._signal->_channels[Channels::SensorAngle] > (90.0f - 15.0f) / 180);
    EXPECT_TRUE(actualSensor._signal->_channels[Channels::SensorAngle] < (90.0f + 15.0f) / 180);
}

TEST_F(SensorTests, targetConcealed)
{
    CollectionDescription data;
    data._cells = {
        CellDescription().id(1).pos({100.0f, 100.0f}).cellTypeData(SensorDescription().autoTriggerInterval(3).minDensity(0.2f)),
        CellDescription().id(2).pos({101.0f, 101.0f}),
        CellDescription().id(3).pos({101.0f, 99.0f}),
    };
    data.addConnection(1, 2);
    data.addConnection(1, 3);
    data.addConnection(2, 3);
    data.add(DescriptionEditService::get().get().createRect(
        DescriptionEditService::CreateRectParameters().center({190.0f, 100.0f}).width(16).height(16).cellDistance(1.0f)));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualSensor = actualData.getCellRef(1);

    EXPECT_TRUE(approxCompare(0.0f, actualSensor._signal->_channels[Channels::SensorFoundResult]));
}

TEST_F(SensorTests, targetNotConcealed)
{
    CollectionDescription data;
    data._cells = {
        CellDescription().id(1).pos({100.0f, 100.0f}).cellTypeData(SensorDescription().autoTriggerInterval(3).minDensity(0.2f)),
        CellDescription().id(2).pos({101.0f, 101.0f}),
        CellDescription().id(3).pos({101.0f, 99.0f}),
    };
    data.addConnection(1, 2);
    data.addConnection(1, 3);
    data.addConnection(2, 3);
    data.add(DescriptionEditService::get().get().createRect(
        DescriptionEditService::CreateRectParameters().center({100.0f, 190.0f}).width(16).height(16).cellDistance(1.0f)));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualSensor = actualData.getCellRef(1);

    EXPECT_TRUE(approxCompare(1.0f, actualSensor._signal->_channels[Channels::SensorFoundResult]));
}

TEST_F(SensorTests, foundMassWithMatchingDensity)
{
    CollectionDescription data;
    data._cells = {
        CellDescription().id(1).pos({100.0f, 100.0f}).cellTypeData(SensorDescription().autoTriggerInterval(3).minDensity(0.7f)),
        CellDescription().id(2).pos({101.0f, 100.0f}),
    };
    data.addConnection(1, 2);

    data.add(DescriptionEditService::get().createRect(
        DescriptionEditService::CreateRectParameters().center({100.0f, 10.0f}).width(16).height(16).cellDistance(1.5f)));
    data.add(DescriptionEditService::get().createRect(
        DescriptionEditService::CreateRectParameters().center({100.0f, 200.0f}).width(16).height(16).cellDistance(1.0f)));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualSensor = actualData.getCellRef(1);

    EXPECT_TRUE(approxCompare(1.0f, actualSensor._signal->_channels[Channels::SensorFoundResult]));
    EXPECT_TRUE(actualSensor._signal->_channels[Channels::SensorDensity] > 0.7f);
    EXPECT_TRUE(actualSensor._signal->_channels[Channels::SensorDistance] < 1.0f - 80.0f / 256);
    EXPECT_TRUE(actualSensor._signal->_channels[Channels::SensorDistance] > 1.0f - 105.0f / 256);
    EXPECT_TRUE(actualSensor._signal->_channels[Channels::SensorAngle] > (90.0f - 15.0f) / 180);
    EXPECT_TRUE(actualSensor._signal->_channels[Channels::SensorAngle] < (90.0f + 15.0f) / 180);
}

TEST_F(SensorTests, scanForOtherMutants_found)
{
    auto data = CollectionDescription().creatures({
        CreatureDescription().id(0).mutationId(6).cells({
            CellDescription()
                .id(1)
                .pos({100.0f, 100.0f})
                .cellTypeData(SensorDescription().restrictToCreatures(SensorRestrictToCreatures_RestrictToOtherMutants)),
            CellDescription().id(2).pos({101.0f, 100.0f}),
        }),
        CreatureDescription().mutationId(7).cells({
            DescriptionEditService::get()
                .createRect(
                    DescriptionEditService::CreateRectParameters().center({100.0f, 10.0f}).width(16).height(16).cellDistance(0.5f).cellType(BaseDescription()))
                ._cells,
        }),
    });
    data.addConnection(1, 2);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualSensorCell = actualData.getCellRef(1);

    EXPECT_TRUE(approxCompare(1.0f, actualSensorCell._signal->_channels[Channels::SensorFoundResult]));
    EXPECT_TRUE(actualSensorCell._signal->_channels[Channels::SensorDensity] > 0.3f);
    EXPECT_TRUE(actualSensorCell._signal->_channels[Channels::SensorDistance] < 1.0f - 80.0f / 256);
    EXPECT_TRUE(actualSensorCell._signal->_channels[Channels::SensorDistance] > 1.0f - 105.0f / 256);
    EXPECT_TRUE(actualSensorCell._signal->_channels[Channels::SensorAngle] > (-90.0f - 15.0f) / 180);
    EXPECT_TRUE(actualSensorCell._signal->_channels[Channels::SensorAngle] < (-90.0f + 15.0f) / 180);
}

TEST_F(SensorTests, scanForOtherMutants_found_wallBehind)
{
    auto data = CollectionDescription().creatures({
        CreatureDescription().id(0).mutationId(6).cells({
            CellDescription()
                .id(1)
                .pos({100.0f, 100.0f})
                .cellTypeData(SensorDescription().restrictToCreatures(SensorRestrictToCreatures_RestrictToOtherMutants)),
            CellDescription().id(2).pos({101.0f, 100.0f}),
        }),
        CreatureDescription().mutationId(7).cells({
            DescriptionEditService::get()
                .createRect(
                    DescriptionEditService::CreateRectParameters().center({200.0f, 100.0f}).width(16).height(16).cellDistance(0.5f).cellType(BaseDescription()))
                ._cells,
        }),
    });
    data.addConnection(1, 2);
    data.add(DescriptionEditService::get().createRect(
        DescriptionEditService::CreateRectParameters().center({220.0f, 100.0f}).width(1).height(16).cellDistance(0.5f).cellType(StructureCellDescription())));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualSensorCell = actualData.getCellRef(1);

    EXPECT_TRUE(approxCompare(1.0f, actualSensorCell._signal->_channels[Channels::SensorFoundResult]));
}

TEST_F(SensorTests, scanForOtherMutants_notFound_wallInBetween)
{
    auto data = CollectionDescription().creatures({
        CreatureDescription().id(0).mutationId(6).cells({
            CellDescription()
                .id(1)
                .pos({100.0f, 100.0f})
                .cellTypeData(SensorDescription().restrictToCreatures(SensorRestrictToCreatures_RestrictToOtherMutants)),
            CellDescription().id(2).pos({101.0f, 100.0f}),
        }),
        CreatureDescription().mutationId(7).cells({
            DescriptionEditService::get()
                .createRect(
                    DescriptionEditService::CreateRectParameters().center({200.0f, 100.0f}).width(16).height(16).cellDistance(0.5f).cellType(BaseDescription()))
                ._cells,
        }),
    });
    data.addConnection(1, 2);
    data.add(DescriptionEditService::get().createRect(
        DescriptionEditService::CreateRectParameters().center({150.0f, 100.0f}).width(1).height(16).cellDistance(0.5f).cellType(StructureCellDescription())));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualSensorCell = actualData.getCellRef(1);

    EXPECT_TRUE(approxCompare(0.0f, actualSensorCell._signal->_channels[Channels::SensorFoundResult]));
}

TEST_F(SensorTests, scanForOtherMutants_notFound_sameMutationId)
{
    auto data = CollectionDescription().creatures({
        CreatureDescription().id(0).mutationId(7).cells({
            CellDescription()
                .id(1)
                .pos({100.0f, 100.0f})
                .cellTypeData(SensorDescription().restrictToCreatures(SensorRestrictToCreatures_RestrictToOtherMutants)),
            CellDescription().id(2).pos({101.0f, 100.0f}),
        }),
        CreatureDescription().mutationId(7).cells({
            DescriptionEditService::get()
                .createRect(
                    DescriptionEditService::CreateRectParameters().center({10.0f, 100.0f}).width(16).height(16).cellDistance(0.5f).cellType(BaseDescription()))
                ._cells,
        }),
    });
    data.addConnection(1, 2);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualSensorCell = actualData.getCellRef(1);

    EXPECT_TRUE(approxCompare(0.0f, actualSensorCell._signal->_channels[Channels::SensorFoundResult]));
}

TEST_F(SensorTests, scanForOtherMutants_notFound_structure)
{
    auto data = CollectionDescription().creatures({
        CreatureDescription().id(0).mutationId(7).cells({
            CellDescription()
                .id(1)
                .pos({100.0f, 100.0f})
                .cellTypeData(SensorDescription().restrictToCreatures(SensorRestrictToCreatures_RestrictToOtherMutants)),
            CellDescription().id(2).pos({101.0f, 100.0f}),
        }),
    });
    data.add(DescriptionEditService::get().createRect(
        DescriptionEditService::CreateRectParameters().center({10.0f, 100.0f}).width(16).height(16).cellDistance(0.5f).cellType(StructureCellDescription())));
    data.addConnection(1, 2);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualSensorCell = actualData.getCellRef(1);

    EXPECT_TRUE(approxCompare(0.0f, actualSensorCell._signal->_channels[Channels::SensorFoundResult]));
}

TEST_F(SensorTests, scanForOtherMutants_notFound_freeCell)
{
    auto data = CollectionDescription().creatures({
        CreatureDescription().id(0).mutationId(7).cells({
            CellDescription()
                .id(1)
                .pos({100.0f, 100.0f})
                .cellTypeData(SensorDescription().restrictToCreatures(SensorRestrictToCreatures_RestrictToOtherMutants)),
            CellDescription().id(2).pos({101.0f, 100.0f}),
        }),
    });
    data.add(DescriptionEditService::get().createRect(
        DescriptionEditService::CreateRectParameters().center({10.0f, 100.0f}).width(16).height(16).cellDistance(0.5f).cellType(FreeCellDescription())));
    data.addConnection(1, 2);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualSensorCell = actualData.getCellRef(1);

    EXPECT_TRUE(approxCompare(0.0f, actualSensorCell._signal->_channels[Channels::SensorFoundResult]));
}

TEST_F(SensorTests, scanForSameMutants_found)
{
    auto data = CollectionDescription().creatures({
        CreatureDescription().id(0).mutationId(6).cells({
            CellDescription()
                .id(1)
                .pos({100.0f, 100.0f}).cellTypeData(SensorDescription().restrictToCreatures(SensorRestrictToCreatures_RestrictToSameMutants)),
            CellDescription().id(2).pos({101.0f, 100.0f}),
        }),
        CreatureDescription().mutationId(6).cells({
            DescriptionEditService::get()
                .createRect(
                    DescriptionEditService::CreateRectParameters().center({10.0f, 100.0f}).width(16).height(16).cellDistance(0.5f).cellType(BaseDescription()))
                ._cells,
        }),
    });
    data.addConnection(1, 2);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualSensorCell = actualData.getCellRef(1);

    EXPECT_TRUE(approxCompare(1.0f, actualSensorCell._signal->_channels[Channels::SensorFoundResult]));
}

TEST_F(SensorTests, scanForSameMutants_notFound_otherMutationId)
{
    auto const MutantId = 6;
    for (int otherMutantId = 0; otherMutantId < 100; ++otherMutantId) {
        if (otherMutantId == MutantId) {
            continue;
        }

        auto data = CollectionDescription().creatures({
            CreatureDescription().id(0).mutationId(MutantId).cells({
                CellDescription()
                    .id(1)
                    .pos({100.0f, 100.0f})
                    .cellTypeData(SensorDescription().restrictToCreatures(SensorRestrictToCreatures_RestrictToSameMutants)),
                CellDescription().id(2).pos({101.0f, 100.0f}),
            }),
            CreatureDescription()
                .mutationId(otherMutantId)
                .cells({
                    DescriptionEditService::get()
                        .createRect(DescriptionEditService::CreateRectParameters()
                                        .center({10.0f, 100.0f})
                                        .width(16)
                                        .height(16)
                                        .cellDistance(0.5f)
                                        .cellType(BaseDescription()))
                        ._cells,
                }),
        });
        data.addConnection(1, 2);

        _simulationFacade->clear();
        _simulationFacade->setCurrentTimestep(0ull);
        _simulationFacade->setSimulationData(data);
        _simulationFacade->calcTimesteps(1);

        auto actualData = _simulationFacade->getSimulationData();
        auto actualSensorCell = actualData.getCellRef(1);

        EXPECT_TRUE(approxCompare(0.0f, actualSensorCell._signal->_channels[Channels::SensorFoundResult]));
    }
}

TEST_F(SensorTests, scanForSameMutants_notFound_structure)
{
    auto const MutantId = 6;

    auto data = CollectionDescription().creatures({
        CreatureDescription().id(0).mutationId(MutantId).cells({
            CellDescription()
                .id(1)
                .pos({100.0f, 100.0f}).cellTypeData(SensorDescription().restrictToCreatures(SensorRestrictToCreatures_RestrictToSameMutants)),
            CellDescription().id(2).pos({101.0f, 100.0f}),
        }),
    });
    data.add(DescriptionEditService::get().createRect(
        DescriptionEditService::CreateRectParameters().center({10.0f, 100.0f}).width(16).height(16).cellDistance(0.5f).cellType(StructureCellDescription())));
    data.addConnection(1, 2);

    _simulationFacade->clear();
    _simulationFacade->setCurrentTimestep(0ull);
    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualSensorCell = actualData.getCellRef(1);

    EXPECT_TRUE(approxCompare(0.0f, actualSensorCell._signal->_channels[Channels::SensorFoundResult]));
}

TEST_F(SensorTests, scanForSameMutants_notFound_freeCell)
{
    auto const MutantId = 6;

    auto data = CollectionDescription().creatures({
        CreatureDescription().id(0).mutationId(MutantId).cells({
            CellDescription().id(1).pos({100.0f, 100.0f}).cellTypeData(SensorDescription().restrictToCreatures(SensorRestrictToCreatures_RestrictToSameMutants)),
            CellDescription().id(2).pos({101.0f, 100.0f}),
        }),
    });
    data.add(DescriptionEditService::get().createRect(
        DescriptionEditService::CreateRectParameters().center({10.0f, 100.0f}).width(16).height(16).cellDistance(0.5f).cellType(FreeCellDescription())));
    data.addConnection(1, 2);

    _simulationFacade->clear();
    _simulationFacade->setCurrentTimestep(0ull);
    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualSensorCell = actualData.getCellRef(1);

    EXPECT_TRUE(approxCompare(0.0f, actualSensorCell._signal->_channels[Channels::SensorFoundResult]));
}

TEST_F(SensorTests, scanForStructures_found)
{
    CollectionDescription data;
    data._cells = {
        CellDescription().id(1).pos({100.0f, 100.0f}).cellTypeData(SensorDescription().restrictToCreatures(SensorRestrictToCreatures_RestrictToStructures)),
        CellDescription().id(2).pos({101.0f, 100.0f}),
    };
    data.addConnection(1, 2);

    data.add(DescriptionEditService::get().createRect(
        DescriptionEditService::CreateRectParameters().center({10.0f, 100.0f}).width(16).height(16).cellDistance(0.5f).cellType(StructureCellDescription())));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualSensorCell = actualData.getCellRef(1);

    EXPECT_TRUE(approxCompare(1.0f, actualSensorCell._signal->_channels[Channels::SensorFoundResult]));
}

TEST_F(SensorTests, scanForStructures_notFound)
{
    CollectionDescription data;
    data._cells = {
        CellDescription().id(1).pos({100.0f, 100.0f}).cellTypeData(SensorDescription().restrictToCreatures(SensorRestrictToCreatures_RestrictToStructures)),
        CellDescription().id(2).pos({101.0f, 100.0f}),
    };
    data.addConnection(1, 2);

    data.add(DescriptionEditService::get().createRect(
        DescriptionEditService::CreateRectParameters().center({10.0f, 100.0f}).width(16).height(16).cellDistance(0.5f).cellType(BaseDescription())));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualSensorCell = actualData.getCellRef(1);

    EXPECT_TRUE(approxCompare(0.0f, actualSensorCell._signal->_channels[Channels::SensorFoundResult]));
}

TEST_F(SensorTests, scanForFreeCells_found)
{
    CollectionDescription data;
    data._cells = {
        CellDescription().id(1).pos({100.0f, 100.0f}).cellTypeData(SensorDescription().restrictToCreatures(SensorRestrictToCreatures_RestrictToFreeCells)),
        CellDescription().id(2).pos({101.0f, 100.0f}),
    };
    data.addConnection(1, 2);

    data.add(DescriptionEditService::get().createRect(
        DescriptionEditService::CreateRectParameters().center({10.0f, 100.0f}).width(16).height(16).cellDistance(0.5f).cellType(FreeCellDescription())));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualSensorCell = actualData.getCellRef(1);

    EXPECT_TRUE(approxCompare(1.0f, actualSensorCell._signal->_channels[Channels::SensorFoundResult]));
}

TEST_F(SensorTests, scanForFreeCells_notFound)
{
    CollectionDescription data;
    data._cells = {
        CellDescription().id(1).pos({100.0f, 100.0f}).cellTypeData(SensorDescription().restrictToCreatures(SensorRestrictToCreatures_RestrictToFreeCells)),
        CellDescription().id(2).pos({101.0f, 100.0f}),
    };
    data.addConnection(1, 2);

    data.add(DescriptionEditService::get().createRect(
        DescriptionEditService::CreateRectParameters().center({10.0f, 100.0f}).width(16).height(16).cellDistance(0.5f).cellType(StructureCellDescription())));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualSensorCell = actualData.getCellRef(1);

    EXPECT_TRUE(approxCompare(0.0f, actualSensorCell._signal->_channels[Channels::SensorFoundResult]));
}

TEST_F(SensorTests, scanForLessComplexMutants_found)
{
    for (int otherGenomeComplexity = 0; otherGenomeComplexity < 500; ++otherGenomeComplexity) {
        auto data = CollectionDescription().creatures({
            CreatureDescription().id(0).genomeComplexity(1000.0f).cells({
                CellDescription()
                    .id(1)
                    .pos({100.0f, 100.0f})
                    .cellTypeData(SensorDescription().restrictToCreatures(SensorRestrictToCreatures_RestrictToLessComplexMutants)),
                CellDescription().id(2).pos({101.0f, 100.0f}),
            }),
            CreatureDescription()
                .genomeComplexity(toFloat(otherGenomeComplexity))
                .cells({
                    DescriptionEditService::get()
                        .createRect(DescriptionEditService::CreateRectParameters()
                                        .center({10.0f, 100.0f})
                                        .width(16)
                                        .height(16)
                                        .cellDistance(0.5f)
                                        .cellType(BaseDescription()))
                        ._cells,
                }),
        });
        data.addConnection(1, 2);

        _simulationFacade->clear();
        _simulationFacade->setCurrentTimestep(0ull);
        _simulationFacade->setSimulationData(data);
        _simulationFacade->calcTimesteps(1);

        auto actualData = _simulationFacade->getSimulationData();
        auto actualSensorCell = actualData.getCellRef(1);

        EXPECT_TRUE(approxCompare(1.0f, actualSensorCell._signal->_channels[Channels::SensorFoundResult]));
    }
}

TEST_F(SensorTests, scanForLessComplexMutants_notFound_otherMoreComplex)
{
    for (int otherGenomeComplexity = 1000; otherGenomeComplexity < 2001; ++otherGenomeComplexity) {

        auto data = CollectionDescription().creatures({
            CreatureDescription().id(0).genomeComplexity(1000.0f).cells({
                CellDescription()
                    .id(1)
                    .pos({100.0f, 100.0f})
                    .cellTypeData(SensorDescription().restrictToCreatures(SensorRestrictToCreatures_RestrictToLessComplexMutants)),
                CellDescription().id(2).pos({101.0f, 100.0f}),
            }),
            CreatureDescription()
                .genomeComplexity(toFloat(otherGenomeComplexity) + 0.01f)
                .cells({
                    DescriptionEditService::get()
                        .createRect(DescriptionEditService::CreateRectParameters()
                                        .center({10.0f, 100.0f})
                                        .width(16)
                                        .height(16)
                                        .cellDistance(0.5f)
                                        .cellType(BaseDescription()))
                        ._cells,
                }),
        });
        data.addConnection(1, 2);

        _simulationFacade->clear();
        _simulationFacade->setCurrentTimestep(0ull);
        _simulationFacade->setSimulationData(data);
        _simulationFacade->calcTimesteps(1);

        auto actualData = _simulationFacade->getSimulationData();
        auto actualSensorCell = actualData.getCellRef(1);

        EXPECT_TRUE(approxCompare(0.0f, actualSensorCell._signal->_channels[Channels::SensorFoundResult]));
    }
}

TEST_F(SensorTests, scanForLessComplexMutants_notFound_structure)
{
    auto data = CollectionDescription().creatures({
        CreatureDescription().id(0).genomeComplexity(1000.0f).cells({
            CellDescription()
                .id(1)
                .pos({100.0f, 100.0f})
                .cellTypeData(SensorDescription().restrictToCreatures(SensorRestrictToCreatures_RestrictToLessComplexMutants)),
            CellDescription().id(2).pos({101.0f, 100.0f}),
        }),
    });
    data.add(DescriptionEditService::get().createRect(
        DescriptionEditService::CreateRectParameters().center({10.0f, 100.0f}).width(16).height(16).cellDistance(0.5f).cellType(StructureCellDescription())));
    data.addConnection(1, 2);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualSensorCell = actualData.getCellRef(1);

    EXPECT_TRUE(approxCompare(0.0f, actualSensorCell._signal->_channels[Channels::SensorFoundResult]));
}

TEST_F(SensorTests, scanForLessComplexMutants_notFound_freeCell)
{
    auto data = CollectionDescription().creatures({
        CreatureDescription().id(0).genomeComplexity(1000.0f).cells({
            CellDescription()
                .id(1)
                .pos({100.0f, 100.0f})
                .cellTypeData(SensorDescription().restrictToCreatures(SensorRestrictToCreatures_RestrictToLessComplexMutants)),
            CellDescription().id(2).pos({101.0f, 100.0f}),
        }),
    });
    data.add(DescriptionEditService::get().createRect(
        DescriptionEditService::CreateRectParameters().center({10.0f, 100.0f}).width(16).height(16).cellDistance(0.5f).cellType(FreeCellDescription())));
    data.addConnection(1, 2);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualSensorCell = actualData.getCellRef(1);

    EXPECT_TRUE(approxCompare(0.0f, actualSensorCell._signal->_channels[Channels::SensorFoundResult]));
}

TEST_F(SensorTests, scanForMoreComplexMutants_found)
{
    for (int otherGenomeComplexity = 1000; otherGenomeComplexity < 2001; ++otherGenomeComplexity) {
        auto data = CollectionDescription().creatures({
            CreatureDescription().id(0).genomeComplexity(500.0f).cells({
                CellDescription()
                    .id(1)
                    .pos({100.0f, 100.0f})
                    .cellTypeData(SensorDescription().restrictToCreatures(SensorRestrictToCreatures_RestrictToMoreComplexMutants)),
                CellDescription().id(2).pos({101.0f, 100.0f}),
            }),
            CreatureDescription()
                .genomeComplexity(toFloat(otherGenomeComplexity))
                .cells({
                    DescriptionEditService::get()
                        .createRect(DescriptionEditService::CreateRectParameters()
                                        .center({10.0f, 100.0f})
                                        .width(16)
                                        .height(16)
                                        .cellDistance(0.5f)
                                        .cellType(BaseDescription()))
                        ._cells,
                }),
        });
        data.addConnection(1, 2);

        _simulationFacade->clear();
        _simulationFacade->setCurrentTimestep(0ull);
        _simulationFacade->setSimulationData(data);
        _simulationFacade->calcTimesteps(1);

        auto actualData = _simulationFacade->getSimulationData();
        auto actualSensorCell = actualData.getCellRef(1);

        EXPECT_TRUE(approxCompare(1.0f, actualSensorCell._signal->_channels[Channels::SensorFoundResult]));
    }
}

TEST_F(SensorTests, scanForMoreComplexMutants_notFound_otherLessComplex)
{
    for (int otherGenomeComplexity = 0; otherGenomeComplexity < 500; ++otherGenomeComplexity) {
        auto data = CollectionDescription().creatures({
            CreatureDescription().id(0).genomeComplexity(500.0f).cells({
                CellDescription()
                    .id(1)
                    .pos({100.0f, 100.0f})
                    .cellTypeData(SensorDescription().restrictToCreatures(SensorRestrictToCreatures_RestrictToMoreComplexMutants)),
                CellDescription().id(2).pos({101.0f, 100.0f}),
            }),
            CreatureDescription()
                .genomeComplexity(toFloat(otherGenomeComplexity))
                .cells({
                    DescriptionEditService::get()
                        .createRect(DescriptionEditService::CreateRectParameters()
                                        .center({10.0f, 100.0f})
                                        .width(16)
                                        .height(16)
                                        .cellDistance(0.5f)
                                        .cellType(BaseDescription()))
                        ._cells,
                }),
        });
        data.addConnection(1, 2);


        _simulationFacade->clear();
        _simulationFacade->setCurrentTimestep(0ull);
        _simulationFacade->setSimulationData(data);
        _simulationFacade->calcTimesteps(1);

        auto actualData = _simulationFacade->getSimulationData();
        auto actualSensorCell = actualData.getCellRef(1);

        EXPECT_TRUE(approxCompare(0.0f, actualSensorCell._signal->_channels[Channels::SensorFoundResult]));
    }
}

TEST_F(SensorTests, scanForMoreComplexMutants_notFound_structure)
{
    auto data = CollectionDescription().creatures({
        CreatureDescription().id(0).genomeComplexity(100.0f).cells({
            CellDescription()
                .id(1)
                .pos({100.0f, 100.0f})
                .cellTypeData(SensorDescription().restrictToCreatures(SensorRestrictToCreatures_RestrictToMoreComplexMutants)),
            CellDescription().id(2).pos({101.0f, 100.0f}),
        }),
    });
    data.add(DescriptionEditService::get().createRect(
        DescriptionEditService::CreateRectParameters().center({10.0f, 100.0f}).width(16).height(16).cellDistance(0.5f).cellType(StructureCellDescription())));
    data.addConnection(1, 2);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualSensorCell = actualData.getCellRef(1);

    EXPECT_TRUE(approxCompare(0.0f, actualSensorCell._signal->_channels[Channels::SensorFoundResult]));
}

TEST_F(SensorTests, scanForMoreComplexMutants_notFound_freeCell)
{
    auto data = CollectionDescription().creatures({
        CreatureDescription().id(0).genomeComplexity(100.0f).cells({
            CellDescription()
                .id(1)
                .pos({100.0f, 100.0f})
                .cellTypeData(SensorDescription().restrictToCreatures(SensorRestrictToCreatures_RestrictToMoreComplexMutants)),
            CellDescription().id(2).pos({101.0f, 100.0f}),
        }),
    });
    data.add(DescriptionEditService::get().createRect(
        DescriptionEditService::CreateRectParameters().center({10.0f, 100.0f}).width(16).height(16).cellDistance(0.5f).cellType(FreeCellDescription())));
    data.addConnection(1, 2);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualSensorCell = actualData.getCellRef(1);

    EXPECT_TRUE(approxCompare(0.0f, actualSensorCell._signal->_channels[Channels::SensorFoundResult]));
}

TEST_F(SensorTests, minRange_found)
{
    CollectionDescription data;
    data._cells = {
        CellDescription().id(1).pos({100.0f, 100.0f}).cellTypeData(SensorDescription().minRange(50)),
        CellDescription().id(2).pos({101.0f, 100.0f}),
    };
    data.addConnection(1, 2);

    data.add(DescriptionEditService::get().createRect(
        DescriptionEditService::CreateRectParameters().center({10.0f, 100.0f}).width(16).height(16).cellDistance(0.5f).cellType(BaseDescription())));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualSensorCell = actualData.getCellRef(1);

    EXPECT_TRUE(approxCompare(1.0f, actualSensorCell._signal->_channels[Channels::SensorFoundResult]));
}

TEST_F(SensorTests, minRange_notFound)
{
    CollectionDescription data;
    data._cells = {
        CellDescription().id(1).pos({100.0f, 100.0f}).cellTypeData(SensorDescription().minRange(120)),
        CellDescription().id(2).pos({101.0f, 100.0f}),
    };
    data.addConnection(1, 2);

    data.add(DescriptionEditService::get().createRect(
        DescriptionEditService::CreateRectParameters().center({10.0f, 100.0f}).width(16).height(16).cellDistance(0.5f).cellType(BaseDescription())));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualSensorCell = actualData.getCellRef(1);

    EXPECT_TRUE(approxCompare(0.0f, actualSensorCell._signal->_channels[Channels::SensorFoundResult]));
}

TEST_F(SensorTests, maxRange_found)
{
    CollectionDescription data;
    data._cells = {
        CellDescription().id(1).pos({100.0f, 100.0f}).cellTypeData(SensorDescription().maxRange(120)),
        CellDescription().id(2).pos({101.0f, 100.0f}),
    };
    data.addConnection(1, 2);

    data.add(DescriptionEditService::get().createRect(
        DescriptionEditService::CreateRectParameters().center({10.0f, 100.0f}).width(16).height(16).cellDistance(0.5f).cellType(BaseDescription())));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualSensorCell = actualData.getCellRef(1);

    EXPECT_TRUE(approxCompare(1.0f, actualSensorCell._signal->_channels[Channels::SensorFoundResult]));
}

TEST_F(SensorTests, maxRange_notFound)
{
    CollectionDescription data;
    data._cells = {
        CellDescription().id(1).pos({100.0f, 100.0f}).cellTypeData(SensorDescription().maxRange(50)),
        CellDescription().id(2).pos({101.0f, 100.0f}),
    };
    data.addConnection(1, 2);

    data.add(DescriptionEditService::get().createRect(
        DescriptionEditService::CreateRectParameters().center({10.0f, 100.0f}).width(16).height(16).cellDistance(0.5f).cellType(BaseDescription())));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualSensorCell = actualData.getCellRef(1);

    EXPECT_TRUE(approxCompare(0.0f, actualSensorCell._signal->_channels[Channels::SensorFoundResult]));
}
