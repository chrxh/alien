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
    DataDescription data;
    data.addCells({
        CellDescription().setId(1).setPos({100.0f, 100.0f}).setCellTypeData(SensorDescription().setAutoTriggerInterval(15)),
    });
    _simulationFacade->setSimulationData(data);

    {
        _simulationFacade->calcTimesteps(1);
        auto actualSensor = getCell(_simulationFacade->getSimulationData(), 1);
        EXPECT_TRUE(actualSensor.signal.has_value());
    }
    {
        _simulationFacade->calcTimesteps(1);
        auto actualSensor = getCell(_simulationFacade->getSimulationData(), 1);
        EXPECT_FALSE(actualSensor.signal.has_value());
    }
    {
        _simulationFacade->calcTimesteps(14);
        auto actualSensor = getCell(_simulationFacade->getSimulationData(), 1);
        EXPECT_TRUE(actualSensor.signal.has_value());
    }
    {
        _simulationFacade->calcTimesteps(1);
        auto actualSensor = getCell(_simulationFacade->getSimulationData(), 1);
        EXPECT_FALSE(actualSensor.signal.has_value());
    }
}

TEST_F(SensorTests_New, manuallyTriggered_noSignal)
{
    DataDescription data;
    data.addCells({
        CellDescription().setId(1).setPos({100.0f, 100.0f}).setCellTypeData(SensorDescription().setAutoTriggerInterval(0)),
    });
    _simulationFacade->setSimulationData(data);

    for(int i = 0; i < 100; ++i) {
        _simulationFacade->calcTimesteps(1);
        auto actualSensor = getCell(_simulationFacade->getSimulationData(), 1);
        EXPECT_FALSE(actualSensor.signal.has_value());
    }
}

TEST_F(SensorTests_New, manuallyTriggered_signal)
{
    DataDescription data;
    data.addCells({
        CellDescription().setId(1).setPos({100.0f, 100.0f}).setCellTypeData(SensorDescription().setAutoTriggerInterval(0)),
        CellDescription().setId(2).setPos({101.0f, 100.0f}).setSignal({1, 0, 0, 0, 0, 0, 0, 0}),
    });
    data.addConnection(1, 2);
    _simulationFacade->setSimulationData(data);

    _simulationFacade->calcTimesteps(1);
    auto actualSensor = getCell(_simulationFacade->getSimulationData(), 1);
    EXPECT_TRUE(actualSensor.signal.has_value());
}

TEST_F(SensorTests_New, aboveMinDensity)
{
    DataDescription data;
    data.addCells({
        CellDescription().setId(1).setPos({100.0f, 100.0f}).setCellTypeData(SensorDescription().setAutoTriggerInterval(3).setMinDensity(0.2f)),
        CellDescription().setId(2).setPos({101.0f, 100.0f}),
    });
    data.addConnection(1, 2);
    data.add(DescriptionEditService::get().get().createRect(
        DescriptionEditService::CreateRectParameters().center({10.0f, 100.0f}).width(16).height(16).cellDistance(1.0f)));

    _simulationFacade->setSimulationData(data);

    _simulationFacade->calcTimesteps(1);
    auto actualSensor = getCell(_simulationFacade->getSimulationData(), 1);
    EXPECT_TRUE(actualSensor.signal.has_value());
    EXPECT_TRUE(approxCompare(1.0f, actualSensor.signal->channels[0]));
    EXPECT_TRUE(actualSensor.signal->channels[1] > 0.2f);
    EXPECT_TRUE(actualSensor.signal->channels[2] < 1.0f - (100.0f - 10.0f - 16.0f) / 256);
    EXPECT_TRUE(actualSensor.signal->channels[2] > 1.0f - (100.0f - 10.0f + 16.0f) / 256);
    EXPECT_TRUE(actualSensor.signal->channels[3] > -15.0f / 365);
    EXPECT_TRUE(actualSensor.signal->channels[3] < 15.0f / 365);
}

TEST_F(SensorTests_New, belowMinDensity)
{
    DataDescription data;
    data.addCells({
        CellDescription().setId(1).setPos({100.0f, 100.0f}).setCellTypeData(SensorDescription().setAutoTriggerInterval(3).setMinDensity(0.1f)),
        CellDescription().setId(2).setPos({101.0f, 100.0f}),
    });
    data.addConnection(1, 2);
    data.add(DescriptionEditService::get().get().createRect(
        DescriptionEditService::CreateRectParameters().center({2.0f, 2.0f}).width(10).height(10).cellDistance(2.5f)));

    _simulationFacade->setSimulationData(data);

    _simulationFacade->calcTimesteps(1);
    auto actualSensor = getCell(_simulationFacade->getSimulationData(), 1);
    EXPECT_TRUE(actualSensor.signal.has_value());
    EXPECT_TRUE(approxCompare(1.0f, actualSensor.signal->channels[0]));
}

TEST_F(SensorTests_New, targetAbove)
{
    DataDescription data;
    data.addCells({
        CellDescription().setId(1).setPos({100.0f, 100.0f}).setCellTypeData(SensorDescription().setAutoTriggerInterval(3).setMinDensity(0.2f)),
        CellDescription().setId(2).setPos({101.0f, 100.0f}),
    });
    data.addConnection(1, 2);
    data.add(DescriptionEditService::get().get().createRect(
        DescriptionEditService::CreateRectParameters().center({100.0f, 10.0f}).width(16).height(16).cellDistance(1.0f)));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualSensor = getCell(actualData, 1);

    EXPECT_TRUE(approxCompare(1.0f, actualSensor.signal->channels[0]));
    EXPECT_TRUE(actualSensor.signal->channels[1] > 0.2f);
    EXPECT_TRUE(actualSensor.signal->channels[2] < 1.0f - (100.0f - 10.0f - 16.0f) / 256);
    EXPECT_TRUE(actualSensor.signal->channels[2] > 1.0f - (100.0f - 10.0f + 16.0f) / 256);
    EXPECT_TRUE(actualSensor.signal->channels[3] > 70.0f / 365);
    EXPECT_TRUE(actualSensor.signal->channels[3] < 105.0f / 365);
}

TEST_F(SensorTests_New, targetBelow)
{
    DataDescription data;
    data.addCells({
        CellDescription().setId(1).setPos({100.0f, 100.0f}).setCellTypeData(SensorDescription().setAutoTriggerInterval(3).setMinDensity(0.2f)),
        CellDescription().setId(2).setPos({101.0f, 100.0f}),
    });
    data.addConnection(1, 2);
    data.add(DescriptionEditService::get().get().createRect(
        DescriptionEditService::CreateRectParameters().center({100.0f, 190.0f}).width(16).height(16).cellDistance(1.0f)));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualSensor = getCell(actualData, 1);

    EXPECT_TRUE(approxCompare(1.0f, actualSensor.signal->channels[0]));
    EXPECT_TRUE(actualSensor.signal->channels[1] > 0.2f);
    EXPECT_TRUE(actualSensor.signal->channels[2] < 1.0f - (100.0f - 10.0f - 16.0f) / 256);
    EXPECT_TRUE(actualSensor.signal->channels[2] > 1.0f - (100.0f - 10.0f + 16.0f) / 256);
    EXPECT_TRUE(actualSensor.signal->channels[3] < -70.0f / 365);
    EXPECT_TRUE(actualSensor.signal->channels[3] > -105.0f / 365);
}

TEST_F(SensorTests_New, targetConcealed)
{
    DataDescription data;
    data.addCells({
        CellDescription().setId(1).setPos({100.0f, 100.0f}).setCellTypeData(SensorDescription().setAutoTriggerInterval(3).setMinDensity(0.2f)),
        CellDescription().setId(2).setPos({101.0f, 101.0f}),
        CellDescription().setId(3).setPos({101.0f, 99.0f}),
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

    EXPECT_TRUE(approxCompare(0.0f, actualSensor.signal->channels[0]));
}

TEST_F(SensorTests_New, targetNotConcealed)
{
    DataDescription data;
    data.addCells({
        CellDescription().setId(1).setPos({100.0f, 100.0f}).setCellTypeData(SensorDescription().setAutoTriggerInterval(3).setMinDensity(0.2f)),
        CellDescription().setId(2).setPos({101.0f, 101.0f}),
        CellDescription().setId(3).setPos({101.0f, 99.0f}),
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

    EXPECT_TRUE(approxCompare(1.0f, actualSensor.signal->channels[0]));
}

TEST_F(SensorTests_New, foundMassWithMatchingDensity)
{
    DataDescription data;
    data.addCells({
        CellDescription().setId(1).setPos({100.0f, 100.0f}).setCellTypeData(SensorDescription().setAutoTriggerInterval(3).setMinDensity(0.7f)),
        CellDescription().setId(2).setPos({101.0f, 100.0f}),
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

    EXPECT_TRUE(approxCompare(1.0f, actualSensor.signal->channels[0]));
    EXPECT_TRUE(actualSensor.signal->channels[1] > 0.7f);
    EXPECT_TRUE(actualSensor.signal->channels[2] < 1.0f - 80.0f / 256);
    EXPECT_TRUE(actualSensor.signal->channels[2] > 1.0f - 105.0f / 256);
    EXPECT_TRUE(actualSensor.signal->channels[3] < -70.0f / 365);
    EXPECT_TRUE(actualSensor.signal->channels[3] > -105.0f / 365);
}

TEST_F(SensorTests_New, scanForOtherMutants_found)
{
    DataDescription data;
    data.addCells(
        {CellDescription()
             .setId(1)
             .setMutationId(6)
             .setPos({100.0f, 100.0f})
            .setCellTypeData(SensorDescription().setRestrictToMutants(SensorRestrictToMutants_RestrictToOtherMutants)),
        CellDescription().setId(2).setMutationId(6).setPos({101.0f, 100.0f}),
    });
    data.addConnection(1, 2);

    data.add(DescriptionEditService::get().createRect(DescriptionEditService::CreateRectParameters()
                                                          .center({10.0f, 100.0f})
                                                          .width(16)
                                                          .height(16)
                                                          .cellDistance(0.5f)
                                                          .mutationId(7)
                                                          .cellType(BaseDescription())));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualSensorCell = getCell(actualData, 1);

    EXPECT_TRUE(approxCompare(1.0f, actualSensorCell.signal->channels[0]));
    EXPECT_TRUE(actualSensorCell.signal->channels[1] > 0.3f);
    EXPECT_TRUE(actualSensorCell.signal->channels[2] < 1.0f - 80.0f / 256);
    EXPECT_TRUE(actualSensorCell.signal->channels[2] > 1.0f - 105.0f / 256);
    EXPECT_TRUE(actualSensorCell.signal->channels[3] > -15.0f / 365);
    EXPECT_TRUE(actualSensorCell.signal->channels[3] < 15.0f / 365);
}

TEST_F(SensorTests_New, scanForOtherMutants_found_wallBehind)
{
    DataDescription data;
    data.addCells(
        {CellDescription()
             .setId(1)
             .setMutationId(6)
             .setPos({100.0f, 100.0f})
             .setCellTypeData(SensorDescription().setRestrictToMutants(SensorRestrictToMutants_RestrictToOtherMutants)),
        CellDescription().setId(2).setMutationId(6).setPos({101.0f, 100.0f}),
    });
    data.addConnection(1, 2);

    data.add(DescriptionEditService::get().createRect(
        DescriptionEditService::CreateRectParameters().center({0.0f, 100.0f}).width(1).height(16).cellDistance(0.5f).cellType(StructureCellDescription())));

    data.add(DescriptionEditService::get().createRect(DescriptionEditService::CreateRectParameters()
                                                          .center({10.0f, 100.0f})
                                                          .width(16)
                                                          .height(16)
                                                          .cellDistance(0.5f)
                                                          .mutationId(7)
                                                          .cellType(BaseDescription())));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualSensorCell = getCell(actualData, 1);

    EXPECT_TRUE(approxCompare(1.0f, actualSensorCell.signal->channels[0]));
}

TEST_F(SensorTests_New, scanForOtherMutants_notFound_wallInBetween)
{
    DataDescription data;
    data.addCells(
        {CellDescription()
             .setId(1)
             .setMutationId(7)
             .setPos({100.0f, 100.0f})
             .setCellTypeData(SensorDescription().setRestrictToMutants(SensorRestrictToMutants_RestrictToOtherMutants)),
         CellDescription().setId(2).setMutationId(7).setPos({101.0f, 100.0f})});
    data.addConnection(1, 2);

    data.add(DescriptionEditService::get().createRect(
        DescriptionEditService::CreateRectParameters().center({50.0f, 100.0f}).width(1).height(16).cellDistance(0.5f).cellType(StructureCellDescription())));

    data.add(DescriptionEditService::get().createRect(DescriptionEditService::CreateRectParameters()
                                                          .center({10.0f, 100.0f})
                                                          .width(16)
                                                          .height(16)
                                                          .cellDistance(0.5f)
                                                          .mutationId(7)
                                                          .cellType(BaseDescription())));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualSensorCell = getCell(actualData, 1);

    EXPECT_TRUE(approxCompare(0.0f, actualSensorCell.signal->channels[0]));
}

TEST_F(SensorTests_New, scanForOtherMutants_notFound_sameMutationId)
{
    DataDescription data;
    data.addCells(
        {CellDescription()
             .setId(1)
             .setMutationId(7)
             .setPos({100.0f, 100.0f})
             .setCellTypeData(SensorDescription().setRestrictToMutants(SensorRestrictToMutants_RestrictToOtherMutants)),
        CellDescription().setId(2).setMutationId(7).setPos({101.0f, 100.0f}),
    });
    data.addConnection(1, 2);

    data.add(DescriptionEditService::get().createRect(DescriptionEditService::CreateRectParameters()
                                                          .center({10.0f, 100.0f})
                                                          .width(16)
                                                          .height(16)
                                                          .cellDistance(0.5f)
                                                          .mutationId(7)
                                                          .cellType(BaseDescription())));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualSensorCell = getCell(actualData, 1);

    EXPECT_TRUE(approxCompare(0.0f, actualSensorCell.signal->channels[0]));
}

TEST_F(SensorTests_New, scanForSameMutants_found)
{
    DataDescription data;
    data.addCells(
        {CellDescription()
             .setId(1)
             .setMutationId(6)
             .setPos({100.0f, 100.0f})
             .setCellTypeData(SensorDescription().setRestrictToMutants(SensorRestrictToMutants_RestrictToSameMutants)),
        CellDescription().setId(2).setMutationId(6).setPos({101.0f, 100.0f}),
    });
    data.addConnection(1, 2);

    data.add(DescriptionEditService::get().createRect(DescriptionEditService::CreateRectParameters()
                                                          .center({10.0f, 100.0f})
                                                          .width(16)
                                                          .height(16)
                                                          .cellDistance(0.5f)
                                                          .mutationId(6)
                                                          .cellType(BaseDescription())));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualSensorCell = getCell(actualData, 1);

    EXPECT_TRUE(approxCompare(1.0f, actualSensorCell.signal->channels[0]));
}

TEST_F(SensorTests_New, scanForSameMutants_notFound)
{
    auto const MutantId = 6;
    for (int otherMutantId = 0; otherMutantId < 100; ++otherMutantId) {
        if (otherMutantId == MutantId) {
            continue;
        }
        DataDescription data;
        data.addCells(
            {CellDescription()
                 .setId(1)
                 .setMutationId(MutantId)
                 .setPos({100.0f, 100.0f})
                 .setCellTypeData(SensorDescription().setRestrictToMutants(SensorRestrictToMutants_RestrictToSameMutants)),
             CellDescription()
                 .setId(2)
                 .setMutationId(MutantId).setPos({101.0f, 100.0f}),
        });
        data.addConnection(1, 2);

        data.add(DescriptionEditService::get().createRect(DescriptionEditService::CreateRectParameters()
                                                              .center({10.0f, 100.0f})
                                                              .width(16)
                                                              .height(16)
                                                              .cellDistance(0.5f)
                                                              .mutationId(otherMutantId)
                                                              .cellType(BaseDescription())));

        _simulationFacade->clear();
        _simulationFacade->setCurrentTimestep(0ull);
        _simulationFacade->setSimulationData(data);
        _simulationFacade->calcTimesteps(1);

        auto actualData = _simulationFacade->getSimulationData();
        auto actualSensorCell = getCell(actualData, 1);

        EXPECT_TRUE(approxCompare(0.0f, actualSensorCell.signal->channels[0]));
    }
}

TEST_F(SensorTests_New, scanForStructures_found)
{
    DataDescription data;
    data.addCells(
        {CellDescription()
             .setId(1)
             .setPos({100.0f, 100.0f})
             .setCellTypeData(SensorDescription().setRestrictToMutants(SensorRestrictToMutants_RestrictToStructures)),
        CellDescription().setId(2).setPos({101.0f, 100.0f}),
    });
    data.addConnection(1, 2);

    data.add(DescriptionEditService::get().createRect(
        DescriptionEditService::CreateRectParameters().center({10.0f, 100.0f}).width(16).height(16).cellDistance(0.5f).cellType(StructureCellDescription())));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualSensorCell = getCell(actualData, 1);

    EXPECT_TRUE(approxCompare(1.0f, actualSensorCell.signal->channels[0]));
}

TEST_F(SensorTests_New, scanForStructures_notFound)
{
    DataDescription data;
    data.addCells(
        {CellDescription()
             .setId(1)
             .setPos({100.0f, 100.0f})
             .setCellTypeData(SensorDescription().setRestrictToMutants(SensorRestrictToMutants_RestrictToStructures)),
        CellDescription().setId(2).setPos({101.0f, 100.0f}),
    });
    data.addConnection(1, 2);

    data.add(DescriptionEditService::get().createRect(
        DescriptionEditService::CreateRectParameters().center({10.0f, 100.0f}).width(16).height(16).cellDistance(0.5f).cellType(BaseDescription())));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualSensorCell = getCell(actualData, 1);

    EXPECT_TRUE(approxCompare(0.0f, actualSensorCell.signal->channels[0]));
}

TEST_F(SensorTests_New, scanForFreeCells_found)
{
    DataDescription data;
    data.addCells(
        {CellDescription()
             .setId(1)
             .setPos({100.0f, 100.0f})
             .setCellTypeData(SensorDescription().setRestrictToMutants(SensorRestrictToMutants_RestrictToFreeCells)),
        CellDescription().setId(2).setPos({101.0f, 100.0f}),
    });
    data.addConnection(1, 2);

    data.add(DescriptionEditService::get().createRect(
        DescriptionEditService::CreateRectParameters().center({10.0f, 100.0f}).width(16).height(16).cellDistance(0.5f).cellType(FreeCellDescription())));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualSensorCell = getCell(actualData, 1);

    EXPECT_TRUE(approxCompare(1.0f, actualSensorCell.signal->channels[0]));
}

TEST_F(SensorTests_New, scanForFreeCells_notFound)
{
    DataDescription data;
    data.addCells(
        {CellDescription()
             .setId(1)
             .setPos({100.0f, 100.0f})
             .setCellTypeData(SensorDescription().setRestrictToMutants(SensorRestrictToMutants_RestrictToFreeCells)),
        CellDescription().setId(2).setPos({101.0f, 100.0f}),
    });
    data.addConnection(1, 2);

    data.add(DescriptionEditService::get().createRect(
        DescriptionEditService::CreateRectParameters().center({10.0f, 100.0f}).width(16).height(16).cellDistance(0.5f).cellType(StructureCellDescription())));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualSensorCell = getCell(actualData, 1);

    EXPECT_TRUE(approxCompare(0.0f, actualSensorCell.signal->channels[0]));
}

TEST_F(SensorTests_New, scanForLessComplexMutants_found)
{
    for (int otherGenomeComplexity = 0; otherGenomeComplexity < 500; ++otherGenomeComplexity) {
        DataDescription data;
        data.addCells(
            {CellDescription()
                 .setId(1)
                 .setPos({100.0f, 100.0f})
                 .setGenomeComplexity(1000.0f)
                 .setCellTypeData(SensorDescription().setRestrictToMutants(SensorRestrictToMutants_RestrictToLessComplexMutants)),
            CellDescription().setId(2).setPos({101.0f, 100.0f}),
        });
        data.addConnection(1, 2);

        data.add(DescriptionEditService::get().createRect(DescriptionEditService::CreateRectParameters()
                                                              .center({10.0f, 100.0f})
                                                              .cellType(BaseDescription())
                                                              .width(16)
                                                              .height(16)
                                                              .cellDistance(0.5f)
                                                              .genomeComplexity(toFloat(otherGenomeComplexity))));

        _simulationFacade->clear();
        _simulationFacade->setCurrentTimestep(0ull);
        _simulationFacade->setSimulationData(data);
        _simulationFacade->calcTimesteps(1);

        auto actualData = _simulationFacade->getSimulationData();
        auto actualSensorCell = getCell(actualData, 1);

        EXPECT_TRUE(approxCompare(1.0f, actualSensorCell.signal->channels[0]));
    }
}

TEST_F(SensorTests_New, scanForLessComplexMutants_notFound_otherMoreComplex)
{
    for (int otherGenomeComplexity = 1000; otherGenomeComplexity < 2001; ++otherGenomeComplexity) {
        DataDescription data;
        data.addCells(
            {CellDescription()
                 .setId(1)
                 .setPos({100.0f, 100.0f})
                 .setGenomeComplexity(1000.0f)
                 .setCellTypeData(SensorDescription().setRestrictToMutants(SensorRestrictToMutants_RestrictToLessComplexMutants)),
            CellDescription().setId(2).setMutationId(5).setPos({101.0f, 100.0f}),
        });
        data.addConnection(1, 2);

        data.add(DescriptionEditService::get().createRect(DescriptionEditService::CreateRectParameters()
                                                              .center({10.0f, 100.0f})
                                                              .width(16)
                                                              .height(16)
                                                              .cellDistance(0.5f)
                                                              .genomeComplexity(toFloat(otherGenomeComplexity))
                                                              .cellType(BaseDescription())));

        _simulationFacade->clear();
        _simulationFacade->setCurrentTimestep(0ull);
        _simulationFacade->setSimulationData(data);
        _simulationFacade->calcTimesteps(1);

        auto actualData = _simulationFacade->getSimulationData();
        auto actualSensorCell = getCell(actualData, 1);

        EXPECT_TRUE(approxCompare(0.0f, actualSensorCell.signal->channels[0]));
    }
}

TEST_F(SensorTests_New, scanForLessComplexMutants_notFound_structure)
{
    DataDescription data;
    data.addCells(
        {CellDescription()
             .setId(1)
             .setMutationId(100)
             .setPos({100.0f, 100.0f})
             .setGenomeComplexity(1000.0f)
             .setCellTypeData(SensorDescription().setRestrictToMutants(SensorRestrictToMutants_RestrictToLessComplexMutants)),
         CellDescription().setId(2).setMutationId(100).setPos({101.0f, 100.0f}).setCellTypeData(OscillatorDescription()).setSignal({1, 0, 0, 0, 0, 0, 0, 0})});
    data.addConnection(1, 2);

    data.add(DescriptionEditService::get().createRect(DescriptionEditService::CreateRectParameters()
                                                          .center({10.0f, 100.0f})
                                                          .width(16)
                                                          .height(16)
                                                          .cellDistance(0.5f)
                                                          .cellType(StructureCellDescription())
                                                          .genomeComplexity(100.0f)));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualSensorCell = getCell(actualData, 1);

    EXPECT_TRUE(approxCompare(0.0f, actualSensorCell.signal->channels[0]));
}

TEST_F(SensorTests_New, scanForLessComplexMutants_notFound_freeCell)
{
    DataDescription data;
    data.addCells(
        {CellDescription()
             .setId(1)
             .setMutationId(100)
             .setGenomeComplexity(1000.0f)
             .setPos({100.0f, 100.0f})
             .setCellTypeData(SensorDescription().setRestrictToMutants(SensorRestrictToMutants_RestrictToLessComplexMutants)),
        CellDescription().setId(2).setMutationId(100).setPos({101.0f, 100.0f}),
    });
    data.addConnection(1, 2);

    data.add(DescriptionEditService::get().createRect(DescriptionEditService::CreateRectParameters()
                                                          .center({10.0f, 100.0f})
                                                          .width(16)
                                                          .height(16)
                                                          .cellDistance(0.5f)
                                                          .cellType(FreeCellDescription())
                                                          .genomeComplexity(100.0f)));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualSensorCell = getCell(actualData, 1);

    EXPECT_TRUE(approxCompare(0.0f, actualSensorCell.signal->channels[0]));
}

TEST_F(SensorTests_New, scanForMoreComplexMutants_found)
{
    for (int otherGenomeComplexity = 1000; otherGenomeComplexity < 2001; ++otherGenomeComplexity) {
        DataDescription data;
        data.addCells(
            {CellDescription()
                 .setId(1)
                 .setPos({100.0f, 100.0f})
                 .setGenomeComplexity(500.0f)
                 .setCellTypeData(SensorDescription().setRestrictToMutants(SensorRestrictToMutants_RestrictToMoreComplexMutants)),
            CellDescription().setId(2).setMutationId(5).setPos({101.0f, 100.0f}),
        });
        data.addConnection(1, 2);

        data.add(DescriptionEditService::get().createRect(DescriptionEditService::CreateRectParameters()
                                                              .center({10.0f, 100.0f})
                                                              .width(16)
                                                              .height(16)
                                                              .cellDistance(0.5f)
                                                              .genomeComplexity(toFloat(otherGenomeComplexity))
                                                              .cellType(BaseDescription())));

        _simulationFacade->clear();
        _simulationFacade->setCurrentTimestep(0ull);
        _simulationFacade->setSimulationData(data);
        _simulationFacade->calcTimesteps(1);

        auto actualData = _simulationFacade->getSimulationData();
        auto actualSensorCell = getCell(actualData, 1);

        EXPECT_TRUE(approxCompare(1.0f, actualSensorCell.signal->channels[0]));
    }
}

TEST_F(SensorTests_New, scanForMoreComplexMutants_notFound_otherLessComplex)
{
    for (int otherGenomeComplexity = 0; otherGenomeComplexity < 500; ++otherGenomeComplexity) {
        DataDescription data;
        data.addCells(
            {CellDescription()
                 .setId(1)
                 .setPos({100.0f, 100.0f})
                 .setGenomeComplexity(500.0f)
                 .setCellTypeData(SensorDescription().setRestrictToMutants(SensorRestrictToMutants_RestrictToMoreComplexMutants)),
            CellDescription().setId(2).setMutationId(5).setPos({101.0f, 100.0f}),
        });
        data.addConnection(1, 2);

        data.add(DescriptionEditService::get().createRect(DescriptionEditService::CreateRectParameters()
                                                              .center({10.0f, 100.0f})
                                                              .width(16)
                                                              .height(16)
                                                              .cellDistance(0.5f)
                                                              .genomeComplexity(toFloat(otherGenomeComplexity))
                                                              .cellType(BaseDescription())));

        _simulationFacade->clear();
        _simulationFacade->setCurrentTimestep(0ull);
        _simulationFacade->setSimulationData(data);
        _simulationFacade->calcTimesteps(1);

        auto actualData = _simulationFacade->getSimulationData();
        auto actualSensorCell = getCell(actualData, 1);

        EXPECT_TRUE(approxCompare(0.0f, actualSensorCell.signal->channels[0]));
    }
}

TEST_F(SensorTests_New, scanForMoreComplexMutants_notFound_structure)
{
    DataDescription data;
    data.addCells(
        {CellDescription()
             .setId(1)
             .setMutationId(100)
             .setGenomeComplexity(100.0f)
             .setPos({100.0f, 100.0f})
             .setCellTypeData(SensorDescription().setRestrictToMutants(SensorRestrictToMutants_RestrictToMoreComplexMutants)),
        CellDescription().setId(2).setMutationId(100).setPos({101.0f, 100.0f}),
    });
    data.addConnection(1, 2);

    data.add(DescriptionEditService::get().createRect(DescriptionEditService::CreateRectParameters()
                                                          .center({10.0f, 100.0f})
                                                          .width(16)
                                                          .height(16)
                                                          .cellDistance(0.5f)
                                                          .cellType(StructureCellDescription())
                                                          .genomeComplexity(1000.0f)));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualSensorCell = getCell(actualData, 1);

    EXPECT_TRUE(approxCompare(0.0f, actualSensorCell.signal->channels[0]));
}

TEST_F(SensorTests_New, scanForMoreComplexMutants_notFound_freeCell)
{
    _parameters.cellTypeAttackerSensorDetectionFactor[0] = 1.0f;
    _simulationFacade->setSimulationParameters(_parameters);
    DataDescription data;
    data.addCells(
        {CellDescription()
             .setId(1)
             .setMutationId(100)
             .setGenomeComplexity(100.0f)
             .setPos({100.0f, 100.0f})
             .setCellTypeData(SensorDescription().setRestrictToMutants(SensorRestrictToMutants_RestrictToMoreComplexMutants)),
        CellDescription().setId(2).setMutationId(100).setPos({101.0f, 100.0f}),
    });
    data.addConnection(1, 2);

    data.add(DescriptionEditService::get().createRect(DescriptionEditService::CreateRectParameters()
                                                          .center({10.0f, 100.0f})
                                                          .width(16)
                                                          .height(16)
                                                          .cellDistance(0.5f)
                                                          .cellType(FreeCellDescription())
                                                          .genomeComplexity(1000.0f)));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualSensorCell = getCell(actualData, 1);

    EXPECT_TRUE(approxCompare(0.0f, actualSensorCell.signal->channels[0]));
}

TEST_F(SensorTests_New, minRange_found)
{
    DataDescription data;
    data.addCells(
        {CellDescription().setId(1).setPos({100.0f, 100.0f}).setCellTypeData(SensorDescription().setMinRange(50)),
        CellDescription().setId(2).setPos({101.0f, 100.0f}),
    });
    data.addConnection(1, 2);

    data.add(DescriptionEditService::get().createRect(
        DescriptionEditService::CreateRectParameters().center({10.0f, 100.0f}).width(16).height(16).cellDistance(0.5f).cellType(BaseDescription())));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualSensorCell = getCell(actualData, 1);

    EXPECT_TRUE(approxCompare(1.0f, actualSensorCell.signal->channels[0]));
}

TEST_F(SensorTests_New, minRange_notFound)
{
    DataDescription data;
    data.addCells(
        {CellDescription().setId(1).setPos({100.0f, 100.0f}).setCellTypeData(SensorDescription().setMinRange(120)),
        CellDescription().setId(2).setPos({101.0f, 100.0f}),
    });
    data.addConnection(1, 2);

    data.add(DescriptionEditService::get().createRect(
        DescriptionEditService::CreateRectParameters().center({10.0f, 100.0f}).width(16).height(16).cellDistance(0.5f).cellType(BaseDescription())));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualSensorCell = getCell(actualData, 1);

    EXPECT_TRUE(approxCompare(0.0f, actualSensorCell.signal->channels[0]));
}

TEST_F(SensorTests_New, maxRange_found)
{
    DataDescription data;
    data.addCells(
        {CellDescription().setId(1).setPos({100.0f, 100.0f}).setCellTypeData(SensorDescription().setMaxRange(120)),
        CellDescription().setId(2).setPos({101.0f, 100.0f}),
    });
    data.addConnection(1, 2);

    data.add(DescriptionEditService::get().createRect(
        DescriptionEditService::CreateRectParameters().center({10.0f, 100.0f}).width(16).height(16).cellDistance(0.5f).cellType(BaseDescription())));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualSensorCell = getCell(actualData, 1);

    EXPECT_TRUE(approxCompare(1.0f, actualSensorCell.signal->channels[0]));
}

TEST_F(SensorTests_New, maxRange_notFound)
{
    DataDescription data;
    data.addCells(
        {CellDescription().setId(1).setPos({100.0f, 100.0f}).setCellTypeData(SensorDescription().setMaxRange(50)),
        CellDescription().setId(2).setPos({101.0f, 100.0f}),
    });
    data.addConnection(1, 2);

    data.add(DescriptionEditService::get().createRect(
        DescriptionEditService::CreateRectParameters().center({10.0f, 100.0f}).width(16).height(16).cellDistance(0.5f).cellType(BaseDescription())));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualSensorCell = getCell(actualData, 1);

    EXPECT_TRUE(approxCompare(0.0f, actualSensorCell.signal->channels[0]));
}
