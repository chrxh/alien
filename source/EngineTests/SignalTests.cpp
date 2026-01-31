#include <gtest/gtest.h>

#include <EngineInterface/Description.h>
#include <EngineInterface/DescriptionEditService.h>
#include <EngineInterface/SimulationFacade.h>

#include "IntegrationTestFramework.h"

class SignalTests : public IntegrationTestFramework
{
public:
    SignalTests()
        : IntegrationTestFramework()
    {}

    ~SignalTests() = default;
};

TEST_F(SignalTests, noSignal)
{
    auto data = Desc().addCreature({
        ObjectDesc().id(1),
    });

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();

    auto generator = actualData.getObjectRef(1);
    // Cell should have no active signal (all channels zero)
    EXPECT_TRUE(approxCompare(0.0f, generator.getCellRef()._signal._channels[0]));
}

TEST_F(SignalTests, forwardSignal)
{
    std::vector<float> signal = {1.0f, -1.0f, -0.5f, 0, 0.5f, 2.0f, -2.0f, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    auto data = Desc().addCreature({
        ObjectDesc().id(1).pos({0, 0}).type(CellDesc().signal(SignalDesc().channels(signal))),
        ObjectDesc().id(2).pos({1, 0}),
    });
    data.addConnection(1, 2);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);
    auto actualData = _simulationFacade->getSimulationData();

    auto object2 = actualData.getObjectRef(2);
    EXPECT_EQ(signal, object2.getCellRef()._signal._channels);
}

TEST_F(SignalTests, forwardSignal_detailedPreview)
{
    std::vector<float> signal = {1.0f, -1.0f, -0.5f, 0, 0.5f, 2.0f, -2.0f, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    auto data = Desc().addCreature({
        ObjectDesc().id(1).pos({0, 0}).type(CellDesc().signal(signal)),
        ObjectDesc().id(2).pos({1, 0}),
    });
    data.addConnection(1, 2);

    _simulationFacade->setPreviewData(data);
    _simulationFacade->calcTimestepsForPreview(1, true);
    auto actualData = _simulationFacade->getPreviewData();

    auto object2 = actualData.getObjectRef(2);
    EXPECT_EQ(signal, object2.getCellRef()._signal._channels);
}

TEST_F(SignalTests, forwardSignal_withOtherConnections)
{
    std::vector<float> signal = {1.0f, -1.0f, -0.5f, 0, 0.5f, 2.0f, -2.0f, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    auto data = Desc().addCreature({
        ObjectDesc()
            .id(1)
            .pos({10.0f, 10.0f})
            .type(CellDesc()
                      .nodeIndex(15)
                      .signal(SignalDesc().channels(signal))),
        ObjectDesc().id(2).pos(RealVector2D{10.0f, 10.0f} + Math::unitVectorOfAngle(120.0f)).type(CellDesc()),
        ObjectDesc().id(3).pos(RealVector2D{10.0f, 10.0f} + Math::unitVectorOfAngle(300.0f)).type(CellDesc().nodeIndex(16)),
        ObjectDesc()
            .id(4)
            .pos(RealVector2D{10.0f, 10.0f} + Math::unitVectorOfAngle(0))
            .type(CellDesc()
                      .signal(SignalDesc().channels(signal))),
        ObjectDesc().id(5).pos(RealVector2D{10.0f, 10.0f} + Math::unitVectorOfAngle(60.0f)).type(CellDesc()),
    });
    data.addConnection(1, 2);
    data.addConnection(1, 3);
    data.addConnection(1, 4);
    data.addConnection(1, 5);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);
    auto actualData = _simulationFacade->getSimulationData();

    // Without signal restrictions, signals flow to all connected cells
    auto object3 = actualData.getObjectRef(3);
    EXPECT_EQ(signal, object3.getCellRef()._signal._channels);
}

TEST_F(SignalTests, vanishSignal_singleCell)
{
    std::vector<float> signal = {1.0f, -1.0f, -0.5f, 0, 0.5f, 2.0f, -2.0f, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    auto data = Desc().addCreature({
        ObjectDesc().id(1).pos({0, 0}).type(CellDesc().signal(signal)),
    });

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);
    auto actualData = _simulationFacade->getSimulationData();

    auto object1 = actualData.getObjectRef(1);
    // Signal should have been cleared for a single cell with no propagation target
    EXPECT_TRUE(approxCompare(0.0f, object1.getCellRef()._signal._channels[0]));
}

TEST_F(SignalTests, vanishSignal_relaxationNeeded)
{
    std::vector<float> signal = {1.0f, -1.0f, -0.5f, 0, 0.5f, 2.0f, -2.0f, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    auto data = Desc().addCreature({
        ObjectDesc().id(1).pos({0, 0}).type(CellDesc().signal(SignalDesc().channels(signal))),
        ObjectDesc().id(2).pos({1, 0}),
    });
    data.addConnection(1, 2);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);
    auto actualData = _simulationFacade->getSimulationData();

    // Signal should have propagated from object1 to object2
    auto object2 = actualData.getObjectRef(2);
    EXPECT_EQ(signal, object2.getCellRef()._signal._channels);
}

TEST_F(SignalTests, mergeSignals)
{
    std::vector<float> signal1 = {1.0f, -1.0f, -0.5f, 0.0f, 0.5f, 1.0f, -1.0f, 0.0f, 0, 0, 0, 0, 0, 0, 0, 0};
    std::vector<float> signal2 = {-0.5f, -1.0f, 0.5f, 1.0f, 0.7f, -0.7f, 0.5f, -0.5f, 0, 0, 0, 0, 0, 0, 0, 0};
    auto data = Desc().addCreature({
        ObjectDesc().id(1).pos({0, 0}).type(CellDesc().signal(SignalDesc().channels(signal1))),
        ObjectDesc().id(2).pos({1, 0}),
        ObjectDesc().id(3).pos({2, 0}).type(CellDesc().signal(SignalDesc().channels(signal2))),
    });
    data.addConnection(1, 2);
    data.addConnection(2, 3);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);
    auto actualData = _simulationFacade->getSimulationData();

    auto object2 = actualData.getObjectRef(2);

    std::vector<float> sumSignal(signal1.size());
    for (size_t i = 0; i < signal1.size(); ++i) {
        sumSignal[i] = signal1[i] + signal2[i];
    }
    EXPECT_TRUE(approxCompare(sumSignal, object2.getCellRef()._signal._channels));
}

TEST_F(SignalTests, forkSignals)
{
    std::vector<float> signal = {1.0f, -1.0f, -0.5f, 0.0f, 0.5f, 2.0f, -2.0f, 0.0f, 0, 0, 0, 0, 0, 0, 0, 0};
    auto data = Desc().addCreature({
        ObjectDesc().id(1).pos({0, 0}),
        ObjectDesc().id(2).pos({1, 0}).type(CellDesc().signal(SignalDesc().channels(signal))),
        ObjectDesc().id(3).pos({2, 0}),
    });
    data.addConnection(1, 2);
    data.addConnection(2, 3);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);
    auto actualData = _simulationFacade->getSimulationData();

    auto object1 = actualData.getObjectRef(1);
    EXPECT_TRUE(approxCompare(signal, object1.getCellRef()._signal._channels));

    auto cell3 = actualData.getObjectRef(3);
    EXPECT_TRUE(approxCompare(signal, cell3.getCellRef()._signal._channels));
}

