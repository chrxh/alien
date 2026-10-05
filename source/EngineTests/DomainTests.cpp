#include <gtest/gtest.h>

#include <Base/GlobalSettings.h>
#include <Base/Math.h>

#include <Data/Descs.h>

#include <EngineInterface/SimulationFacade.h>

#include "IntegrationTestFramework.h"

// The world is split into two domains on the first GPU, their strips meet at x = 500 and at the world border
class DomainTests : public IntegrationTestFramework
{
public:
    DomainTests()
        : IntegrationTestFramework(enableDomains())
    {}

    ~DomainTests() override
    {
        GlobalSettings::get().setNumDomains(1);
        GlobalSettings::get().setDomainDevices({});
    }

protected:
    // A world size used by no other test, so that the decomposed simulation is not reused by them
    static IntVector2D enableDomains()
    {
        GlobalSettings::get().setNumDomains(2);
        GlobalSettings::get().setDomainDevices({0, 0});
        return {1000, 300};
    }

    void expectConsistentDomains()
    {
        auto errors = _simulationFacade->testOnly_getDomainConsistencyErrors();
        EXPECT_TRUE(errors.empty()) << errors.size() << " errors, the first one: " << (errors.empty() ? "" : errors.front());
    }

    // The sensor faces upwards, so that a target to the right appears at +90 degrees
    static ContentDesc createSensor(RealVector2D const& pos, SensorModeDesc const& mode)
    {
        auto result = ContentDesc().addCreature(
            {
                ObjectDesc().id(1).pos(pos).type(CellDesc().frontAngle(0.0f).cellType(SensorDesc().autoTrigger(true).mode(mode))),
                ObjectDesc().id(2).pos({pos.x, pos.y - 1.0f}),
            },
            CreatureDesc().id(1));
        result.addConnection(1, 2);
        return result;
    }

    static void addVerticalWall(ContentDesc& data, float x, float startY, float endY)
    {
        for (auto y = startY; y <= endY; y += 1.0f) {
            data._objects.emplace_back(ObjectDesc().pos({x, y}).type(SolidDesc()));
        }
    }

    static void addEnergyBlock(ContentDesc& data, RealVector2D const& upperLeft, RealVector2D const& lowerRight)
    {
        for (auto y = upperLeft.y; y <= lowerRight.y; y += 2.0f) {
            for (auto x = upperLeft.x; x <= lowerRight.x; x += 1.0f) {
                data._energies.emplace_back(EnergyDesc().pos({x, y}).energy(10.0f));
            }
        }
    }

    NeuralActivityDesc calcSensorActivity(ContentDesc const& data)
    {
        _simulationFacade->setSimulationData(data);
        _simulationFacade->calcTimesteps(TIMESTEPS_PER_CELL_FUNCTION);
        return _simulationFacade->getSimulationData().getObjectRef(1).getCellRef()._neuralActivity;
    }
};

TEST_F(DomainTests, freeCellCrossesStripBorder)
{
    auto data = ContentDesc().objects({ObjectDesc().id(1).pos({490.0f, 150.0f}).vel({1.0f, 0.0f}).type(FreeCellDesc())});

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(40);

    auto actualData = _simulationFacade->getSimulationData();
    ASSERT_EQ(1, actualData._objects.size());
    EXPECT_EQ(1, actualData._objects.front()._id);
    EXPECT_GT(actualData._objects.front()._pos.x, 510.0f);
    expectConsistentDomains();
}

TEST_F(DomainTests, energyParticleCrossesWorldBorder)
{
    auto data = ContentDesc().energies({EnergyDesc().id(1).pos({995.0f, 150.0f}).vel({0.5f, 0.0f}).energy(10.0f)});

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(40);

    auto actualData = _simulationFacade->getSimulationData();
    ASSERT_EQ(1, actualData._energies.size());
    EXPECT_EQ(1, actualData._energies.front()._id);
    EXPECT_GT(actualData._energies.front()._pos.x, 5.0f);
    EXPECT_LT(actualData._energies.front()._pos.x, 500.0f);
    EXPECT_TRUE(approxCompare(10.0f, actualData._energies.front()._energy));
    expectConsistentDomains();
}

TEST_F(DomainTests, connectionAcrossStripBorderIsKept)
{
    auto data = ContentDesc().objects({
        ObjectDesc().id(1).pos({499.5f, 150.0f}).type(FreeCellDesc()),
        ObjectDesc().id(2).pos({500.5f, 150.0f}).type(FreeCellDesc()),
    });
    data.addConnection(1, 2);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(30);

    auto actualData = _simulationFacade->getSimulationData();
    ASSERT_EQ(2, actualData._objects.size());
    auto const& object1 = actualData.getObjectRef(1);
    auto const& object2 = actualData.getObjectRef(2);
    EXPECT_TRUE(object1.isConnectedTo(2));
    EXPECT_TRUE(object2.isConnectedTo(1));
    EXPECT_TRUE(approxCompare(1.0f, Math::length(object2._pos - object1._pos), 0.1f));
    expectConsistentDomains();
}

TEST_F(DomainTests, creatureCrossesStripBorder)
{
    auto data = ContentDesc().addCreature(
        {
            ObjectDesc().id(1).pos({490.0f, 150.0f}).vel({1.0f, 0.0f}),
            ObjectDesc().id(2).pos({491.0f, 150.0f}).vel({1.0f, 0.0f}),
        },
        CreatureDesc().id(1));
    data.addConnection(1, 2);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(40);

    auto actualData = _simulationFacade->getSimulationData();
    ASSERT_EQ(2, actualData._objects.size());
    ASSERT_EQ(1, actualData._creatures.size());
    auto const& cell1 = actualData.getObjectRef(1);
    auto const& cell2 = actualData.getObjectRef(2);
    EXPECT_GT(cell1._pos.x, 510.0f);
    EXPECT_TRUE(cell1.isConnectedTo(2));
    EXPECT_EQ(cell1.getCellRef()._creatureId, cell2.getCellRef()._creatureId);
    expectConsistentDomains();
}

TEST_F(DomainTests, energyIsPreserved)
{
    ContentDesc data;
    for (int i = 0; i < 20; ++i) {
        auto y = 50.0f + toFloat(i) * 10.0f;
        data._objects.emplace_back(ObjectDesc().pos({480.0f, y}).vel({0.5f, 0.0f}).type(FreeCellDesc()));
        data._objects.emplace_back(ObjectDesc().pos({520.0f, y + 5.0f}).vel({-0.5f, 0.0f}).type(FreeCellDesc()));
        data._energies.emplace_back(EnergyDesc().pos({10.0f, y}).vel({-0.5f, 0.0f}).energy(10.0f));
    }
    auto origEnergy = getEnergy(data);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(100);

    auto actualData = _simulationFacade->getSimulationData();
    EXPECT_EQ(40, actualData._objects.size());
    EXPECT_TRUE(approxCompare(origEnergy, getEnergy(actualData)));
    expectConsistentDomains();
}

TEST_F(DomainTests, sensorDetectsSolidInOtherStrip)
{
    auto data = createSensor({490.0f, 150.0f}, DetectSolidDesc());
    addVerticalWall(data, 600.0f, 110.0f, 190.0f);

    auto signals = calcSensorActivity(data)._signals;

    EXPECT_TRUE(approxCompare(1.0f, signals[Channels::SensorFoundResult]));
    auto expectedDistance = 1.0f - (110.0f - 0.75f) / 256.0f;
    EXPECT_TRUE(approxCompare(expectedDistance, signals[Channels::SensorDistance], 0.005f)) << "Actual distance " << signals[Channels::SensorDistance];
    EXPECT_TRUE(approxCompare(0.5f, signals[Channels::SensorAngle], 0.05f)) << "Actual angle " << signals[Channels::SensorAngle];
}

TEST_F(DomainTests, sensorDetectsSolidAcrossWorldBorder)
{
    auto data = createSensor({40.0f, 150.0f}, DetectSolidDesc());
    addVerticalWall(data, 930.0f, 110.0f, 190.0f);

    auto signals = calcSensorActivity(data)._signals;

    EXPECT_TRUE(approxCompare(1.0f, signals[Channels::SensorFoundResult]));
    auto expectedDistance = 1.0f - (110.0f - 0.75f) / 256.0f;
    EXPECT_TRUE(approxCompare(expectedDistance, signals[Channels::SensorDistance], 0.005f)) << "Actual distance " << signals[Channels::SensorDistance];
    EXPECT_TRUE(approxCompare(-0.5f, signals[Channels::SensorAngle], 0.05f)) << "Actual angle " << signals[Channels::SensorAngle];
}

TEST_F(DomainTests, sensorDetectsEnergyInOtherStrip)
{
    auto data = createSensor({490.0f, 150.0f}, DetectEnergyDesc().minDensity(0.05f));
    addEnergyBlock(data, {600.0f, 120.0f}, {607.0f, 180.0f});

    auto signals = calcSensorActivity(data)._signals;

    // The samples on a ray are 8 units apart
    EXPECT_TRUE(approxCompare(1.0f, signals[Channels::SensorFoundResult]));
    EXPECT_GE(signals[Channels::SensorDistance], 1.0f - 119.0f / 256.0f);
    EXPECT_LE(signals[Channels::SensorDistance], 1.0f - 109.0f / 256.0f);
    EXPECT_TRUE(approxCompare(0.5f, signals[Channels::SensorAngle], 0.05f)) << "Actual angle " << signals[Channels::SensorAngle];
}

TEST_F(DomainTests, sensorRayBlockedInOwnStrip)
{
    auto data = createSensor({460.0f, 150.0f}, DetectEnergyDesc().minDensity(0.05f));
    addVerticalWall(data, 480.0f, 50.0f, 250.0f);
    addEnergyBlock(data, {600.0f, 120.0f}, {607.0f, 180.0f});

    auto signals = calcSensorActivity(data)._signals;

    EXPECT_TRUE(approxCompare(0.0f, signals[Channels::SensorFoundResult]));
}

TEST_F(DomainTests, sensorRayBlockedInOtherStrip)
{
    auto data = createSensor({490.0f, 150.0f}, DetectEnergyDesc().minDensity(0.05f));
    addVerticalWall(data, 560.0f, 50.0f, 250.0f);
    addEnergyBlock(data, {600.0f, 120.0f}, {607.0f, 180.0f});

    auto signals = calcSensorActivity(data)._signals;

    EXPECT_TRUE(approxCompare(0.0f, signals[Channels::SensorFoundResult]));
}
