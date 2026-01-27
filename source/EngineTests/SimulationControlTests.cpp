#include <gtest/gtest.h>

#include <EngineInterface/SimulationFacade.h>
#include <EngineInterface/SimulationParameters.h>

#include <EngineImpl/SimulationFacadeImpl.h>

class SimulationControlTests : public ::testing::Test
{
public:
    SimulationControlTests() = default;
    virtual ~SimulationControlTests() = default;

protected:
    void SetUp() override
    {
        _simulationFacade = std::make_shared<_SimulationFacadeImpl>();
    }

    void TearDown() override
    {
        if (_simulationFacade) {
            _simulationFacade->closeSimulation();
            _simulationFacade.reset();
        }
    }

    SimulationFacade _simulationFacade;
    SimulationParameters _parameters;
};

TEST_F(SimulationControlTests, newSimulation_preservesNonZeroTimestep)
{
    uint64_t expectedTimestep = 12345;
    IntVector2D worldSize{100, 100};

    _simulationFacade->newSimulation(expectedTimestep, worldSize, _parameters);

    auto actualTimestep = _simulationFacade->getCurrentTimestep();
    EXPECT_EQ(expectedTimestep, actualTimestep);
}

TEST_F(SimulationControlTests, newSimulation_preservesLargeTimestep)
{
    uint64_t expectedTimestep = 9876543210ULL;
    IntVector2D worldSize{100, 100};

    _simulationFacade->newSimulation(expectedTimestep, worldSize, _parameters);

    auto actualTimestep = _simulationFacade->getCurrentTimestep();
    EXPECT_EQ(expectedTimestep, actualTimestep);
}

TEST_F(SimulationControlTests, newSimulation_zeroTimestep)
{
    uint64_t expectedTimestep = 0;
    IntVector2D worldSize{100, 100};

    _simulationFacade->newSimulation(expectedTimestep, worldSize, _parameters);

    auto actualTimestep = _simulationFacade->getCurrentTimestep();
    EXPECT_EQ(expectedTimestep, actualTimestep);
}
