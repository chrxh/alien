#include "IntegrationTestFramework.h"

#include <boost/range/combine.hpp>

#include <Base/Math.h>

#include <EngineInterface/SimulationFacade.h>
#include <EngineInterface/SimulationParameters.h>

#include <EngineTestData/DescriptionTestDataFactory.h>

#include <EngineImpl/SimulationFacadeImpl.h>

// Static member initialization
std::map<std::string, std::shared_ptr<IntegrationTestFramework::TestSuiteContext>> IntegrationTestFramework::_contextMap;

// TestSuiteContext destructor implementation
IntegrationTestFramework::TestSuiteContext::~TestSuiteContext()
{
    if (simulationFacade) {
        simulationFacade->closeSimulation();
    }
}

// Test listener to clean up contexts after each test suite
class IntegrationTestListener : public ::testing::EmptyTestEventListener
{
    void OnTestSuiteEnd(const ::testing::TestSuite& test_suite) override
    {
        // Clean up the context for this test suite
        IntegrationTestFramework::_contextMap.erase(test_suite.name());
    }
};

// Register the listener (done once at program startup)
namespace
{
    struct ListenerRegistrar
    {
        ListenerRegistrar()
        {
            auto& listeners = ::testing::UnitTest::GetInstance()->listeners();
            listeners.Append(new IntegrationTestListener());
        }
    };
    static ListenerRegistrar registrar;
}

IntegrationTestFramework::IntegrationTestFramework(IntVector2D const& universeSize)
    : _universeSize(universeSize)
{
    // Get the test suite name (e.g., "CellConnectionTests")
    auto testInfo = ::testing::UnitTest::GetInstance()->current_test_info();
    std::string testSuiteName = testInfo ? testInfo->test_suite_name() : "Unknown";
    
    // Get or create context for this test suite
    auto& context = _contextMap[testSuiteName];
    if (!context) {
        context = std::make_shared<TestSuiteContext>();
        context->simulationFacade = std::make_shared<_SimulationFacadeImpl>();
        
        SimulationParameters parameters;
        for (int i = 0; i < MAX_COLORS; ++i) {
            parameters.radiationType1_strength.baseValue[i] = 0;
        }
        context->simulationFacade->newSimulation(0, _universeSize, parameters);
    }
    
    context->refCount++;
    _context = context;
    _simulationFacade = context->simulationFacade;
}

IntegrationTestFramework::~IntegrationTestFramework()
{
    if (_context) {
        _context->refCount--;
    }
}

void IntegrationTestFramework::SetUp()
{
    _simulationFacade->clear();
    _simulationFacade->setCurrentTimestep(0);
    _parameters = _simulationFacade->getSimulationParameters();
}

double IntegrationTestFramework::getEnergy(Description const& data) const
{
    double result = 0;
    for (auto const& cell : data._cells) {
        result += cell._energy;
    }
    for (auto const& creature : data._creatures) {
        for (auto const& cell : creature._cells) {
            result += cell._energy;
        }
    }
    for (auto const& particle : data._particles) {
        result += particle._energy;
    }
    return result;
}

bool IntegrationTestFramework::compare(Description left, Description right) const
{
    return DescriptionTestDataFactory::get().compare(left, right);
}

bool IntegrationTestFramework::compare(CellDescription left, CellDescription right) const
{
    return DescriptionTestDataFactory::get().compare(left, right);
}

bool IntegrationTestFramework::compare(ParticleDescription left, ParticleDescription right) const
{
    return DescriptionTestDataFactory::get().compare(left, right);
}
