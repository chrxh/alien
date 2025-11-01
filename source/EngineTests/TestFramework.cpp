#include "TestFramework.h"

#include <boost/range/combine.hpp>

#include "Base/Math.h"
#include "EngineInterface/SimulationParameters.h"

#include "EngineTestData/DescriptionTestDataFactory.h"

#include "EngineImpl/SimulationFacadeImpl.h"

TestFramework::TestFramework(IntVector2D const& universeSize)
{
    _simulationFacade = std::make_shared<_SimulationFacadeImpl>();
    SimulationParameters parameters;
    for (int i = 0; i < MAX_COLORS; ++i) {
        parameters.radiationType1_strength.baseValue[i] = 0;
    }
    _simulationFacade->newSimulation(0, universeSize, parameters);
    _parameters = _simulationFacade->getSimulationParameters();
}

TestFramework::~TestFramework()
{
    _simulationFacade->closeSimulation();
}

double TestFramework::getEnergy(Description const& data) const
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

bool TestFramework::approxCompare(double expected, double actual, float precision) const
{
    return approxCompare(toFloat(expected), toFloat(actual));
}

bool TestFramework::approxCompare(float expected, float actual, float precision) const
{
    auto absNorm = std::abs(expected) + std::abs(actual);
    if (absNorm < precision) {
        return true;
    }
    return std::abs(expected - actual) / absNorm < precision;
}

bool TestFramework::approxCompare(RealVector2D const& expected, RealVector2D const& actual) const
{
    return approxCompare(expected.x, actual.x) && approxCompare(expected.y, actual.y);
}

bool TestFramework::approxCompare(std::vector<float> const& expected, std::vector<float> const& actual) const
{
    if (expected.size() != actual.size()) {
        return false;
    }
    for (auto const& [expectedElement, actualElement] : boost::combine(expected, actual)) {
        if (!approxCompare(expectedElement, actualElement)) {
            return false;
        }
    }
    return true;
}

bool TestFramework::approxCompareAngles(float expected, float actual, float precision) const
{
    return approxCompare(Math::getNormalizedAngle(expected - actual, -180.0f), 0.0f, precision);
}

bool TestFramework::compare(Description left, Description right) const
{
    return DescriptionTestDataFactory::get().compare(left, right);
}

bool TestFramework::compare(CellDescription left, CellDescription right) const
{
    return DescriptionTestDataFactory::get().compare(left, right);
}

bool TestFramework::compare(ParticleDescription left, ParticleDescription right) const
{
    return DescriptionTestDataFactory::get().compare(left, right);
}
