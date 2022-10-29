#include "IntegrationTestFramework.h"

#include <boost/range/combine.hpp>

#include "Base/Math.h"
#include "EngineInterface/SimulationParameters.h"
#include "EngineImpl/SimulationControllerImpl.h"

IntegrationTestFramework::IntegrationTestFramework(IntVector2D const& universeSize)
{
    _simController = std::make_shared<_SimulationControllerImpl>();
    Settings settings;
    settings.generalSettings.worldSizeX = universeSize.x;
    settings.generalSettings.worldSizeY = universeSize.y;
    settings.simulationParameters.spotValues.radiationFactor = 0;
    _simController->newSimulation(0, settings);
    _parameters = _simController->getSimulationParameters();
}

IntegrationTestFramework::~IntegrationTestFramework()
{
    _simController->closeSimulation();
}

std::unordered_map<uint64_t, CellDescription> IntegrationTestFramework::getCellById(DataDescription const& data) const
{
    std::unordered_map<uint64_t, CellDescription> result;
    for(auto const& cell : data.cells) {
        result.emplace(cell.id, cell);
    }
    return result;
}

CellDescription IntegrationTestFramework::getCell(DataDescription const& data, uint64_t id) const
{
    for (auto const& cell : data.cells) {
        if (cell.id == id) {
            return cell;
        }
    }
    THROW_NOT_IMPLEMENTED();
}

CellDescription IntegrationTestFramework::getOtherCell(DataDescription const& data, uint64_t id) const
{
    for (auto const& cell : data.cells) {
        if (cell.id != id) {
            return cell;
        }
    }
    THROW_NOT_IMPLEMENTED();
}

CellDescription IntegrationTestFramework::getOtherCell(DataDescription const& data, std::set<uint64_t> ids) const
{
    for (auto const& cell : data.cells) {
        if (!ids.contains(cell.id)) {
            return cell;
        }
    }
    THROW_NOT_IMPLEMENTED();
}

void IntegrationTestFramework::expectApproxEqual(float expected, float actual) const
{
    EXPECT_TRUE(std::abs(expected - actual) < 0.01f);
}

void IntegrationTestFramework::expectApproxEqual(RealVector2D const& expected, RealVector2D const& actual) const
{
    EXPECT_TRUE(Math::length(expected - actual) < 0.01f);
}

void IntegrationTestFramework::expectApproxEqual(std::vector<float> const& expected, std::vector<float> const& actual) const
{
    CHECK(expected.size() == actual.size())
    for (auto const& [expectedElement, actualElement] : boost::combine(expected, actual)) {
        expectApproxEqual(expectedElement, actualElement);
    }
}

bool IntegrationTestFramework::compare(DataDescription left, DataDescription right) const
{
    std::sort(left.cells.begin(), left.cells.end(), [](auto const& left, auto const& right) { return left.id < right.id; });
    std::sort(right.cells.begin(), right.cells.end(), [](auto const& left, auto const& right) { return left.id < right.id; });
    std::sort(left.particles.begin(), left.particles.end(), [](auto const& left, auto const& right) { return left.id < right.id; });
    std::sort(right.particles.begin(), right.particles.end(), [](auto const& left, auto const& right) { return left.id < right.id; });
    return left == right;
}

bool IntegrationTestFramework::compare(CellDescription left, CellDescription right) const
{
    left.id = 0;
    right.id = 0;
    return left == right;
}

bool IntegrationTestFramework::compare(ParticleDescription left, ParticleDescription right) const
{
    left.id = 0;
    right.id = 0;
    return left == right;
}
