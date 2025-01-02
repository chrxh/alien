#include "IntegrationTestFramework.h"

#include <boost/range/combine.hpp>

#include "Base/Math.h"
#include "EngineInterface/SimulationParameters.h"
#include "EngineImpl/SimulationFacadeImpl.h"

IntegrationTestFramework::IntegrationTestFramework(std::optional<SimulationParameters> const& parameters_, IntVector2D const& universeSize)
{
    _simulationFacade = std::make_shared<_SimulationFacadeImpl>();
    GeneralSettings generalSettings{universeSize.x, universeSize.y};
    SimulationParameters parameters;
    if (parameters_) {
        parameters = *parameters_;
    } else {
        for (int i = 0; i < MAX_COLORS; ++i) {
            parameters.baseValues.radiationCellAgeStrength[i] = 0;
        }
    }
    _simulationFacade->newSimulation(0, generalSettings, parameters);
    _parameters = _simulationFacade->getSimulationParameters();
}

IntegrationTestFramework::~IntegrationTestFramework()
{
    _simulationFacade->closeSimulation();
}

double IntegrationTestFramework::getEnergy(DataDescription const& data) const
{
    double result = 0;
    for (auto const& cell : data.cells) {
        result += cell.energy;
    }
    for (auto const& particle : data.particles) {
        result += particle.energy;
    }
    return result;
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

std::unordered_map<std::pair<uint64_t, uint64_t>, ConnectionDescription> IntegrationTestFramework::getConnectionById(DataDescription const& data) const
{
    std::unordered_map<std::pair<uint64_t, uint64_t>, ConnectionDescription> result;
    for (auto const& cell : data.cells) {
        for (auto const& connection : cell.connections) {
            result.emplace(std::make_pair(cell.id, connection.cellId), connection);
        }
    }
    return result;
}

ConnectionDescription IntegrationTestFramework::getConnection(DataDescription const& data, uint64_t id, uint64_t otherId) const
{
    auto cell = getCell(data, id);
    for (auto const& connection : cell.connections) {
        if (connection.cellId == otherId) {
            return connection;
        }
    }
    THROW_NOT_IMPLEMENTED();
}

bool IntegrationTestFramework::hasConnection(DataDescription const& data, uint64_t id, uint64_t otherId) const
{
    auto cell = getCell(data, id);
    for (auto const& connection : cell.connections) {
        if (connection.cellId == otherId) {
            return true;
        }
    }
    return false;
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

bool IntegrationTestFramework::approxCompare(double expected, double actual, float precision) const
{
    return approxCompare(toFloat(expected), toFloat(actual));
}

bool IntegrationTestFramework::approxCompare(float expected, float actual, float precision) const
{
    auto absNorm = std::abs(expected) + std::abs(actual);
    if (absNorm < precision) {
        return true;
    }
    return std::abs(expected - actual) / absNorm < precision;
}

bool IntegrationTestFramework::approxCompare(RealVector2D const& expected, RealVector2D const& actual) const
{
    return approxCompare(expected.x, expected.x) && approxCompare(expected.y, expected.y);
}

bool IntegrationTestFramework::approxCompare(std::vector<float> const& expected, std::vector<float> const& actual) const
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
