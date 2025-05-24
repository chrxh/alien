#include "IntegrationTestFramework.h"

#include <boost/range/combine.hpp>

#include "Base/Math.h"
#include "EngineInterface/SimulationParameters.h"
#include "EngineImpl/SimulationFacadeImpl.h"

IntegrationTestFramework::IntegrationTestFramework(std::optional<SimulationParameters> const& parameters_, IntVector2D const& universeSize)
{
    _simulationFacade = std::make_shared<_SimulationFacadeImpl>();
    SimulationParameters parameters;
    if (parameters_) {
        parameters = *parameters_;
    } else {
        for (int i = 0; i < MAX_COLORS; ++i) {
            parameters.radiationType1_strength.baseValue[i] = 0;
        }
    }
    _simulationFacade->newSimulation(0, universeSize, parameters);
    _parameters = _simulationFacade->getSimulationParameters();
}

IntegrationTestFramework::~IntegrationTestFramework()
{
    _simulationFacade->closeSimulation();
}

double IntegrationTestFramework::getEnergy(CollectionDescription const& data) const
{
    double result = 0;
    for (auto const& cell : data._cells) {
        result += cell._energy;
    }
    for (auto const& particle : data._particles) {
        result += particle._energy;
    }
    return result;
}

std::unordered_map<uint64_t, CellDescription> IntegrationTestFramework::getCellById(CollectionDescription const& data) const
{
    std::unordered_map<uint64_t, CellDescription> result;
    for(auto const& cell : data._cells) {
        result.emplace(cell._id, cell);
    }
    return result;
}

CellDescription IntegrationTestFramework::getCell(CollectionDescription const& data, uint64_t id) const
{
    for (auto const& cell : data._cells) {
        if (cell._id == id) {
            return cell;
        }
    }
    THROW_NOT_IMPLEMENTED();
}

ConnectionDescription IntegrationTestFramework::getConnection(CollectionDescription const& data, uint64_t id, uint64_t otherId) const
{
    auto cell = getCell(data, id);
    for (auto const& connection : cell._connections) {
        if (connection._cellId == otherId) {
            return connection;
        }
    }
    THROW_NOT_IMPLEMENTED();
}

ConnectionDescription IntegrationTestFramework::getConnection(CellDescription const& cell1, CellDescription const& cell2) const
{
    for (auto const& connection : cell1._connections) {
        if (connection._cellId == cell2._id) {
            return connection;
        }
    }
    THROW_NOT_IMPLEMENTED();
}

bool IntegrationTestFramework::hasConnection(CollectionDescription const& data, uint64_t id, uint64_t otherId) const
{
    auto cell = getCell(data, id);
    for (auto const& connection : cell._connections) {
        if (connection._cellId == otherId) {
            return true;
        }
    }
    return false;
}

CellDescription IntegrationTestFramework::getOtherCell(CollectionDescription const& data, uint64_t id) const
{
    for (auto const& cell : data._cells) {
        if (cell._id != id) {
            return cell;
        }
    }
    THROW_NOT_IMPLEMENTED();
}

CellDescription IntegrationTestFramework::getOtherCell(CollectionDescription const& data, std::set<uint64_t> ids) const
{
    for (auto const& cell : data._cells) {
        if (!ids.contains(cell._id)) {
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

bool IntegrationTestFramework::compare(CollectionDescription left, CollectionDescription right) const
{
    std::sort(left._cells.begin(), left._cells.end(), [](auto const& left, auto const& right) { return left._id < right._id; });
    std::sort(right._cells.begin(), right._cells.end(), [](auto const& left, auto const& right) { return left._id < right._id; });
    std::sort(left._particles.begin(), left._particles.end(), [](auto const& left, auto const& right) { return left._id < right._id; });
    std::sort(right._particles.begin(), right._particles.end(), [](auto const& left, auto const& right) { return left._id < right._id; });

    // Equalize genome ids since they are generated during GPU -> CPU transfer
    if (left._cells.size() != right._cells.size()) {
        return false;
    }
    std::unordered_map<uint64_t, uint64_t> leftByRightGenomeId;
    for (auto const& [leftCell, rightCell] : boost::combine(left._cells, right._cells)) {
        if (leftCell._genomeId.has_value() != rightCell._genomeId.has_value()) {
            return false;
        }
        if (leftCell._genomeId.has_value()) {
            leftByRightGenomeId.insert_or_assign(rightCell._genomeId.value(), leftCell._genomeId.value());
        }
    }
    for (auto& genome : right._genomes) {
        if (!leftByRightGenomeId.contains(genome._id)) {
            return false;
        }
        genome._id = leftByRightGenomeId.at(genome._id);
    }
    for (auto& cells : right._cells) {
        if (cells._genomeId.has_value()) {
            if (!leftByRightGenomeId.contains(cells._genomeId.value())) {
                return false;
            }
            cells._genomeId = leftByRightGenomeId.at(cells._genomeId.value());
        }
    }

    std::sort(left._genomes.begin(), left._genomes.end(), [](auto const& left, auto const& right) { return left._id < right._id; });
    std::sort(right._genomes.begin(), right._genomes.end(), [](auto const& left, auto const& right) { return left._id < right._id; });

    return left == right;
}

bool IntegrationTestFramework::compare(CellDescription left, CellDescription right) const
{
    left._id = 0;
    right._id = 0;
    return left == right;
}

bool IntegrationTestFramework::compare(ParticleDescription left, ParticleDescription right) const
{
    left._id = 0;
    right._id = 0;
    return left == right;
}
