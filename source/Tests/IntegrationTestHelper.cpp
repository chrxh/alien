#include <QEventLoop>

#include "EngineInterface/Physics.h"
#include "EngineInterface/SimulationAccess.h"
#include "EngineInterface/SimulationController.h"
#include "EngineInterface/SimulationContext.h"


#include "IntegrationTestHelper.h"

DataDescription IntegrationTestHelper::getContent(SimulationAccess* access, IntRect const& rect)
{
    bool contentReady = false;
    QEventLoop pause;
    auto connection = access->connect(access, &SimulationAccess::dataReadyToRetrieve, [&]() {
        contentReady = true;
        pause.quit();
    });
    ResolveDescription rd;
    rd.resolveCellLinks = true;
    access->requireData(rect, rd);
    if (!contentReady) {
        pause.exec();
    }
    QObject::disconnect(connection);
    return access->retrieveData();
}

void IntegrationTestHelper::updateData(SimulationAccess* access, SimulationContext* context, 
    DataChangeDescription const& data)
{
    QEventLoop pause;
    bool finished = false;
    auto connection1 = access->connect(access, &SimulationAccess::dataUpdated, [&]() {
        finished = true;
        pause.quit();
    });
    auto connection2 =
        context->connect(context, &SimulationContext::errorThrown, [&](QString what) {
            throw std::exception(what.toStdString().c_str());
        });
    access->updateData(data);
    while (!finished) {
        pause.exec();
    }
    QObject::disconnect(connection1);
    QObject::disconnect(connection2);
}

void IntegrationTestHelper::runSimulation(int timesteps, SimulationController* controller)
{
    QEventLoop pause;
    auto context = controller->getContext();
    for (int t = 0; t < timesteps; ++t) {
        bool finished = false;
        auto connection1 = controller->connect(controller, &SimulationController::nextTimestepCalculated, [&]() {
            finished = true;
            pause.quit();
        });
        auto connection2 = context->connect(context, &SimulationContext::errorThrown, [&](QString what) {
            throw std::exception(what.toStdString().c_str());
        });
        controller->calculateSingleTimestep();
        if (!finished) {
            pause.exec();
        }
        QObject::disconnect(connection1);
        QObject::disconnect(connection2);
    }
}

unordered_map<uint64_t, ParticleDescription> IntegrationTestHelper::getParticleByParticleId(DataDescription const& data)
{
    unordered_map<uint64_t, ParticleDescription> result;
    if (data.particles) {
        std::transform(
            data.particles->begin(),
            data.particles->end(),
            std::inserter(result, result.begin()),
            [](ParticleDescription const& desc) { return std::make_pair(desc.id, desc); });
    }
    return result;
}

unordered_map<uint64_t, CellDescription> IntegrationTestHelper::getCellByCellId(DataDescription const& data)
{
    unordered_map<uint64_t, CellDescription> result;
    if (data.clusters) {
        for (ClusterDescription const& cluster : *data.clusters) {
            for (CellDescription const& cell : *cluster.cells) {
                result.insert_or_assign(cell.id, cell);
            }
        }
    }
    return result;
}

unordered_map<uint64_t, ClusterDescription> IntegrationTestHelper::getClusterByCellId(DataDescription const& data)
{
    unordered_map<uint64_t, ClusterDescription> result;
    if (data.clusters) {
        for (ClusterDescription const& cluster : *data.clusters) {
            for (CellDescription const& cell : *cluster.cells) {
                result.insert_or_assign(cell.id, cluster);
            }
        }
    }
    return result;
}

unordered_map<uint64_t, ClusterDescription> IntegrationTestHelper::getClusterByClusterId(DataDescription const& data)
{
    unordered_map<uint64_t, ClusterDescription> result;
    if (data.clusters) {
        for (ClusterDescription const& cluster : *data.clusters) {
            result.insert_or_assign(cluster.id, cluster);
        }
    }
    return result;
}
