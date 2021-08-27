#include "DataRepository.h"

#include <QMatrix4x4>

#include "Base/DebugMacros.h"
#include "Base/NumberGenerator.h"
#include "EngineInterface/DescriptionHelper.h"
#include "EngineInterface/SimulationAccess.h"
#include "EngineInterface/SimulationContext.h"
#include "EngineInterface/SimulationParameters.h"
#include "EngineInterface/SpaceProperties.h"
#include "Notifier.h"

void DataRepository::init(
    Notifier* notifier,
    SimulationAccess* access,
    DescriptionHelper* connector,
    SimulationContext* context)
{
    delete _access;  //to reduce memory usage delete old object first
    _access = nullptr;
    SET_CHILD(_access, access);
    _descHelper = connector;
    _notifier = notifier;
    _numberGenerator = context->getNumberGenerator();
    _parameters = context->getSimulationParameters();
    _universeSize = context->getSpaceProperties()->getSize();
    _unchangedData.clear();
    _data.clear();
    _selectedCellIds.clear();
    _selectedClusterIds.clear();
    _selectedParticleIds.clear();
    _selectedTokenIndex.reset();

    for (auto const& connection : _connections) {
        disconnect(connection);
    }
    _connections.push_back(connect(
        _access,
        &SimulationAccess::dataReadyToRetrieve,
        this,
        &DataRepository::dataFromSimulationAvailable,
        Qt::QueuedConnection));
    _connections.push_back(connect(_access, &SimulationAccess::imageReady, this, &DataRepository::imageReady));
    _connections.push_back(
        connect(_notifier, &Notifier::notifyDataRepositoryChanged, this, &DataRepository::sendDataChangesToSimulation));
}

DataDescription& DataRepository::getDataRef()
{
    return _data;
}

CellDescription& DataRepository::getCellDescRef(uint64_t cellId)
{
    TRY;
    ClusterDescription& clusterDesc = getClusterDescRef(cellId);
    int cellIndex = _navi.cellIndicesByCellIds.at(cellId);
    return clusterDesc.cells->at(cellIndex);
    CATCH;
}

ClusterDescription& DataRepository::getClusterDescRef(uint64_t cellId)
{
    TRY;
    int clusterIndex = _navi.clusterIndicesByCellIds.at(cellId);
    return _data.clusters->at(clusterIndex);
    CATCH;
}

ClusterDescription const& DataRepository::getClusterDescRef(uint64_t cellId) const
{
    TRY;
    int clusterIndex = _navi.clusterIndicesByCellIds.at(cellId);
    return _data.clusters->at(clusterIndex);
    CATCH;
}

ParticleDescription& DataRepository::getParticleDescRef(uint64_t particleId)
{
    TRY;
    int particleIndex = _navi.particleIndicesByParticleIds.at(particleId);
    return _data.particles->at(particleIndex);
    CATCH;
}

ParticleDescription const& DataRepository::getParticleDescRef(uint64_t particleId) const
{
    TRY;
    int particleIndex = _navi.particleIndicesByParticleIds.at(particleId);
    return _data.particles->at(particleIndex);
    CATCH;
}

void DataRepository::setSelectedTokenIndex(boost::optional<uint> const& value)
{
    _selectedTokenIndex = value;
}

boost::optional<uint> DataRepository::getSelectedTokenIndex() const
{
    return _selectedTokenIndex;
}

void DataRepository::addAndSelectCell(QVector2D const& posDelta)
{
    TRY;
    QVector2D pos = _rect.center().toQVector2D() + posDelta;
    int memorySize = _parameters.cellFunctionComputerCellMemorySize;
    auto desc = ClusterDescription()
                    .addCell(CellDescription()
                                 .setEnergy(_parameters.cellFunctionConstructorOffspringCellEnergy)
                                 .setMaxConnections(_parameters.cellMaxBonds)
                                 .setPos(pos)
                                 .setConnectingCells({})
                                 .setMetadata(CellMetadata())
                                 .setFlagTokenBlocked(false)
                                 .setTokenBranchNumber(0)
                                 .setCellFeature(CellFeatureDescription()
                                                     .setType(Enums::CellFunction::COMPUTER)
                                                     .setVolatileData(QByteArray(memorySize, 0))));
    _descHelper->makeValid(desc);
    _data.addCluster(desc);
    _selectedCellIds = {desc.cells->front().id};
    _selectedClusterIds = {desc.id};
    _selectedParticleIds = {};
    _navi.update(_data);
    CATCH;
}

void DataRepository::addAndSelectParticle(QVector2D const& posDelta)
{
    TRY;
    QVector2D pos = _rect.center().toQVector2D() + posDelta;
    auto desc = ParticleDescription().setPos(pos).setVel({}).setEnergy(_parameters.cellMinEnergy / 2.0);
    _descHelper->makeValid(desc);
    _data.addParticle(desc);
    _selectedCellIds = {};
    _selectedClusterIds = {};
    _selectedParticleIds = {desc.id};
    _navi.update(_data);
    CATCH;
}

void DataRepository::addAndSelectData(DataDescription data, QVector2D const& posDelta, Reconnect reconnect)
{
    TRY;
    if (Reconnect::Yes == reconnect) {
        std::unordered_set<uint64_t> cellIds;
        if (auto const& clusters = data.clusters)
            for (auto const& cluster : *clusters) {
                if (auto const& cells = cluster.cells) {
                    for (auto const& cell : *cells) {
                        cellIds.insert(cell.id);
                    }
                }
            }
        _descHelper->reconnect(data, data, cellIds);
    }

    QVector2D centerOfData = data.calcCenter();
    QVector2D targetCenter = _rect.center().toQVector2D() + posDelta;
    QVector2D delta = targetCenter - centerOfData;
    data.shift(delta);

    _selectedCellIds = {};
    _selectedClusterIds = {};
    _selectedParticleIds = {};
    if (data.clusters) {
        for (auto& cluster : *data.clusters) {
            cluster.id = 0;
            _descHelper->makeValid(cluster);
            _data.addCluster(cluster);
            _selectedClusterIds.insert(cluster.id);
            if (cluster.cells) {
                std::transform(
                    cluster.cells->begin(),
                    cluster.cells->end(),
                    std::inserter(_selectedCellIds, _selectedCellIds.begin()),
                    [](auto const& cell) { return cell.id; });
            }
        }
    }
    if (data.particles) {
        for (auto& particle : *data.particles) {
            particle.id = 0;
            _descHelper->makeValid(particle);
            _data.addParticle(particle);
            _selectedParticleIds.insert(particle.id);
        }
    }
    _navi.update(_data);
    CATCH;
}

namespace
{
    QVector2D calcCenter(
        int numCluster,
        int numParticles,
        std::function<ClusterDescription&(int)> clusterResolver,
        std::function<ParticleDescription&(int)> particleResolver)
    {
        QVector2D result;
        int numEntities = 0;
        for (int i = 0; i < numCluster; ++i) {
            auto const& cluster = clusterResolver(i);
            if (!cluster.cells) {
                continue;
            }
            for (auto const& cell : *cluster.cells) {
                result += *cell.pos;
                ++numEntities;
            }
        }
        for (int i = 0; i < numParticles; ++i) {
            auto const& particle = particleResolver(i);
            result += *particle.pos;
            ++numEntities;
        }
        if (numEntities > 0) {
            result /= numEntities;
        }
        return result;
    }

    void rotate(
        double angle,
        int numCluster,
        int numParticles,
        std::function<ClusterDescription&(int)> clusterResolver,
        std::function<ParticleDescription&(int)> particleResolver)
    {
        QVector3D center = calcCenter(numCluster, numParticles, clusterResolver, particleResolver);

        QMatrix4x4 transform;
        transform.setToIdentity();
        transform.translate(center);
        transform.rotate(angle, 0.0, 0.0, 1.0);
        transform.translate(-center);

        for (int i = 0; i < numCluster; ++i) {
            auto& cluster = clusterResolver(i);
            if (!cluster.cells) {
                continue;
            }
            for (auto& cell : *cluster.cells) {
                *cell.pos = transform.map(QVector3D(*cell.pos)).toVector2D();
            }
        }
        for (int i = 0; i < numParticles; ++i) {
            auto& particle = particleResolver(i);
            *particle.pos = transform.map(QVector3D(*particle.pos)).toVector2D();
        }
    }
}

void DataRepository::addDataAtFixedPosition(vector<DataAndAngle> dataAndAngles)
{
    TRY;
    for (auto& dataAndAngle : dataAndAngles) {
        auto& data = dataAndAngle.data;
        auto& rotationAngel = dataAndAngle.angle;
        if (rotationAngel) {
            int numClusters = data.clusters.is_initialized() ? data.clusters->size() : 0;
            int numParticle = data.particles.is_initialized() ? data.particles->size() : 0;
            auto clusterResolver = [&data](int index) -> ClusterDescription& { return data.clusters->at(index); };
            auto particleResolver = [&data](int index) -> ParticleDescription& { return data.particles->at(index); };
            rotate(*dataAndAngle.angle, numClusters, numParticle, clusterResolver, particleResolver);
        }

        if (data.clusters) {
            for (auto& cluster : *data.clusters) {
                cluster.id = 0;
                _descHelper->makeValid(cluster);
                _data.addCluster(cluster);
            }
        }
        if (data.particles) {
            for (auto& particle : *data.particles) {
                particle.id = 0;
                _descHelper->makeValid(particle);
                _data.addParticle(particle);
            }
        }
    }
    _navi.update(_data);
    CATCH;
}

void DataRepository::addRandomParticles(double totalEnergy, double maxEnergyPerParticle)
{
    TRY;
    DataDescription data;
    double remainingEnergy = totalEnergy;
    while (remainingEnergy > FLOATINGPOINT_MEDIUM_PRECISION) {
        double particleEnergy = _numberGenerator->getRandomReal(maxEnergyPerParticle / 100.0, maxEnergyPerParticle);
        particleEnergy = std::min(particleEnergy, remainingEnergy);
        data.addParticle(
            ParticleDescription()
                .setPos(QVector2D(
                    _numberGenerator->getRandomReal(0.0, _universeSize.x),
                    _numberGenerator->getRandomReal(0.0, _universeSize.y)))
                .setVel(QVector2D(
                    _numberGenerator->getRandomReal() * 2.0 - 1.0, _numberGenerator->getRandomReal() * 2.0 - 1.0))
                .setEnergy(particleEnergy));
        remainingEnergy -= particleEnergy;
    }

    addDataAtFixedPosition({{data, boost::optional<double>()}});
    CATCH;
}

namespace
{
    void correctConnections(vector<CellDescription>& cells)
    {
        unordered_set<uint64_t> cellSet;
        std::transform(
            cells.begin(), cells.end(), std::inserter(cellSet, cellSet.begin()), [](CellDescription const& cell) {
                return cell.id;
            });

        for (auto& cell : cells) {
            if (cell.connections) {
                list<ConnectionDescription> newConnections;
                for (auto const& connection : *cell.connections) {
                    if (cellSet.find(connection.cellId) != cellSet.end()) {
                        newConnections.push_back(connection);
                    }
                }
                cell.connections = newConnections;
            }
        }
    }
}


void DataRepository::deleteSelection()
{
    TRY;
    if (_data.clusters) {
        unordered_set<uint64_t> modifiedClusterIds;
        vector<ClusterDescription> newClusters;
        for (auto const& cluster : *_data.clusters) {
            if (_selectedClusterIds.find(cluster.id) == _selectedClusterIds.end()) {
                newClusters.push_back(cluster);
            } else if (cluster.cells) {
                vector<CellDescription> newCells;
                for (auto const& cell : *cluster.cells) {
                    if (_selectedCellIds.find(cell.id) == _selectedCellIds.end()) {
                        newCells.push_back(cell);
                    }
                }
                if (!newCells.empty()) {
                    correctConnections(newCells);
                    ClusterDescription newCluster = cluster;
                    newCluster.cells = newCells;
                    newClusters.push_back(newCluster);
                    modifiedClusterIds.insert(cluster.id);
                }
            }
        }
        _data.clusters = newClusters;
        if (!modifiedClusterIds.empty()) {
            _descHelper->recluster(_data, modifiedClusterIds);
        }
    }
    if (_data.particles) {
        vector<ParticleDescription> newParticles;
        for (auto const& particle : *_data.particles) {
            if (_selectedParticleIds.find(particle.id) == _selectedParticleIds.end()) {
                newParticles.push_back(particle);
            }
        }
        _data.particles = newParticles;
    }
    _selectedCellIds = {};
    _selectedClusterIds = {};
    _selectedParticleIds = {};
    _navi.update(_data);
    CATCH;
}

void DataRepository::deleteExtendedSelection()
{
    TRY;
    if (_data.clusters) {
        vector<ClusterDescription> newClusters;
        for (auto const& cluster : *_data.clusters) {
            if (_selectedClusterIds.find(cluster.id) == _selectedClusterIds.end()) {
                newClusters.push_back(cluster);
            }
        }
        _data.clusters = newClusters;
    }
    if (_data.particles) {
        vector<ParticleDescription> newParticles;
        for (auto const& particle : *_data.particles) {
            if (_selectedParticleIds.find(particle.id) == _selectedParticleIds.end()) {
                newParticles.push_back(particle);
            }
        }
        _data.particles = newParticles;
    }
    _selectedCellIds = {};
    _selectedClusterIds = {};
    _selectedParticleIds = {};
    _navi.update(_data);
    CATCH;
}

void DataRepository::addToken()
{
    addToken(TokenDescription()
                 .setEnergy(_parameters.cellFunctionConstructorOffspringTokenEnergy)
                 .setData(QByteArray(_parameters.tokenMemorySize, 0)));
}

void DataRepository::addToken(TokenDescription const& token)
{
    TRY;
    CHECK(_selectedCellIds.size() == 1);
    auto& cell = getCellDescRef(*_selectedCellIds.begin());

    int numToken = cell.tokens ? cell.tokens->size() : 0;
    if (numToken < _parameters.cellMaxToken) {
        uint pos = _selectedTokenIndex ? *_selectedTokenIndex : numToken;
        cell.addToken(pos, token);
    }
    CATCH;
}

void DataRepository::deleteToken()
{
    TRY;
    CHECK(_selectedCellIds.size() == 1);
    CHECK(_selectedTokenIndex);

    auto& cell = getCellDescRef(*_selectedCellIds.begin());
    cell.delToken(*_selectedTokenIndex);
    CATCH;
}

bool DataRepository::isCellPresent(uint64_t cellId)
{
    return _navi.cellIds.find(cellId) != _navi.cellIds.end();
}

bool DataRepository::isParticlePresent(uint64_t particleId)
{
    return _navi.particleIds.find(particleId) != _navi.particleIds.end();
}

void DataRepository::dataFromSimulationAvailable()
{
    TRY;
    updateInternals(_access->retrieveData());

    Q_EMIT _notifier->notifyDataRepositoryChanged(
        {Receiver::DataEditor, Receiver::VisualEditor, Receiver::ActionController}, UpdateDescription::All);
    CATCH;
}

void DataRepository::sendDataChangesToSimulation(set<Receiver> const& targets)
{
    TRY;
    if (targets.find(Receiver::Simulation) == targets.end()) {
        return;
    }
    DataChangeDescription delta(_unchangedData, _data);
    _access->updateData(delta);
    _unchangedData = _data;
    CATCH;
}

void DataRepository::setSelection(list<uint64_t> const& cellIds, list<uint64_t> const& particleIds)
{
    TRY;
    _selectedCellIds.clear();
    for (uint64_t particleId : cellIds) {
        if (_navi.cellIds.find(particleId) != _navi.cellIds.end()) {
            _selectedCellIds.insert(particleId);
        }
    }

    _selectedParticleIds.clear();
    for (uint64_t particleId : particleIds) {
        if (_navi.particleIds.find(particleId) != _navi.particleIds.end()) {
            _selectedParticleIds.insert(particleId);
        }
    }

    _selectedClusterIds.clear();
    for (uint64_t particleId : cellIds) {
        auto clusterIdByCellIdIter = _navi.clusterIdsByCellIds.find(particleId);
        if (clusterIdByCellIdIter != _navi.clusterIdsByCellIds.end()) {
            _selectedClusterIds.insert(clusterIdByCellIdIter->second);
        }
    }
    CATCH;
}

void DataRepository::updateData(DataDescription const& data)
{
    TRY;
    if (data.clusters) {
        for (auto const& cluster : *data.clusters) {
            updateCluster(cluster);
        }
    }
    if (data.particles) {
        for (auto const& particle : *data.particles) {
            updateParticle(particle);
        }
    }
    CATCH;
}

bool DataRepository::isInSelection(list<uint64_t> const& ids) const
{
    for (uint64_t id : ids) {
        if (!isInSelection(id)) {
            return false;
        }
    }
    return true;
}

bool DataRepository::isInSelection(uint64_t id) const
{
    return (
        _selectedCellIds.find(id) != _selectedCellIds.end()
        || _selectedParticleIds.find(id) != _selectedParticleIds.end());
}

bool DataRepository::isInExtendedSelection(uint64_t id) const
{
    auto clusterIdByCellIdIter = _navi.clusterIdsByCellIds.find(id);
    if (clusterIdByCellIdIter != _navi.clusterIdsByCellIds.end()) {
        uint64_t clusterId = clusterIdByCellIdIter->second;
        return (
            _selectedClusterIds.find(clusterId) != _selectedClusterIds.end()
            || _selectedParticleIds.find(id) != _selectedParticleIds.end());
    }
    return false;
}

bool DataRepository::areEntitiesSelected() const
{
    return !_selectedCellIds.empty() || !_selectedParticleIds.empty();
}

namespace
{
    template <typename T>
    unordered_set<T> calcDifference(unordered_set<T> const& set1, unordered_set<T> const& set2)
    {
        set<T> orderedSet1(set1.begin(), set1.end());
        set<T> orderedSet2(set2.begin(), set2.end());

        set<T> result;
        std::set_difference(
            orderedSet1.begin(), orderedSet1.end(), set2.begin(), set2.end(), std::inserter(result, result.begin()));
        return unordered_set<T>(result.begin(), result.end());
    }
}

unordered_set<uint64_t> DataRepository::getSelectedCellIds() const
{
    return _selectedCellIds;
}

unordered_set<uint64_t> DataRepository::getSelectedParticleIds() const
{
    return _selectedParticleIds;
}

DataDescription DataRepository::getExtendedSelection() const
{
    TRY;
    DataDescription result;
    for (uint64_t clusterId : _selectedClusterIds) {
        int clusterIndex = _navi.clusterIndicesByClusterIds.at(clusterId);
        result.addCluster(_data.clusters->at(clusterIndex));
    }
    for (uint64_t particleId : _selectedParticleIds) {
        result.addParticle(getParticleDescRef(particleId));
    }
    return result;
    CATCH;
}

void DataRepository::moveSelection(QVector2D const& delta)
{
    TRY;
    for (uint64_t cellId : _selectedCellIds) {
        if (isCellPresent(cellId)) {
            int clusterIndex = _navi.clusterIndicesByCellIds.at(cellId);
            int cellIndex = _navi.cellIndicesByCellIds.at(cellId);
            CellDescription& cellDesc = getCellDescRef(cellId);
            cellDesc.pos = *cellDesc.pos + delta;
        }
    }

    for (uint64_t particleId : _selectedParticleIds) {
        if (isParticlePresent(particleId)) {
            ParticleDescription& particleDesc = getParticleDescRef(particleId);
            particleDesc.pos = *particleDesc.pos + delta;
        }
    }
    CATCH;
}

void DataRepository::moveExtendedSelection(QVector2D const& delta)
{
    TRY;
    list<uint64_t> extSelectedCellIds;
    for (auto clusterIdByCellId : _navi.clusterIdsByCellIds) {
        uint64_t cellId = clusterIdByCellId.first;
        uint64_t clusterId = clusterIdByCellId.second;
        if (_selectedClusterIds.find(clusterId) != _selectedClusterIds.end()) {
            extSelectedCellIds.push_back(cellId);
        }
    }

    for (uint64_t cellId : extSelectedCellIds) {
        if (isCellPresent(cellId)) {
            int clusterIndex = _navi.clusterIndicesByCellIds.at(cellId);
            int cellIndex = _navi.cellIndicesByCellIds.at(cellId);
            CellDescription& cellDesc = getCellDescRef(cellId);
            cellDesc.pos = *cellDesc.pos + delta;
        }
    }

    for (uint64_t particleId : _selectedParticleIds) {
        if (isParticlePresent(particleId)) {
            ParticleDescription& particleDesc = getParticleDescRef(particleId);
            particleDesc.pos = *particleDesc.pos + delta;
        }
    }
    CATCH;
}

void DataRepository::reconnectSelectedCells()
{
    TRY;
    _descHelper->reconnect(getDataRef(), _unchangedData, getSelectedCellIds());
    updateAfterCellReconnections();
    CATCH;
}

void DataRepository::rotateSelection(double angle)
{
    TRY;
    vector<uint64_t> selectedClusterIds(_selectedClusterIds.begin(), _selectedClusterIds.end());
    vector<uint64_t> selectedParticleIds(_selectedParticleIds.begin(), _selectedParticleIds.end());
    auto clusterResolver = [&selectedClusterIds, this](int index) -> ClusterDescription& {
        return _data.clusters->at(_navi.clusterIndicesByClusterIds.at(selectedClusterIds.at(index)));
    };
    auto particleResolver = [&selectedParticleIds, this](int index) -> ParticleDescription& {
        return getParticleDescRef(selectedParticleIds.at(index));
    };
    rotate(angle, _selectedClusterIds.size(), _selectedParticleIds.size(), clusterResolver, particleResolver);
    CATCH;
}

void DataRepository::colorizeSelection(int colorCode)
{
    TRY;
    for (uint64_t cellId : _selectedCellIds) {
        if (isCellPresent(cellId)) {
            int cellIndex = _navi.cellIndicesByCellIds.at(cellId);
            CellDescription& cellDesc = getCellDescRef(cellId);
            cellDesc.metadata->color = colorCode;
        }
    }
    CATCH;
}

void DataRepository::updateCluster(ClusterDescription const& cluster)
{
    TRY;
    int clusterIndex = _navi.clusterIndicesByClusterIds.at(cluster.id);
    _data.clusters->at(clusterIndex) = cluster;

    _navi.update(_data);
    CATCH;
}

void DataRepository::updateParticle(ParticleDescription const& particle)
{
    TRY;
    int particleIndex = _navi.particleIndicesByParticleIds.at(particle.id);
    _data.particles->at(particleIndex) = particle;

    _navi.update(_data);
    CATCH;
}

void DataRepository::requireDataUpdateFromSimulation(IntRect const& rect)
{
    TRY;
    _rect = {
        {static_cast<float>(rect.p1.x), static_cast<float>(rect.p1.y)},
        {static_cast<float>(rect.p2.x), static_cast<float>(rect.p2.y)}};
    ResolveDescription resolveDesc;
    resolveDesc.resolveCellLinks = true;
    _access->requireData(rect, resolveDesc);
    CATCH;
}

void DataRepository::requirePixelImageFromSimulation(IntRect const& rect, QImagePtr const& target)
{
    TRY;
    _rect = {
        {static_cast<float>(rect.p1.x), static_cast<float>(rect.p1.y)},
        {static_cast<float>(rect.p2.x), static_cast<float>(rect.p2.y)}};
    _access->requirePixelImage(rect, target, _mutex);
    CATCH;
}

void DataRepository::requireVectorImageFromSimulation(
    RealRect const& rect,
    double zoom,
    ImageResource const& image,
    IntVector2D const& imageSize)
{
    _rect = rect;
    _access->requireVectorImage(rect, zoom, image, imageSize, _mutex);
}

std::mutex& DataRepository::getImageMutex()
{
    return _mutex;
}

void DataRepository::updateAfterCellReconnections()
{
    TRY;
    _navi.update(_data);

    _selectedClusterIds.clear();
    for (uint64_t selectedCellId : _selectedCellIds) {
        if (_navi.clusterIdsByCellIds.find(selectedCellId) != _navi.clusterIdsByCellIds.end()) {
            _selectedClusterIds.insert(_navi.clusterIdsByCellIds.at(selectedCellId));
        }
    }
    CATCH;
}

void DataRepository::updateInternals(DataDescription const& data)
{
    TRY;
    _data = data;
    _unchangedData = _data;

    _navi.update(data);

    unordered_set<uint64_t> newSelectedCells;
    std::copy_if(
        _selectedCellIds.begin(),
        _selectedCellIds.end(),
        std::inserter(newSelectedCells, newSelectedCells.begin()),
        [this](uint64_t cellId) { return _navi.cellIds.find(cellId) != _navi.cellIds.end(); });
    _selectedCellIds = newSelectedCells;

    unordered_set<uint64_t> newSelectedClusterIds;
    std::copy_if(
        _selectedClusterIds.begin(),
        _selectedClusterIds.end(),
        std::inserter(newSelectedClusterIds, newSelectedClusterIds.begin()),
        [this](uint64_t clusterId) {
            return _navi.clusterIndicesByClusterIds.find(clusterId) != _navi.clusterIndicesByClusterIds.end();
        });
    _selectedClusterIds = newSelectedClusterIds;

    unordered_set<uint64_t> newSelectedParticles;
    std::copy_if(
        _selectedParticleIds.begin(),
        _selectedParticleIds.end(),
        std::inserter(newSelectedParticles, newSelectedParticles.begin()),
        [this](uint64_t particleId) { return _navi.particleIds.find(particleId) != _navi.particleIds.end(); });
    _selectedParticleIds = newSelectedParticles;
    CATCH;
}
