#include "DescriptionConverterService.h"

#include <cmath>
#include <algorithm>
#include <mdspan>

#include <boost/range/adaptor/map.hpp>

#include "Base/NumberGenerator.h"
#include "Base/Exceptions.h"
#include "EngineInterface/Descriptions.h"
#include "EngineInterface/GenomeConstants.h"


namespace
{
    union BytesAsFloat
    {
        float f;
        uint8_t b[4];
    };

    void convert(DataTO const& dataTO, uint64_t sourceSize, uint64_t sourceIndex, std::vector<uint8_t>& target)
    {
        target.resize(sourceSize);
        for (int i = 0; i < sourceSize; ++i) {
            target[i] = dataTO.auxiliaryData[sourceIndex + i];
        }
    }

    NeuralNetworkDescription convertToNeuronDescription(DataTO const& dataTO, uint64_t sourceIndex)
    {
        NeuralNetworkDescription result;
        auto weights_span = std::mdspan(result._weights.data(), MAX_CHANNELS, MAX_CHANNELS);

        BytesAsFloat bytesAsFloat;
        int index = 0;
        for (int row = 0; row < MAX_CHANNELS; ++row) {
            for (int col = 0; col < MAX_CHANNELS; ++col) {
                for (int i = 0; i < 4; ++i) {
                    bytesAsFloat.b[i] = dataTO.auxiliaryData[sourceIndex + index];
                    ++index;
                }
                weights_span[row, col] = bytesAsFloat.f;
            }
        }
        for (int channel = 0; channel < MAX_CHANNELS; ++channel) {
            for (int i = 0; i < 4; ++i) {
                bytesAsFloat.b[i] = dataTO.auxiliaryData[sourceIndex + index];
                ++index;
            }
            result._biases[channel] = bytesAsFloat.f;
        }
        for (int channel = 0; channel < MAX_CHANNELS; ++channel) {
            result._activationFunctions[channel] = dataTO.auxiliaryData[sourceIndex + index];
            ++index;
        }

        return result;
    }

    template<typename Container, typename SizeType>
    void convert(DataTO const& dataTO, Container const& source, SizeType& targetSize, uint64_t& targetIndex)
    {
        targetSize = source.size();
        if (targetSize > 0) {
            targetIndex = *dataTO.numAuxiliaryData;
            uint64_t size = source.size();
            for (uint64_t i = 0; i < size; ++i) {
                dataTO.auxiliaryData[targetIndex + i] = source.at(i);
            }
            (*dataTO.numAuxiliaryData) += size;
        }
    }

    void convertToNeuronData(DataTO const& dataTO, NeuralNetworkDescription const& neuralNetDesc, uint64_t& targetIndex)
    {
        targetIndex = *dataTO.numAuxiliaryData;
        *dataTO.numAuxiliaryData += NeuralNetworkTO::DataSize;

        auto weights_span = std::mdspan(neuralNetDesc._weights.data(), MAX_CHANNELS, MAX_CHANNELS);

        BytesAsFloat bytesAsFloat;
        int bytePos = 0;
        for (int row = 0; row < MAX_CHANNELS; ++row) {
            for (int col = 0; col < MAX_CHANNELS; ++col) {
                bytesAsFloat.f = weights_span[row, col];
                for (int i = 0; i < 4; ++i) {
                    dataTO.auxiliaryData[targetIndex + bytePos] = bytesAsFloat.b[i];
                    ++bytePos;
                }
            }
        }
        for (int channel = 0; channel < MAX_CHANNELS; ++channel) {
            bytesAsFloat.f = neuralNetDesc._biases[channel];
            for (int i = 0; i < 4; ++i) {
                dataTO.auxiliaryData[targetIndex + bytePos] = bytesAsFloat.b[i];
                ++bytePos;
            }
        }
        for (int channel = 0; channel < MAX_CHANNELS; ++channel) {
            dataTO.auxiliaryData[targetIndex + bytePos] = neuralNetDesc._activationFunctions[channel];
            ++bytePos;
        }
    }
}

DescriptionConverterService::DescriptionConverterService(SimulationParameters const& parameters)
    : _parameters(parameters)
{}

ArraySizes DescriptionConverterService::getArraySizes(DataDescription const& data) const
{
    ArraySizes result;
    result.cellArraySize = data._cells.size();
    result.particleArraySize = data._particles.size();
    for (auto const& cell : data._cells) {
        addAdditionalDataSizeForCell(cell, result.auxiliaryDataSize);
    }
    return result;
}

ArraySizes DescriptionConverterService::getArraySizes(ClusteredDataDescription const& data) const
{
    ArraySizes result;
    for (auto const& cluster : data._clusters) {
        result.cellArraySize += cluster._cells.size();
        for (auto const& cell : cluster._cells) {
            addAdditionalDataSizeForCell(cell, result.auxiliaryDataSize);
        }
    }
    result.particleArraySize = data._particles.size();
    return result;
}

ClusteredDataDescription DescriptionConverterService::convertTOtoClusteredDataDescription(DataTO const& dataTO) const
{
	ClusteredDataDescription result;

    //cells
    std::vector<ClusterDescription> clusters;
    std::unordered_set<int> freeCellIndices;
    for (int i = 0; i < *dataTO.numCells; ++i) {
        freeCellIndices.insert(i);
    }
    std::unordered_map<int, int> cellTOIndexToCellDescIndex;
    std::unordered_map<int, int> cellTOIndexToClusterDescIndex;
    int clusterDescIndex = 0;
    while (!freeCellIndices.empty()) {
        auto freeCellIndex = *freeCellIndices.begin();
        auto createClusterData = scanAndCreateClusterDescription(dataTO, freeCellIndex, freeCellIndices);
        clusters.emplace_back(createClusterData.cluster);

        //update index maps
        cellTOIndexToCellDescIndex.insert(
            createClusterData.cellTOIndexToCellDescIndex.begin(), createClusterData.cellTOIndexToCellDescIndex.end());
        for (auto const& cellTOIndex : createClusterData.cellTOIndexToCellDescIndex | boost::adaptors::map_keys) {
            cellTOIndexToClusterDescIndex.emplace(cellTOIndex, clusterDescIndex);
        }
        ++clusterDescIndex;
    }
    result.addClusters(clusters);

    //particles
    std::vector<ParticleDescription> particles;
    for (int i = 0; i < *dataTO.numParticles; ++i) {
        ParticleTO const& particle = dataTO.particles[i];
        particles.emplace_back(ParticleDescription()
                                   .id(particle.id)
                                   .pos({particle.pos.x, particle.pos.y})
                                   .vel({particle.vel.x, particle.vel.y})
                                   .energy(particle.energy)
                                   .color(particle.color));
    }
    result.addParticles(particles);

    return result;
}

DataDescription DescriptionConverterService::convertTOtoDataDescription(DataTO const& dataTO) const
{
    DataDescription result;

    //cells
    std::vector<CellDescription> cells;
    for (int i = 0; i < *dataTO.numCells; ++i) {
        cells.emplace_back(createCellDescription(dataTO, i));
    }
    result.addCells(cells);

    //particles
    std::vector<ParticleDescription> particles;
    for (int i = 0; i < *dataTO.numParticles; ++i) {
        ParticleTO const& particle = dataTO.particles[i];
        particles.emplace_back(ParticleDescription()
                                   .id(particle.id)
                                   .pos({particle.pos.x, particle.pos.y})
                                   .vel({particle.vel.x, particle.vel.y})
                                   .energy(particle.energy)
                                   .color(particle.color));
    }
    result.addParticles(particles);

    return result;
}

OverlayDescription DescriptionConverterService::convertTOtoOverlayDescription(DataTO const& dataTO) const
{
    OverlayDescription result;
    result.elements.reserve(*dataTO.numCells + *dataTO.numParticles);
    for (int i = 0; i < *dataTO.numCells; ++i) {
        auto const& cellTO = dataTO.cells[i];
        OverlayElementDescription element;
        element.id = cellTO.id;
        element.cell = true;
        element.pos = {cellTO.pos.x, cellTO.pos.y};
        element.cellType = static_cast<CellType>(static_cast<unsigned int>(cellTO.cellType) % CellType_Count);
        element.selected = cellTO.selected;
        result.elements.emplace_back(element);
    }

    for (int i = 0; i < *dataTO.numParticles; ++i) {
        auto const& particleTO = dataTO.particles[i];
        OverlayElementDescription element;
        element.id = particleTO.id;
        element.cell = false;
        element.pos = {particleTO.pos.x, particleTO.pos.y};
        element.selected = particleTO.selected;
        result.elements.emplace_back(element);
    }
    return result;
}

void DescriptionConverterService::convertDescriptionToTO(DataTO& result, ClusteredDataDescription const& description) const
{
    std::unordered_map<uint64_t, int> cellIndexByIds;
    for (auto const& cluster: description._clusters) {
        for (auto const& cell : cluster._cells) {
            addCell(result, cell, cellIndexByIds);
        }
    }
    for (auto const& cluster : description._clusters) {
        for (auto const& cell : cluster._cells) {
            if (cell._id != 0) {
                setConnections(result, cell, cellIndexByIds);
            }
        }
    }
    for (auto const& particle : description._particles) {
        addParticle(result, particle);
    }
}

void DescriptionConverterService::convertDescriptionToTO(DataTO& result, DataDescription const& description) const
{
    std::unordered_map<uint64_t, int> cellIndexByIds;
    for (auto const& cell : description._cells) {
        addCell(result, cell, cellIndexByIds);
    }
    for (auto const& cell : description._cells) {
        if (cell._id != 0) {
            setConnections(result, cell, cellIndexByIds);
        }
    }
    for (auto const& particle : description._particles) {
        addParticle(result, particle);
    }
}

void DescriptionConverterService::convertDescriptionToTO(DataTO& result, CellDescription const& cell) const
{
    std::unordered_map<uint64_t, int> cellIndexByIds;
    addCell(result, cell, cellIndexByIds);
}

void DescriptionConverterService::convertDescriptionToTO(DataTO& result, ParticleDescription const& particle) const
{
    addParticle(result, particle);
}

void DescriptionConverterService::addAdditionalDataSizeForCell(CellDescription const& cell, uint64_t& additionalDataSize) const
{
    additionalDataSize += cell._metadata._name.size() + cell._metadata._description.size();
    switch (cell.getCellType()) {
    case CellType_Base: {
        additionalDataSize += MAX_CHANNELS * (MAX_CHANNELS + 1) * sizeof(float);
    } break;
    case CellType_Depot:
        break;
    case CellType_Constructor:
        additionalDataSize += std::get<ConstructorDescription>(cell._cellTypeData)._genome.size();
        break;
    case CellType_Sensor:
        break;
    case CellType_Oscillator:
        break;
    case CellType_Attacker:
        break;
    case CellType_Injector:
        additionalDataSize += std::get<InjectorDescription>(cell._cellTypeData)._genome.size();
        break;
    case CellType_Muscle:
        break;
    case CellType_Defender:
        break;
    case CellType_Reconnector:
        break;
    case CellType_Detonator:
        break;
    }
}    

namespace
{
    template <typename T>
    void setInplaceDifference(std::unordered_set<T>& a, std::unordered_set<T> const& b)
    {
        for (auto const& element : b) {
            a.erase(element);
        }
    }
}

auto DescriptionConverterService::scanAndCreateClusterDescription(
    DataTO const& dataTO,
    int startCellIndex,
    std::unordered_set<int>& freeCellIndices) const
    -> CreateClusterReturnData
{
    CreateClusterReturnData result; 

    std::unordered_set<int> currentCellIndices;
    currentCellIndices.insert(startCellIndex);
    std::unordered_set<int> scannedCellIndices = currentCellIndices;

    std::vector<CellDescription> cells;
    std::unordered_set<int> nextCellIndices;
    int cellDescIndex = 0;
    do {
        for (auto const& currentCellIndex : currentCellIndices) {
            cells.emplace_back(createCellDescription(dataTO, currentCellIndex));
            result.cellTOIndexToCellDescIndex.emplace(currentCellIndex, cellDescIndex);
            auto const& cellTO = dataTO.cells[currentCellIndex];
            for (int i = 0; i < cellTO.numConnections; ++i) {
                auto connectionTO = cellTO.connections[i];
                if (connectionTO.cellIndex != -1) {
                    if (scannedCellIndices.find(connectionTO.cellIndex) == scannedCellIndices.end()) {
                        nextCellIndices.insert(connectionTO.cellIndex);
                        scannedCellIndices.insert(connectionTO.cellIndex);
                    }
                }
            }
            ++cellDescIndex;
        }
        currentCellIndices = nextCellIndices;
        nextCellIndices.clear();
    } while (!currentCellIndices.empty());

    setInplaceDifference(freeCellIndices, scannedCellIndices);

    result.cluster.addCells(cells);

    return result;
}

CellDescription DescriptionConverterService::createCellDescription(DataTO const& dataTO, int cellIndex) const
{
    CellDescription result;

    auto const& cellTO = dataTO.cells[cellIndex];
    result._id = cellTO.id;
    result._pos = RealVector2D(cellTO.pos.x, cellTO.pos.y);
    result._vel = RealVector2D(cellTO.vel.x, cellTO.vel.y);
    result._energy = cellTO.energy;
    result._stiffness = cellTO.stiffness;
    std::vector<ConnectionDescription> connections;
    for (int i = 0; i < cellTO.numConnections; ++i) {
        auto const& connectionTO = cellTO.connections[i];
        ConnectionDescription connection;
        if (connectionTO.cellIndex != -1) {
            connection._cellId = dataTO.cells[connectionTO.cellIndex].id;
        } else {
            connection._cellId = 0;
        }
        connection._distance = connectionTO.distance;
        connection._angleFromPrevious = connectionTO.angleFromPrevious;
        connections.emplace_back(connection);
    }
    result._connections = connections;
    result._livingState = cellTO.livingState;
    result._creatureId = cellTO.creatureId;
    result._mutationId = cellTO.mutationId;
    result._ancestorMutationId = cellTO.ancestorMutationId;
    result._barrier = cellTO.barrier;
    result._age = cellTO.age;
    result._color = cellTO.color;
    result._absAngleToConnection0 = cellTO.absAngleToConnection0;
    result._genomeComplexity = cellTO.genomeComplexity;
    result._detectedByCreatureId = cellTO.detectedByCreatureId;
    result._cellTypeUsed = cellTO.cellTypeUsed;
    result._genomeNodeIndex = cellTO.genomeNodeIndex;

    auto const& metadataTO = cellTO.metadata;
    auto metadata = CellMetadataDescription();
    if (metadataTO.nameSize > 0) {
        auto const name = std::string(reinterpret_cast<char*>(&dataTO.auxiliaryData[metadataTO.nameDataIndex]), metadataTO.nameSize);
        metadata.name(name);
    }
    if (metadataTO.descriptionSize > 0) {
        auto const description =
            std::string(reinterpret_cast<char*>(&dataTO.auxiliaryData[metadataTO.descriptionDataIndex]), metadataTO.descriptionSize);
        metadata.description(description);
    }
    result._metadata = metadata;

    if (cellTO.cellType != CellType_Structure && cellTO.cellType != CellType_Free) {
        result._neuralNetwork = convertToNeuronDescription(dataTO, cellTO.neuralNetwork.dataIndex);
    }
    switch (cellTO.cellType) {
    case CellType_Base: {
        BaseDescription base;
        result._cellTypeData = base;
    } break;
    case CellType_Depot: {
        DepotDescription transmitter;
        transmitter._mode = cellTO.cellTypeData.transmitter.mode;
        result._cellTypeData = transmitter;
    } break;
    case CellType_Constructor: {
        ConstructorDescription constructor;
        constructor._autoTriggerInterval = cellTO.cellTypeData.constructor.autoTriggerInterval;
        constructor._constructionActivationTime = cellTO.cellTypeData.constructor.constructionActivationTime;
        convert(dataTO, cellTO.cellTypeData.constructor.genomeSize, cellTO.cellTypeData.constructor.genomeDataIndex, constructor._genome);
        constructor._numInheritedGenomeNodes = cellTO.cellTypeData.constructor.numInheritedGenomeNodes;
        constructor._lastConstructedCellId = cellTO.cellTypeData.constructor.lastConstructedCellId;
        constructor._genomeCurrentNodeIndex = cellTO.cellTypeData.constructor.genomeCurrentNodeIndex;
        constructor._genomeCurrentRepetition = cellTO.cellTypeData.constructor.genomeCurrentRepetition;
        constructor._genomeCurrentBranch = cellTO.cellTypeData.constructor.genomeCurrentBranch;
        constructor._offspringCreatureId = cellTO.cellTypeData.constructor.offspringCreatureId;
        constructor._offspringMutationId = cellTO.cellTypeData.constructor.offspringMutationId;
        constructor._genomeGeneration = cellTO.cellTypeData.constructor.genomeGeneration;
        constructor._constructionAngle1 = cellTO.cellTypeData.constructor.constructionAngle1;
        constructor._constructionAngle2 = cellTO.cellTypeData.constructor.constructionAngle2;
        result._cellTypeData = constructor;
    } break;
    case CellType_Sensor: {
        SensorDescription sensor;
        sensor._autoTriggerInterval = cellTO.cellTypeData.sensor.autoTriggerInterval;
        sensor._minDensity = cellTO.cellTypeData.sensor.minDensity;
        sensor._minRange = cellTO.cellTypeData.sensor.minRange >= 0 ? std::make_optional(cellTO.cellTypeData.sensor.minRange) : std::nullopt;
        sensor._maxRange = cellTO.cellTypeData.sensor.maxRange >= 0 ? std::make_optional(cellTO.cellTypeData.sensor.maxRange) : std::nullopt;
        sensor._restrictToColor =
            cellTO.cellTypeData.sensor.restrictToColor != 255 ? std::make_optional(cellTO.cellTypeData.sensor.restrictToColor) : std::nullopt;
        sensor._restrictToMutants = cellTO.cellTypeData.sensor.restrictToMutants;
        result._cellTypeData = sensor;
    } break;
    case CellType_Oscillator: {
        OscillatorDescription oscillator;
        oscillator._autoTriggerInterval = cellTO.cellTypeData.oscillator.autoTriggerInterval;
        oscillator._alternationInterval = cellTO.cellTypeData.oscillator.alternationInterval;
        oscillator._numPulses = cellTO.cellTypeData.oscillator.numPulses;
        result._cellTypeData = oscillator;
    } break;
    case CellType_Attacker: {
        AttackerDescription attacker;
        attacker._mode = cellTO.cellTypeData.attacker.mode;
        result._cellTypeData = attacker;
    } break;
    case CellType_Injector: {
        InjectorDescription injector;
        injector._mode = cellTO.cellTypeData.injector.mode;
        injector._counter = cellTO.cellTypeData.injector.counter;
        convert(dataTO, cellTO.cellTypeData.injector.genomeSize, cellTO.cellTypeData.injector.genomeDataIndex, injector._genome);
        injector._genomeGeneration = cellTO.cellTypeData.injector.genomeGeneration;
        result._cellTypeData = injector;
    } break;
    case CellType_Muscle: {
        MuscleDescription muscle;
        if (cellTO.cellTypeData.muscle.mode == MuscleMode_Bending) {
            AutoBendingDescription bending;
            bending._maxAngleDeviation = cellTO.cellTypeData.muscle.modeData.autoBending.maxAngleDeviation;
            bending._frontBackVelRatio = cellTO.cellTypeData.muscle.modeData.autoBending.frontBackVelRatio;
            bending._initialAngle = cellTO.cellTypeData.muscle.modeData.autoBending.initialAngle;
            bending._lastAngle = cellTO.cellTypeData.muscle.modeData.autoBending.lastAngle;
            bending._forward = cellTO.cellTypeData.muscle.modeData.autoBending.forward;
            bending._activation = cellTO.cellTypeData.muscle.modeData.autoBending.activation;
            bending._activationCountdown = cellTO.cellTypeData.muscle.modeData.autoBending.activationCountdown;
            bending._impulseAlreadyApplied = cellTO.cellTypeData.muscle.modeData.autoBending.impulseAlreadyApplied;
            muscle._mode = bending;
        }
        muscle._frontAngle = cellTO.cellTypeData.muscle.frontAngle;
        muscle._lastMovementX = cellTO.cellTypeData.muscle.lastMovementX;
        muscle._lastMovementY = cellTO.cellTypeData.muscle.lastMovementY;
        result._cellTypeData = muscle;
    } break;
    case CellType_Defender: {
        DefenderDescription defender;
        defender._mode = cellTO.cellTypeData.defender.mode;
        result._cellTypeData = defender;
    } break;
    case CellType_Reconnector: {
        ReconnectorDescription reconnector;
        reconnector._restrictToColor =
            cellTO.cellTypeData.reconnector.restrictToColor != 255 ? std::make_optional(cellTO.cellTypeData.reconnector.restrictToColor) : std::nullopt;
        reconnector._restrictToMutants = cellTO.cellTypeData.reconnector.restrictToMutants;
        result._cellTypeData = reconnector;
    } break;
    case CellType_Detonator: {
        DetonatorDescription detonator;
        detonator._state = cellTO.cellTypeData.detonator.state;
        detonator._countdown = cellTO.cellTypeData.detonator.countdown;
        result._cellTypeData = detonator;
    } break;
    }

    SignalRoutingRestrictionDescription routingRestriction;
    routingRestriction._active = cellTO.signalRoutingRestriction.active;
    routingRestriction._baseAngle = cellTO.signalRoutingRestriction.baseAngle;
    routingRestriction._openingAngle = cellTO.signalRoutingRestriction.openingAngle;
    result._signalRoutingRestriction = routingRestriction;
    result._signalRelaxationTime = cellTO.signalRelaxationTime;
    if (cellTO.signal.active) {
        SignalDescription signal;
        for (int i = 0; i < MAX_CHANNELS; ++i) {
            signal._channels[i] = cellTO.signal.channels[i];
        }
        signal._origin = cellTO.signal.origin;
        signal._targetX = cellTO.signal.targetX;
        signal._targetY = cellTO.signal.targetY;
        result._signal = signal;
    }
    result._activationTime = cellTO.activationTime;
    return result;
}

namespace
{
    void checkAndCorrectInvalidEnergy(float& energy)
    {
        if (std::isnan(energy) || energy < 0 || energy > 1e12) {
            energy = 0;
        }
    }
}

void DescriptionConverterService::addParticle(DataTO const& dataTO, ParticleDescription const& particleDesc) const
{
    auto particleIndex = (*dataTO.numParticles)++;

	ParticleTO& particleTO = dataTO.particles[particleIndex];
	particleTO.id = particleDesc._id == 0 ? NumberGenerator::get().getId() : particleDesc._id;
    particleTO.pos = {particleDesc._pos.x, particleDesc._pos.y};
    particleTO.vel = {particleDesc._vel.x, particleDesc._vel.y};
    particleTO.energy = particleDesc._energy;
    checkAndCorrectInvalidEnergy(particleTO.energy);
    particleTO.color = particleDesc._color;
}

void DescriptionConverterService::addCell(DataTO const& dataTO, CellDescription const& cellDesc, std::unordered_map<uint64_t, int>& cellIndexTOByIds) const
{
    int cellIndex = (*dataTO.numCells)++;
    CellTO& cellTO = dataTO.cells[cellIndex];
    cellTO.id = cellDesc._id == 0 ? NumberGenerator::get().getId() : cellDesc._id;
    cellTO.pos = {cellDesc._pos.x, cellDesc._pos.y};
    cellTO.vel = {cellDesc._vel.x, cellDesc._vel.y};
    cellTO.energy = cellDesc._energy;
    checkAndCorrectInvalidEnergy(cellTO.energy);
    cellTO.stiffness = cellDesc._stiffness;
    cellTO.livingState = cellDesc._livingState;
    cellTO.creatureId = cellDesc._creatureId;
    cellTO.mutationId = cellDesc._mutationId;
    cellTO.ancestorMutationId = cellDesc._ancestorMutationId;
    cellTO.cellType = cellDesc.getCellType();
    cellTO.detectedByCreatureId = cellDesc._detectedByCreatureId;
    cellTO.cellTypeUsed = cellDesc._cellTypeUsed;
    cellTO.genomeNodeIndex = cellDesc._genomeNodeIndex;

    auto cellType = cellDesc.getCellType();
    if (cellType != CellType_Structure && cellType != CellType_Free) {
        convertToNeuronData(dataTO, *cellDesc._neuralNetwork, cellTO.neuralNetwork.dataIndex);
    }
    switch (cellType) {
    case CellType_Base: {
        BaseTO baseTO;
        cellTO.cellTypeData.base = baseTO;
    } break;
    case CellType_Depot: {
        auto const& transmitterDesc = std::get<DepotDescription>(cellDesc._cellTypeData);
        TransmitterTO& transmitterTO = cellTO.cellTypeData.transmitter;
        transmitterTO.mode = transmitterDesc._mode;
    } break;
    case CellType_Constructor: {
        auto const& constructorDesc = std::get<ConstructorDescription>(cellDesc._cellTypeData);
        ConstructorTO& constructorTO = cellTO.cellTypeData.constructor;
        constructorTO.autoTriggerInterval = constructorDesc._autoTriggerInterval;
        constructorTO.constructionActivationTime = constructorDesc._constructionActivationTime;
        CHECK(constructorDesc._genome.size() >= Const::GenomeHeaderSize)
        convert(dataTO, constructorDesc._genome, constructorTO.genomeSize, constructorTO.genomeDataIndex);
        constructorTO.numInheritedGenomeNodes = static_cast<uint16_t>(constructorDesc._numInheritedGenomeNodes);
        constructorTO.lastConstructedCellId = constructorDesc._lastConstructedCellId;
        constructorTO.genomeCurrentNodeIndex = static_cast<uint16_t>(constructorDesc._genomeCurrentNodeIndex);
        constructorTO.genomeCurrentRepetition = static_cast<uint16_t>(constructorDesc._genomeCurrentRepetition);
        constructorTO.genomeCurrentBranch = static_cast<uint8_t>(constructorDesc._genomeCurrentBranch);
        constructorTO.offspringCreatureId = constructorDesc._offspringCreatureId;
        constructorTO.offspringMutationId = constructorDesc._offspringMutationId;
        constructorTO.genomeGeneration = constructorDesc._genomeGeneration;
        constructorTO.constructionAngle1 = constructorDesc._constructionAngle1;
        constructorTO.constructionAngle2 = constructorDesc._constructionAngle2;
    } break;
    case CellType_Sensor: {
        auto const& sensorDesc = std::get<SensorDescription>(cellDesc._cellTypeData);
        SensorTO& sensorTO = cellTO.cellTypeData.sensor;
        sensorTO.autoTriggerInterval = sensorDesc._autoTriggerInterval;
        sensorTO.restrictToColor = sensorDesc._restrictToColor.value_or(255);
        sensorTO.restrictToMutants = sensorDesc._restrictToMutants;
        sensorTO.minDensity = sensorDesc._minDensity;
        sensorTO.minRange = static_cast<int8_t>(sensorDesc._minRange.value_or(-1));
        sensorTO.maxRange = static_cast<int8_t>(sensorDesc._maxRange.value_or(-1));
    } break;
    case CellType_Oscillator: {
        auto const& oscillatorDesc = std::get<OscillatorDescription>(cellDesc._cellTypeData);
        OscillatorTO& oscillatorTO = cellTO.cellTypeData.oscillator;
        oscillatorTO.autoTriggerInterval = oscillatorDesc._autoTriggerInterval;
        oscillatorTO.alternationInterval = oscillatorDesc._alternationInterval;
        oscillatorTO.numPulses = oscillatorDesc._numPulses;
    } break;
    case CellType_Attacker: {
        auto const& attackerDesc = std::get<AttackerDescription>(cellDesc._cellTypeData);
        AttackerTO& attackerTO = cellTO.cellTypeData.attacker;
        attackerTO.mode = attackerDesc._mode;
    } break;
    case CellType_Injector: {
        auto const& injectorDesc = std::get<InjectorDescription>(cellDesc._cellTypeData);
        InjectorTO& injectorTO = cellTO.cellTypeData.injector;
        injectorTO.mode = injectorDesc._mode;
        injectorTO.counter = injectorDesc._counter;
        CHECK(injectorDesc._genome.size() >= Const::GenomeHeaderSize)
        convert(dataTO, injectorDesc._genome, injectorTO.genomeSize, injectorTO.genomeDataIndex);
        injectorTO.genomeGeneration = injectorDesc._genomeGeneration;
    } break;
    case CellType_Muscle: {
        auto const& muscleDesc = std::get<MuscleDescription>(cellDesc._cellTypeData);
        MuscleTO& muscleTO = cellTO.cellTypeData.muscle;
        muscleTO.mode = muscleDesc.getMode();
        if (muscleTO.mode == MuscleMode_Bending) {
            auto const& bendingDesc = std::get<AutoBendingDescription>(muscleDesc._mode);
            AutoBendingTO& bendingTO = muscleTO.modeData.autoBending;
            bendingTO.maxAngleDeviation = bendingDesc._maxAngleDeviation;
            bendingTO.frontBackVelRatio = bendingDesc._frontBackVelRatio;
            bendingTO.initialAngle = bendingDesc._initialAngle;
            bendingTO.lastAngle = bendingDesc._lastAngle;
            bendingTO.forward = bendingDesc._forward;
            bendingTO.activation = bendingDesc._activation;
            bendingTO.activationCountdown = bendingDesc._activationCountdown;
            bendingTO.impulseAlreadyApplied = bendingDesc._impulseAlreadyApplied;
        }
        muscleTO.frontAngle = muscleDesc._frontAngle;
        muscleTO.lastMovementX = muscleDesc._lastMovementX;
        muscleTO.lastMovementY = muscleDesc._lastMovementY;
    } break;
    case CellType_Defender: {
        auto const& defenderDesc = std::get<DefenderDescription>(cellDesc._cellTypeData);
        DefenderTO& defenderTO = cellTO.cellTypeData.defender;
        defenderTO.mode = defenderDesc._mode;
    } break;
    case CellType_Reconnector: {
        auto const& reconnectorDesc = std::get<ReconnectorDescription>(cellDesc._cellTypeData);
        ReconnectorTO& reconnectorTO = cellTO.cellTypeData.reconnector;
        reconnectorTO.restrictToColor = toUInt8(reconnectorDesc._restrictToColor.value_or(255));
        reconnectorTO.restrictToMutants = reconnectorDesc._restrictToMutants;
    } break;
    case CellType_Detonator: {
        auto const& detonatorDesc = std::get<DetonatorDescription>(cellDesc._cellTypeData);
        DetonatorTO& detonatorTO = cellTO.cellTypeData.detonator;
        detonatorTO.state = detonatorDesc._state;
        detonatorTO.countdown = detonatorDesc._countdown;
    } break;
    }
    cellTO.signalRoutingRestriction.active = cellDesc._signalRoutingRestriction._active;
    cellTO.signalRoutingRestriction.baseAngle = cellDesc._signalRoutingRestriction._baseAngle;
    cellTO.signalRoutingRestriction.openingAngle = cellDesc._signalRoutingRestriction._openingAngle;
    cellTO.signalRelaxationTime = cellDesc._signalRelaxationTime;
    cellTO.signal.active = cellDesc._signal.has_value();
    if (cellTO.signal.active) {
        for (int i = 0; i < MAX_CHANNELS; ++i) {
            cellTO.signal.channels[i] = cellDesc._signal->_channels[i];
        }
        cellTO.signal.origin = cellDesc._signal->_origin;
        cellTO.signal.targetX = cellDesc._signal->_targetX;
        cellTO.signal.targetY = cellDesc._signal->_targetY;
    }
    cellTO.activationTime = cellDesc._activationTime;
    cellTO.numConnections = 0;
    cellTO.barrier = cellDesc._barrier;
    cellTO.age = cellDesc._age;
    cellTO.color = cellDesc._color;
    cellTO.absAngleToConnection0 = cellDesc._absAngleToConnection0;
    cellTO.genomeComplexity = cellDesc._genomeComplexity;
    convert(dataTO, cellDesc._metadata._name, cellTO.metadata.nameSize, cellTO.metadata.nameDataIndex);
    convert(dataTO, cellDesc._metadata._description, cellTO.metadata.descriptionSize, cellTO.metadata.descriptionDataIndex);
	cellIndexTOByIds.insert_or_assign(cellTO.id, cellIndex);
}

void DescriptionConverterService::setConnections(DataTO const& dataTO, CellDescription const& cellToAdd, std::unordered_map<uint64_t, int> const& cellIndexByIds) const
{
    int index = 0;
    auto& cellTO = dataTO.cells[cellIndexByIds.at(cellToAdd._id)];
    float angleOffset = 0;
    for (ConnectionDescription const& connection : cellToAdd._connections) {
        if (connection._cellId != 0) {
            cellTO.connections[index].cellIndex = cellIndexByIds.at(connection._cellId);
            cellTO.connections[index].distance = connection._distance;
            cellTO.connections[index].angleFromPrevious = connection._angleFromPrevious + angleOffset;
            ++index;
            angleOffset = 0;
        } else {
            angleOffset += connection._angleFromPrevious;
        }
    }
    if (angleOffset != 0 && index > 0) {
        cellTO.connections[0].angleFromPrevious += angleOffset;
    }
    cellTO.numConnections = index;
}