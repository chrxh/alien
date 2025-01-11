#include "DescriptionConverter.h"

#include <cmath>
#include <algorithm>
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

    NeuronDescription convertToNeuronDescription(DataTO const& dataTO, uint64_t sourceIndex)
    {
        NeuronDescription result;

        BytesAsFloat bytesAsFloat;
        int index = 0;
        for (int row = 0; row < MAX_CHANNELS; ++row) {
            for (int col = 0; col < MAX_CHANNELS; ++col) {
                for (int i = 0; i < 4; ++i) {
                    bytesAsFloat.b[i] = dataTO.auxiliaryData[sourceIndex + index];
                    ++index;
                }
                result.weights[row][col] = bytesAsFloat.f;
            }
        }
        for (int channel = 0; channel < MAX_CHANNELS; ++channel) {
            for (int i = 0; i < 4; ++i) {
                bytesAsFloat.b[i] = dataTO.auxiliaryData[sourceIndex + index];
                ++index;
            }
            result.biases[channel] = bytesAsFloat.f;
        }
        for (int channel = 0; channel < MAX_CHANNELS; ++channel) {
            result.activationFunctions[channel] = dataTO.auxiliaryData[sourceIndex + index];
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

    void convertToNeuronData(DataTO const& dataTO, NeuronDescription const& neuronDesc, uint64_t& targetIndex)
    {
        targetIndex = *dataTO.numAuxiliaryData;
        *dataTO.numAuxiliaryData *= NeuronTO::NeuronDataSize;

        BytesAsFloat bytesAsFloat;
        int bytePos = 0;
        for (int row = 0; row < MAX_CHANNELS; ++row) {
            for (int col = 0; col < MAX_CHANNELS; ++col) {
                bytesAsFloat.f = neuronDesc.weights[row][col];
                for (int i = 0; i < 4; ++i) {
                    dataTO.auxiliaryData[targetIndex + bytePos] = bytesAsFloat.b[i];
                    ++bytePos;
                }
            }
        }
        for (int channel = 0; channel < MAX_CHANNELS; ++channel) {
            bytesAsFloat.f = neuronDesc.biases[channel];
            for (int i = 0; i < 4; ++i) {
                dataTO.auxiliaryData[targetIndex + bytePos] = bytesAsFloat.b[i];
                ++bytePos;
            }
        }
        for (int channel = 0; channel < MAX_CHANNELS; ++channel) {
            dataTO.auxiliaryData[targetIndex + bytePos] = neuronDesc.activationFunctions[channel];
            ++bytePos;
        }
    }
}

DescriptionConverter::DescriptionConverter(SimulationParameters const& parameters)
    : _parameters(parameters)
{}

ArraySizes DescriptionConverter::getArraySizes(DataDescription const& data) const
{
    ArraySizes result;
    result.cellArraySize = data.cells.size();
    result.particleArraySize = data.particles.size();
    for (auto const& cell : data.cells) {
        addAdditionalDataSizeForCell(cell, result.auxiliaryDataSize);
    }
    return result;
}

ArraySizes DescriptionConverter::getArraySizes(ClusteredDataDescription const& data) const
{
    ArraySizes result;
    for (auto const& cluster : data.clusters) {
        result.cellArraySize += cluster.cells.size();
        for (auto const& cell : cluster.cells) {
            addAdditionalDataSizeForCell(cell, result.auxiliaryDataSize);
        }
    }
    result.particleArraySize = data.particles.size();
    return result;
}

ClusteredDataDescription DescriptionConverter::convertTOtoClusteredDataDescription(DataTO const& dataTO) const
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
                                   .setId(particle.id)
                                   .setPos({particle.pos.x, particle.pos.y})
                                   .setVel({particle.vel.x, particle.vel.y})
                                   .setEnergy(particle.energy)
                                   .setColor(particle.color));
    }
    result.addParticles(particles);

    return result;
}

DataDescription DescriptionConverter::convertTOtoDataDescription(DataTO const& dataTO) const
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
                                   .setId(particle.id)
                                   .setPos({particle.pos.x, particle.pos.y})
                                   .setVel({particle.vel.x, particle.vel.y})
                                   .setEnergy(particle.energy)
                                   .setColor(particle.color));
    }
    result.addParticles(particles);

    return result;
}

OverlayDescription DescriptionConverter::convertTOtoOverlayDescription(DataTO const& dataTO) const
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

void DescriptionConverter::convertDescriptionToTO(DataTO& result, ClusteredDataDescription const& description) const
{
    std::unordered_map<uint64_t, int> cellIndexByIds;
    for (auto const& cluster: description.clusters) {
        for (auto const& cell : cluster.cells) {
            addCell(result, cell, cellIndexByIds);
        }
    }
    for (auto const& cluster : description.clusters) {
        for (auto const& cell : cluster.cells) {
            if (cell.id != 0) {
                setConnections(result, cell, cellIndexByIds);
            }
        }
    }
    for (auto const& particle : description.particles) {
        addParticle(result, particle);
    }
}

void DescriptionConverter::convertDescriptionToTO(DataTO& result, DataDescription const& description) const
{
    std::unordered_map<uint64_t, int> cellIndexByIds;
    for (auto const& cell : description.cells) {
        addCell(result, cell, cellIndexByIds);
    }
    for (auto const& cell : description.cells) {
        if (cell.id != 0) {
            setConnections(result, cell, cellIndexByIds);
        }
    }
    for (auto const& particle : description.particles) {
        addParticle(result, particle);
    }
}

void DescriptionConverter::convertDescriptionToTO(DataTO& result, CellDescription const& cell) const
{
    std::unordered_map<uint64_t, int> cellIndexByIds;
    addCell(result, cell, cellIndexByIds);
}

void DescriptionConverter::convertDescriptionToTO(DataTO& result, ParticleDescription const& particle) const
{
    addParticle(result, particle);
}

void DescriptionConverter::addAdditionalDataSizeForCell(CellDescription const& cell, uint64_t& additionalDataSize) const
{
    additionalDataSize += cell.metadata.name.size() + cell.metadata.description.size();
    switch (cell.getCellType()) {
    case CellType_Neuron: {
        additionalDataSize += MAX_CHANNELS * (MAX_CHANNELS + 1) * sizeof(float);
    } break;
    case CellType_Transmitter:
        break;
    case CellType_Constructor:
        additionalDataSize += std::get<ConstructorDescription>(*cell.cellTypeData).genome.size();
        break;
    case CellType_Sensor:
        break;
    case CellType_Oscillator:
        break;
    case CellType_Attacker:
        break;
    case CellType_Injector:
        additionalDataSize += std::get<InjectorDescription>(*cell.cellTypeData).genome.size();
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

auto DescriptionConverter::scanAndCreateClusterDescription(
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

CellDescription DescriptionConverter::createCellDescription(DataTO const& dataTO, int cellIndex) const
{
    CellDescription result;

    auto const& cellTO = dataTO.cells[cellIndex];
    result.id = cellTO.id;
    result.pos = RealVector2D(cellTO.pos.x, cellTO.pos.y);
    result.vel = RealVector2D(cellTO.vel.x, cellTO.vel.y);
    result.energy = cellTO.energy;
    result.stiffness = cellTO.stiffness;
    std::vector<ConnectionDescription> connections;
    for (int i = 0; i < cellTO.numConnections; ++i) {
        auto const& connectionTO = cellTO.connections[i];
        ConnectionDescription connection;
        if (connectionTO.cellIndex != -1) {
            connection.cellId = dataTO.cells[connectionTO.cellIndex].id;
        } else {
            connection.cellId = 0;
        }
        connection.distance = connectionTO.distance;
        connection.angleFromPrevious = connectionTO.angleFromPrevious;
        connections.emplace_back(connection);
    }
    result.connections = connections;
    result.livingState = cellTO.livingState;
    result.creatureId = cellTO.creatureId;
    result.mutationId = cellTO.mutationId;
    result.ancestorMutationId = cellTO.ancestorMutationId;
    result.barrier = cellTO.barrier;
    result.age = cellTO.age;
    result.color = cellTO.color;
    result.genomeComplexity = cellTO.genomeComplexity;
    result.detectedByCreatureId = cellTO.detectedByCreatureId;
    result.cellTypeUsed = cellTO.cellTypeUsed;
    result.genomeNodeIndex = cellTO.genomeNodeIndex;

    auto const& metadataTO = cellTO.metadata;
    auto metadata = CellMetadataDescription();
    if (metadataTO.nameSize > 0) {
        auto const name = std::string(reinterpret_cast<char*>(&dataTO.auxiliaryData[metadataTO.nameDataIndex]), metadataTO.nameSize);
        metadata.setName(name);
    }
    if (metadataTO.descriptionSize > 0) {
        auto const description =
            std::string(reinterpret_cast<char*>(&dataTO.auxiliaryData[metadataTO.descriptionDataIndex]), metadataTO.descriptionSize);
        metadata.setDescription(description);
    }
    result.metadata = metadata;

    switch (cellTO.cellType) {
    case CellType_Neuron: {
        result.cellTypeData = convertToNeuronDescription(dataTO, cellTO.cellTypeData.neuron.neuronDataIndex);
    } break;
    case CellType_Transmitter: {
        TransmitterDescription transmitter;
        transmitter.mode = cellTO.cellTypeData.transmitter.mode;
        result.cellTypeData = transmitter;
    } break;
    case CellType_Constructor: {
        ConstructorDescription constructor;
        constructor.activationMode = cellTO.cellTypeData.constructor.activationMode;
        constructor.constructionActivationTime = cellTO.cellTypeData.constructor.constructionActivationTime;
        convert(dataTO, cellTO.cellTypeData.constructor.genomeSize, cellTO.cellTypeData.constructor.genomeDataIndex, constructor.genome);
        constructor.numInheritedGenomeNodes = cellTO.cellTypeData.constructor.numInheritedGenomeNodes;
        constructor.lastConstructedCellId = cellTO.cellTypeData.constructor.lastConstructedCellId;
        constructor.genomeCurrentNodeIndex = cellTO.cellTypeData.constructor.genomeCurrentNodeIndex;
        constructor.genomeCurrentRepetition = cellTO.cellTypeData.constructor.genomeCurrentRepetition;
        constructor.currentBranch = cellTO.cellTypeData.constructor.currentBranch;
        constructor.offspringCreatureId = cellTO.cellTypeData.constructor.offspringCreatureId;
        constructor.offspringMutationId = cellTO.cellTypeData.constructor.offspringMutationId;
        constructor.genomeGeneration = cellTO.cellTypeData.constructor.genomeGeneration;
        constructor.constructionAngle1 = cellTO.cellTypeData.constructor.constructionAngle1;
        constructor.constructionAngle2 = cellTO.cellTypeData.constructor.constructionAngle2;
        result.cellTypeData = constructor;
    } break;
    case CellType_Sensor: {
        SensorDescription sensor;
        sensor.minDensity = cellTO.cellTypeData.sensor.minDensity;
        sensor.minRange = cellTO.cellTypeData.sensor.minRange >= 0 ? std::make_optional(cellTO.cellTypeData.sensor.minRange) : std::nullopt;
        sensor.maxRange = cellTO.cellTypeData.sensor.maxRange >= 0 ? std::make_optional(cellTO.cellTypeData.sensor.maxRange) : std::nullopt;
        sensor.restrictToColor =
            cellTO.cellTypeData.sensor.restrictToColor != 255 ? std::make_optional(cellTO.cellTypeData.sensor.restrictToColor) : std::nullopt;
        sensor.restrictToMutants = cellTO.cellTypeData.sensor.restrictToMutants;
        sensor.memoryChannel1 = cellTO.cellTypeData.sensor.memoryChannel1;
        sensor.memoryChannel2 = cellTO.cellTypeData.sensor.memoryChannel2;
        sensor.memoryChannel3 = cellTO.cellTypeData.sensor.memoryChannel3;
        sensor.memoryTargetX = cellTO.cellTypeData.sensor.memoryTargetX;
        sensor.memoryTargetY = cellTO.cellTypeData.sensor.memoryTargetY;
        result.cellTypeData = sensor;
    } break;
    case CellType_Oscillator: {
        OscillatorDescription oscillator;
        oscillator.pulseMode = cellTO.cellTypeData.oscillator.pulseMode;
        oscillator.alternationMode = cellTO.cellTypeData.oscillator.alternationMode;
        result.cellTypeData = oscillator;
    } break;
    case CellType_Attacker: {
        AttackerDescription attacker;
        attacker.mode = cellTO.cellTypeData.attacker.mode;
        result.cellTypeData = attacker;
    } break;
    case CellType_Injector: {
        InjectorDescription injector;
        injector.mode = cellTO.cellTypeData.injector.mode;
        injector.counter = cellTO.cellTypeData.injector.counter;
        convert(dataTO, cellTO.cellTypeData.injector.genomeSize, cellTO.cellTypeData.injector.genomeDataIndex, injector.genome);
        injector.genomeGeneration = cellTO.cellTypeData.injector.genomeGeneration;
        result.cellTypeData = injector;
    } break;
    case CellType_Muscle: {
        MuscleDescription muscle;
        muscle.mode = cellTO.cellTypeData.muscle.mode;
        muscle.lastBendingDirection = cellTO.cellTypeData.muscle.lastBendingDirection;
        muscle.lastBendingSourceIndex = cellTO.cellTypeData.muscle.lastBendingSourceIndex;
        muscle.consecutiveBendingAngle = cellTO.cellTypeData.muscle.consecutiveBendingAngle;
        muscle.lastMovementX = cellTO.cellTypeData.muscle.lastMovementX;
        muscle.lastMovementY = cellTO.cellTypeData.muscle.lastMovementY;
        result.cellTypeData = muscle;
    } break;
    case CellType_Defender: {
        DefenderDescription defender;
        defender.mode = cellTO.cellTypeData.defender.mode;
        result.cellTypeData = defender;
    } break;
    case CellType_Reconnector: {
        ReconnectorDescription reconnector;
        reconnector.restrictToColor =
            cellTO.cellTypeData.reconnector.restrictToColor != 255 ? std::make_optional(cellTO.cellTypeData.reconnector.restrictToColor) : std::nullopt;
        reconnector.restrictToMutants = cellTO.cellTypeData.reconnector.restrictToMutants;
        result.cellTypeData = reconnector;
    } break;
    case CellType_Detonator: {
        DetonatorDescription detonator;
        detonator.state = cellTO.cellTypeData.detonator.state;
        detonator.countdown = cellTO.cellTypeData.detonator.countdown;
        result.cellTypeData = detonator;
    } break;
    }

    if (cellTO.signalRoutingRestriction.active) {
        SignalRoutingRestrictionDescription routingRestriction;
        routingRestriction.baseAngle = cellTO.signalRoutingRestriction.baseAngle;
        routingRestriction.openingAngle = cellTO.signalRoutingRestriction.openingAngle;
        result.signalRoutingRestriction = routingRestriction;
    }
    if (cellTO.signal.active) {
        SignalDescription signal;
        for (int i = 0; i < MAX_CHANNELS; ++i) {
            signal.channels[i] = cellTO.signal.channels[i];
        }
        signal.origin = cellTO.signal.origin;
        signal.targetX = cellTO.signal.targetX;
        signal.targetY = cellTO.signal.targetY;
        signal.prevCellIds.resize(cellTO.signal.numPrevCells);
        for (int i = 0; i < cellTO.signal.numPrevCells; ++i) {
            signal.prevCellIds[i] = cellTO.signal.prevCellIds[i];
        }
        result.signal = signal;
    }
    result.activationTime = cellTO.activationTime;
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

void DescriptionConverter::addParticle(DataTO const& dataTO, ParticleDescription const& particleDesc) const
{
    auto particleIndex = (*dataTO.numParticles)++;

	ParticleTO& particleTO = dataTO.particles[particleIndex];
	particleTO.id = particleDesc.id == 0 ? NumberGenerator::get().getId() : particleDesc.id;
    particleTO.pos = {particleDesc.pos.x, particleDesc.pos.y};
    particleTO.vel = {particleDesc.vel.x, particleDesc.vel.y};
    particleTO.energy = particleDesc.energy;
    checkAndCorrectInvalidEnergy(particleTO.energy);
    particleTO.color = particleDesc.color;
}

void DescriptionConverter::addCell(
    DataTO const& dataTO, CellDescription const& cellDesc, std::unordered_map<uint64_t, int>& cellIndexTOByIds) const
{
    int cellIndex = (*dataTO.numCells)++;
    CellTO& cellTO = dataTO.cells[cellIndex];
    cellTO.id = cellDesc.id == 0 ? NumberGenerator::get().getId() : cellDesc.id;
	cellTO.pos= { cellDesc.pos.x, cellDesc.pos.y };
    cellTO.vel = {cellDesc.vel.x, cellDesc.vel.y};
    cellTO.energy = cellDesc.energy;
    checkAndCorrectInvalidEnergy(cellTO.energy);
    cellTO.stiffness = cellDesc.stiffness;
    cellTO.livingState = cellDesc.livingState;
    cellTO.creatureId = cellDesc.creatureId;
    cellTO.mutationId = cellDesc.mutationId;
    cellTO.ancestorMutationId = cellDesc.ancestorMutationId;
    cellTO.cellType = cellDesc.getCellType();
    cellTO.detectedByCreatureId = cellDesc.detectedByCreatureId;
    cellTO.cellTypeUsed = cellDesc.cellTypeUsed;
    cellTO.genomeNodeIndex = cellDesc.genomeNodeIndex;
    switch (cellDesc.getCellType()) {
    case CellType_Neuron: {
        auto const& neuronDesc = std::get<NeuronDescription>(*cellDesc.cellTypeData);

        NeuronTO neuronTO;
        convertToNeuronData(dataTO, neuronDesc, neuronTO.neuronDataIndex);
        cellTO.cellTypeData.neuron = neuronTO;
    } break;
    case CellType_Transmitter: {
        auto const& transmitterDesc = std::get<TransmitterDescription>(*cellDesc.cellTypeData);
        TransmitterTO transmitterTO;
        transmitterTO.mode = transmitterDesc.mode;
        cellTO.cellTypeData.transmitter = transmitterTO;
    } break;
    case CellType_Constructor: {
        auto const& constructorDesc = std::get<ConstructorDescription>(*cellDesc.cellTypeData);
        ConstructorTO constructorTO;
        constructorTO.activationMode = constructorDesc.activationMode;
        constructorTO.constructionActivationTime = constructorDesc.constructionActivationTime;
        CHECK(constructorDesc.genome.size() >= Const::GenomeHeaderSize)
        convert(dataTO, constructorDesc.genome, constructorTO.genomeSize, constructorTO.genomeDataIndex);
        constructorTO.numInheritedGenomeNodes = static_cast<uint16_t>(constructorDesc.numInheritedGenomeNodes);
        constructorTO.lastConstructedCellId = constructorDesc.lastConstructedCellId;
        constructorTO.genomeCurrentNodeIndex = static_cast<uint16_t>(constructorDesc.genomeCurrentNodeIndex);
        constructorTO.genomeCurrentRepetition = static_cast<uint16_t>(constructorDesc.genomeCurrentRepetition);
        constructorTO.currentBranch = static_cast<uint8_t>(constructorDesc.currentBranch);
        constructorTO.offspringCreatureId = constructorDesc.offspringCreatureId;
        constructorTO.offspringMutationId = constructorDesc.offspringMutationId;
        constructorTO.genomeGeneration = constructorDesc.genomeGeneration;
        constructorTO.constructionAngle1 = constructorDesc.constructionAngle1;
        constructorTO.constructionAngle2 = constructorDesc.constructionAngle2;
        cellTO.cellTypeData.constructor = constructorTO;
    } break;
    case CellType_Sensor: {
        auto const& sensorDesc = std::get<SensorDescription>(*cellDesc.cellTypeData);
        SensorTO sensorTO;
        sensorTO.restrictToColor = sensorDesc.restrictToColor.value_or(255);
        sensorTO.restrictToMutants = sensorDesc.restrictToMutants;
        sensorTO.minDensity = sensorDesc.minDensity;
        sensorTO.minRange = static_cast<int8_t>(sensorDesc.minRange.value_or(-1));
        sensorTO.maxRange = static_cast<int8_t>(sensorDesc.maxRange.value_or(-1));
        sensorTO.memoryChannel1 = sensorDesc.memoryChannel1;
        sensorTO.memoryChannel2 = sensorDesc.memoryChannel2;
        sensorTO.memoryChannel3 = sensorDesc.memoryChannel3;
        sensorTO.memoryTargetX = sensorDesc.memoryTargetX;
        sensorTO.memoryTargetY = sensorDesc.memoryTargetY;
        cellTO.cellTypeData.sensor = sensorTO;
    } break;
    case CellType_Oscillator: {
        auto const& oscillatorDesc = std::get<OscillatorDescription>(*cellDesc.cellTypeData);
        OscillatorTO oscillatorTO;
        oscillatorTO.pulseMode = oscillatorDesc.pulseMode;
        oscillatorTO.alternationMode = oscillatorDesc.alternationMode;
        cellTO.cellTypeData.oscillator = oscillatorTO;
    } break;
    case CellType_Attacker: {
        auto const& attackerDesc = std::get<AttackerDescription>(*cellDesc.cellTypeData);
        AttackerTO attackerTO;
        attackerTO.mode = attackerDesc.mode;
        cellTO.cellTypeData.attacker = attackerTO;
    } break;
    case CellType_Injector: {
        auto const& injectorDesc = std::get<InjectorDescription>(*cellDesc.cellTypeData);
        InjectorTO injectorTO;
        injectorTO.mode = injectorDesc.mode;
        injectorTO.counter = injectorDesc.counter;
        CHECK(injectorDesc.genome.size() >= Const::GenomeHeaderSize)
        convert(dataTO, injectorDesc.genome, injectorTO.genomeSize, injectorTO.genomeDataIndex);
        injectorTO.genomeGeneration = injectorDesc.genomeGeneration;
        cellTO.cellTypeData.injector = injectorTO;
    } break;
    case CellType_Muscle: {
        auto const& muscleDesc = std::get<MuscleDescription>(*cellDesc.cellTypeData);
        MuscleTO muscleTO;
        muscleTO.mode = muscleDesc.mode;
        muscleTO.lastBendingDirection = muscleDesc.lastBendingDirection;
        muscleTO.lastBendingSourceIndex = muscleDesc.lastBendingSourceIndex;
        muscleTO.consecutiveBendingAngle = muscleDesc.consecutiveBendingAngle;
        muscleTO.lastMovementX = muscleDesc.lastMovementX;
        muscleTO.lastMovementY = muscleDesc.lastMovementY;
        cellTO.cellTypeData.muscle = muscleTO;
    } break;
    case CellType_Defender: {
        auto const& defenderDesc = std::get<DefenderDescription>(*cellDesc.cellTypeData);
        DefenderTO defenderTO;
        defenderTO.mode = defenderDesc.mode;
        cellTO.cellTypeData.defender = defenderTO;
    } break;
    case CellType_Reconnector: {
        auto const& reconnectorDesc = std::get<ReconnectorDescription>(*cellDesc.cellTypeData);
        ReconnectorTO reconnectorTO;
        reconnectorTO.restrictToColor = toUInt8(reconnectorDesc.restrictToColor.value_or(255));
        reconnectorTO.restrictToMutants = reconnectorDesc.restrictToMutants;
        cellTO.cellTypeData.reconnector = reconnectorTO;
    } break;
    case CellType_Detonator: {
        auto const& detonatorDesc = std::get<DetonatorDescription>(*cellDesc.cellTypeData);
        DetonatorTO detonatorTO;
        detonatorTO.state = detonatorDesc.state;
        detonatorTO.countdown = detonatorDesc.countdown;
        cellTO.cellTypeData.detonator = detonatorTO;
    } break;
    }
    cellTO.signalRoutingRestriction.active = cellDesc.signalRoutingRestriction.has_value();
    if (cellTO.signalRoutingRestriction.active) {
        cellTO.signalRoutingRestriction.baseAngle = cellDesc.signalRoutingRestriction->baseAngle;
        cellTO.signalRoutingRestriction.openingAngle = cellDesc.signalRoutingRestriction->openingAngle;
    }
    cellTO.signal.active = cellDesc.signal.has_value();
    if (cellTO.signal.active) {
        for (int i = 0; i < MAX_CHANNELS; ++i) {
            cellTO.signal.channels[i] = cellDesc.signal->channels[i];
        }
        cellTO.signal.origin = cellDesc.signal->origin;
        cellTO.signal.targetX = cellDesc.signal->targetX;
        cellTO.signal.targetY = cellDesc.signal->targetY;
        cellTO.signal.numPrevCells = toInt(cellDesc.signal->prevCellIds.size());
        for (int i = 0; i < cellTO.signal.numPrevCells; ++i) {
            cellTO.signal.prevCellIds[i] = cellDesc.signal->prevCellIds[i];
        }
    }
    cellTO.activationTime = cellDesc.activationTime;
    cellTO.numConnections = 0;
    cellTO.barrier = cellDesc.barrier;
    cellTO.age = cellDesc.age;
    cellTO.color = cellDesc.color;
    cellTO.genomeComplexity = cellDesc.genomeComplexity;
    convert(dataTO, cellDesc.metadata.name, cellTO.metadata.nameSize, cellTO.metadata.nameDataIndex);
    convert(dataTO, cellDesc.metadata.description, cellTO.metadata.descriptionSize, cellTO.metadata.descriptionDataIndex);
	cellIndexTOByIds.insert_or_assign(cellTO.id, cellIndex);
}

void DescriptionConverter::setConnections(DataTO const& dataTO, CellDescription const& cellToAdd, std::unordered_map<uint64_t, int> const& cellIndexByIds) const
{
    int index = 0;
    auto& cellTO = dataTO.cells[cellIndexByIds.at(cellToAdd.id)];
    float angleOffset = 0;
    for (ConnectionDescription const& connection : cellToAdd.connections) {
        if (connection.cellId != 0) {
            cellTO.connections[index].cellIndex = cellIndexByIds.at(connection.cellId);
            cellTO.connections[index].distance = connection.distance;
            cellTO.connections[index].angleFromPrevious = connection.angleFromPrevious + angleOffset;
            ++index;
            angleOffset = 0;
        } else {
            angleOffset += connection.angleFromPrevious;
        }
    }
    if (angleOffset != 0 && index > 0) {
        cellTO.connections[0].angleFromPrevious += angleOffset;
    }
    cellTO.numConnections = index;
}