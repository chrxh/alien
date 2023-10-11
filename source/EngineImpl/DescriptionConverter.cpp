#include "DescriptionConverter.h"

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

    void convert(DataTO const& dataTO, uint64_t sourceSize, uint64_t sourceIndex, std::vector<float>& target)
    {
        BytesAsFloat bytesAsFloat;
        target.resize(sourceSize / 4);
        for (uint64_t i = 0; i < sourceSize / 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                bytesAsFloat.b[j] = dataTO.auxiliaryData[sourceIndex + i * 4 + j];
            }
            target[i] = bytesAsFloat.f;
        }
    }

    template<typename Container>
    void convert(DataTO const& dataTO, Container const& source, int& targetSize, uint64_t& targetIndex)
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

    template <>
    void convert(DataTO const& dataTO, std::vector<float> const& source, int& targetSize, uint64_t& targetIndex)
    {
        BytesAsFloat bytesAsFloat;
        targetSize = source.size() * 4;
        if (targetSize > 0) {
            targetIndex = *dataTO.numAuxiliaryData;
            uint64_t size = source.size();
            for (uint64_t i = 0; i < size; ++i) {
                bytesAsFloat.f = source.at(i);
                for (int j = 0; j < 4; ++j) {
                    dataTO.auxiliaryData[targetIndex + i * 4 + j] = bytesAsFloat.b[j];
                }
            }
            (*dataTO.numAuxiliaryData) += targetSize;
        }
    }

    std::vector<float> unitWeightsAndBias(std::vector<std::vector<float>> const& weights, std::vector<float> const& bias)
    {
        std::vector<float> result(MAX_CHANNELS * MAX_CHANNELS + MAX_CHANNELS, 0);
        for (int row = 0; row < MAX_CHANNELS; ++row) {
            for (int col = 0; col < MAX_CHANNELS; ++col) {
                result[col + row * MAX_CHANNELS] = weights[row][col];
            }
        }
        for (int col = 0; col < MAX_CHANNELS; ++col) {
            result[col + MAX_CHANNELS * MAX_CHANNELS] = bias[col];
        }

        return result;
    }

    std::pair<std::vector<std::vector<float>>, std::vector<float>> splitWeightsAndBias(std::vector<float> const& weightsAndBias)
    {
        std::vector<std::vector<float>> weights(MAX_CHANNELS, std::vector<float>(MAX_CHANNELS, 0));
        for (int row = 0; row < MAX_CHANNELS; ++row) {
            for (int col = 0; col < MAX_CHANNELS; ++col) {
                weights[row][col] = weightsAndBias[col + row * MAX_CHANNELS];
            }
        }

        std::vector<float> bias;
        bias = std::vector<float>(weightsAndBias.begin() + MAX_CHANNELS * MAX_CHANNELS, weightsAndBias.end());

        return std::make_pair(weights, bias);
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
        element.cellType = static_cast<CellFunction>(static_cast<unsigned int>(cellTO.cellFunction) % CellFunction_Count);
        element.selected = cellTO.selected;
        element.executionOrderNumber = cellTO.executionOrderNumber;
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
    switch (cell.getCellFunctionType()) {
    case CellFunction_Neuron: {
        additionalDataSize += MAX_CHANNELS * (MAX_CHANNELS + 1) * sizeof(float);
    } break;
    case CellFunction_Transmitter:
        break;
    case CellFunction_Constructor:
        additionalDataSize += std::get<ConstructorDescription>(*cell.cellFunction).genome.size();
        break;
    case CellFunction_Sensor:
        break;
    case CellFunction_Nerve:
        break;
    case CellFunction_Attacker:
        break;
    case CellFunction_Injector:
        additionalDataSize += std::get<InjectorDescription>(*cell.cellFunction).genome.size();
        break;
    case CellFunction_Muscle:
        break;
    case CellFunction_Defender:
        break;
    case CellFunction_Placeholder:
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
    result.maxConnections = cellTO.maxConnections;
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
    result.inputExecutionOrderNumber = cellTO.inputExecutionOrderNumber >= 0 ? std::make_optional(cellTO.inputExecutionOrderNumber) : std::nullopt;
    result.outputBlocked = cellTO.outputBlocked;
    result.executionOrderNumber = cellTO.executionOrderNumber;
    result.barrier = cellTO.barrier;
    result.age = cellTO.age;
    result.color = cellTO.color;
    result.genomeSize = cellTO.genomeSize;

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

    switch (cellTO.cellFunction) {
    case CellFunction_Neuron: {
        NeuronDescription neuron;
        std::vector<float> weigthsAndBias;
        convert(dataTO, sizeof(float) * MAX_CHANNELS * (MAX_CHANNELS + 1), cellTO.cellFunctionData.neuron.weightsAndBiasesDataIndex, weigthsAndBias);
        std::tie(neuron.weights, neuron.biases) = splitWeightsAndBias(weigthsAndBias);
        result.cellFunction = neuron;
    } break;
    case CellFunction_Transmitter: {
        TransmitterDescription transmitter;
        transmitter.mode = cellTO.cellFunctionData.transmitter.mode;
        result.cellFunction = transmitter;
    } break;
    case CellFunction_Constructor: {
        ConstructorDescription constructor;
        constructor.activationMode = cellTO.cellFunctionData.constructor.activationMode;
        constructor.constructionActivationTime = cellTO.cellFunctionData.constructor.constructionActivationTime;
        convert(dataTO, cellTO.cellFunctionData.constructor.genomeSize, cellTO.cellFunctionData.constructor.genomeDataIndex, constructor.genome);
        constructor.genomeCurrentNodeIndex = toInt(cellTO.cellFunctionData.constructor.genomeCurrentNodeIndex);
        constructor.genomeCurrentRepetition = toInt(cellTO.cellFunctionData.constructor.genomeCurrentRepetition);
        constructor.isConstructionBuilt = toInt(cellTO.cellFunctionData.constructor.isConstructionBuilt);
        constructor.offspringCreatureId = cellTO.cellFunctionData.constructor.offspringCreatureId;
        constructor.offspringMutationId = cellTO.cellFunctionData.constructor.offspringMutationId;
        constructor.genomeGeneration = cellTO.cellFunctionData.constructor.genomeGeneration;
        constructor.constructionAngle1 = cellTO.cellFunctionData.constructor.constructionAngle1;
        constructor.constructionAngle2 = cellTO.cellFunctionData.constructor.constructionAngle2;
        result.cellFunction = constructor;
    } break;
    case CellFunction_Sensor: {
        SensorDescription sensor;
        if (cellTO.cellFunctionData.sensor.mode == SensorMode_FixedAngle) {
            sensor.fixedAngle = cellTO.cellFunctionData.sensor.angle;
        }
        sensor.minDensity = cellTO.cellFunctionData.sensor.minDensity;
        sensor.color = cellTO.cellFunctionData.sensor.color;
        sensor.targetedCreatureId = cellTO.cellFunctionData.sensor.targetedCreatureId;
        sensor.memoryChannel1 = cellTO.cellFunctionData.sensor.memoryChannel1;
        sensor.memoryChannel2 = cellTO.cellFunctionData.sensor.memoryChannel2;
        sensor.memoryChannel3 = cellTO.cellFunctionData.sensor.memoryChannel3;
        result.cellFunction = sensor;
    } break;
    case CellFunction_Nerve: {
        NerveDescription nerve;
        nerve.pulseMode = cellTO.cellFunctionData.nerve.pulseMode;
        nerve.alternationMode = cellTO.cellFunctionData.nerve.alternationMode;
        result.cellFunction = nerve;
    } break;
    case CellFunction_Attacker: {
        AttackerDescription attacker;
        attacker.mode = cellTO.cellFunctionData.attacker.mode;
        result.cellFunction = attacker;
    } break;
    case CellFunction_Injector: {
        InjectorDescription injector;
        injector.mode = cellTO.cellFunctionData.injector.mode;
        injector.counter = cellTO.cellFunctionData.injector.counter;
        convert(dataTO, cellTO.cellFunctionData.injector.genomeSize, cellTO.cellFunctionData.injector.genomeDataIndex, injector.genome);
        injector.genomeGeneration = cellTO.cellFunctionData.injector.genomeGeneration;
        result.cellFunction = injector;
    } break;
    case CellFunction_Muscle: {
        MuscleDescription muscle;
        muscle.mode = cellTO.cellFunctionData.muscle.mode;
        muscle.lastBendingDirection = cellTO.cellFunctionData.muscle.lastBendingDirection;
        muscle.lastBendingSourceIndex = cellTO.cellFunctionData.muscle.lastBendingSourceIndex;
        muscle.consecutiveBendingAngle = cellTO.cellFunctionData.muscle.consecutiveBendingAngle;
        result.cellFunction = muscle;
    } break;
    case CellFunction_Defender: {
        DefenderDescription defender;
        defender.mode = cellTO.cellFunctionData.defender.mode;
        result.cellFunction = defender;
    } break;
    case CellFunction_Placeholder: {
        PlaceHolderDescription placeHolder;
        result.cellFunction = placeHolder;
    } break;
    }

    for (int i = 0; i < MAX_CHANNELS; ++i) {
        result.activity.channels[i] = cellTO.activity.channels[i];
    }
    result.activationTime = cellTO.activationTime;
    return result;
}

void DescriptionConverter::addParticle(DataTO const& dataTO, ParticleDescription const& particleDesc) const
{
    auto particleIndex = (*dataTO.numParticles)++;

	ParticleTO& particleTO = dataTO.particles[particleIndex];
	particleTO.id = particleDesc.id == 0 ? NumberGenerator::getInstance().getId() : particleDesc.id;
    particleTO.pos = {particleDesc.pos.x, particleDesc.pos.y};
    particleTO.vel = {particleDesc.vel.x, particleDesc.vel.y};
	particleTO.energy = particleDesc.energy;
    particleTO.color = particleDesc.color;
}

void DescriptionConverter::addCell(
    DataTO const& dataTO, CellDescription const& cellDesc, std::unordered_map<uint64_t, int>& cellIndexTOByIds) const
{
    int cellIndex = (*dataTO.numCells)++;
    CellTO& cellTO = dataTO.cells[cellIndex];
    cellTO.id = cellDesc.id == 0 ? NumberGenerator::getInstance().getId() : cellDesc.id;
	cellTO.pos= { cellDesc.pos.x, cellDesc.pos.y };
    cellTO.vel = {cellDesc.vel.x, cellDesc.vel.y};
    cellTO.energy = cellDesc.energy;
    cellTO.stiffness = cellDesc.stiffness;
    cellTO.maxConnections = cellDesc.maxConnections;
    cellTO.executionOrderNumber = cellDesc.executionOrderNumber;
    cellTO.livingState = cellDesc.livingState;
    cellTO.creatureId = cellDesc.creatureId;
    cellTO.mutationId = cellDesc.mutationId;
    cellTO.inputExecutionOrderNumber = cellDesc.inputExecutionOrderNumber.value_or(-1);
    cellTO.outputBlocked = cellDesc.outputBlocked;
    cellTO.cellFunction = cellDesc.getCellFunctionType();
    switch (cellDesc.getCellFunctionType()) {
    case CellFunction_Neuron: {
        NeuronTO neuronTO;
        auto const& neuronDesc = std::get<NeuronDescription>(*cellDesc.cellFunction);
        std::vector<float> weigthsAndBias = unitWeightsAndBias(neuronDesc.weights, neuronDesc.biases);
        int targetSize;
        convert(dataTO, weigthsAndBias, targetSize, neuronTO.weightsAndBiasesDataIndex);
        CHECK(targetSize == sizeof(float) * MAX_CHANNELS * (MAX_CHANNELS + 1));
        cellTO.cellFunctionData.neuron = neuronTO;
    } break;
    case CellFunction_Transmitter: {
        auto const& transmitterDesc = std::get<TransmitterDescription>(*cellDesc.cellFunction);
        TransmitterTO transmitterTO;
        transmitterTO.mode = transmitterDesc.mode;
        cellTO.cellFunctionData.transmitter = transmitterTO;
    } break;
    case CellFunction_Constructor: {
        auto const& constructorDesc = std::get<ConstructorDescription>(*cellDesc.cellFunction);
        ConstructorTO constructorTO;
        constructorTO.activationMode = constructorDesc.activationMode;
        constructorTO.constructionActivationTime = constructorDesc.constructionActivationTime;
        CHECK(constructorDesc.genome.size() >= Const::GenomeHeaderSize)
        convert(dataTO, constructorDesc.genome, constructorTO.genomeSize, constructorTO.genomeDataIndex);
        constructorTO.genomeCurrentNodeIndex = constructorDesc.genomeCurrentNodeIndex;
        constructorTO.genomeCurrentRepetition = constructorDesc.genomeCurrentRepetition;
        constructorTO.isConstructionBuilt = constructorDesc.isConstructionBuilt;
        constructorTO.offspringCreatureId = constructorDesc.offspringCreatureId;
        constructorTO.offspringMutationId = constructorDesc.offspringMutationId;
        constructorTO.genomeGeneration = constructorDesc.genomeGeneration;
        constructorTO.constructionAngle1 = constructorDesc.constructionAngle1;
        constructorTO.constructionAngle2 = constructorDesc.constructionAngle2;
        cellTO.cellFunctionData.constructor = constructorTO;
    } break;
    case CellFunction_Sensor: {
        auto const& sensorDesc = std::get<SensorDescription>(*cellDesc.cellFunction);
        SensorTO sensorTO;
        sensorTO.mode = sensorDesc.getSensorMode();
        sensorTO.color = sensorDesc.color;
        sensorTO.minDensity = sensorDesc.minDensity;
        sensorTO.angle = sensorDesc.fixedAngle.value_or(0);
        sensorTO.targetedCreatureId = sensorDesc.targetedCreatureId;
        sensorTO.memoryChannel1 = sensorDesc.memoryChannel1;
        sensorTO.memoryChannel2 = sensorDesc.memoryChannel2;
        sensorTO.memoryChannel3 = sensorDesc.memoryChannel3;
        cellTO.cellFunctionData.sensor = sensorTO;
    } break;
    case CellFunction_Nerve: {
        auto const& nerveDesc = std::get<NerveDescription>(*cellDesc.cellFunction);
        NerveTO nerveTO;
        nerveTO.pulseMode = nerveDesc.pulseMode;
        nerveTO.alternationMode = nerveDesc.alternationMode;
        cellTO.cellFunctionData.nerve = nerveTO;
    } break;
    case CellFunction_Attacker: {
        auto const& attackerDesc = std::get<AttackerDescription>(*cellDesc.cellFunction);
        AttackerTO attackerTO;
        attackerTO.mode = attackerDesc.mode;
        cellTO.cellFunctionData.attacker = attackerTO;
    } break;
    case CellFunction_Injector: {
        auto const& injectorDesc = std::get<InjectorDescription>(*cellDesc.cellFunction);
        InjectorTO injectorTO;
        injectorTO.mode = injectorDesc.mode;
        injectorTO.counter = injectorDesc.counter;
        CHECK(injectorDesc.genome.size() >= Const::GenomeHeaderSize)
        convert(dataTO, injectorDesc.genome, injectorTO.genomeSize, injectorTO.genomeDataIndex);
        injectorTO.genomeGeneration = injectorDesc.genomeGeneration;
        cellTO.cellFunctionData.injector = injectorTO;
    } break;
    case CellFunction_Muscle: {
        auto const& muscleDesc = std::get<MuscleDescription>(*cellDesc.cellFunction);
        MuscleTO muscleTO;
        muscleTO.mode = muscleDesc.mode;
        muscleTO.lastBendingDirection = muscleDesc.lastBendingDirection;
        muscleTO.lastBendingSourceIndex = muscleDesc.lastBendingSourceIndex;
        muscleTO.consecutiveBendingAngle = muscleDesc.consecutiveBendingAngle;
        cellTO.cellFunctionData.muscle = muscleTO;
    } break;
    case CellFunction_Defender: {
        auto const& defenderDesc = std::get<DefenderDescription>(*cellDesc.cellFunction);
        DefenderTO defenderTO;
        defenderTO.mode = defenderDesc.mode;
        cellTO.cellFunctionData.defender = defenderTO;
    } break;
    case CellFunction_Placeholder: {
        PlaceHolderTO placeHolderTO;
        cellTO.cellFunctionData.placeHolder = placeHolderTO;
    } break;
    }
    for (int i = 0; i < MAX_CHANNELS; ++i) {
        cellTO.activity.channels[i] = cellDesc.activity.channels[i];
    }
    cellTO.activationTime = cellDesc.activationTime;
    cellTO.numConnections = 0;
    cellTO.barrier = cellDesc.barrier;
    cellTO.age = cellDesc.age;
    cellTO.color = cellDesc.color;
    cellTO.genomeSize = cellDesc.genomeSize;
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