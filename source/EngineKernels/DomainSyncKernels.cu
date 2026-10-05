#include "DomainSyncKernels.cuh"

#include "ConstructorHelper.cuh"
#include "DetonatorProcessor.cuh"
#include "DomainOpProcessor.cuh"
#include "ObjectConnectionProcessor.cuh"
#include "SensorProcessor.cuh"

namespace
{
    auto constexpr NoId = VALUE_NOT_SET_UINT64;
    auto constexpr NoReferenceKey = 0xffffffffffffffffull;

    __device__ __inline__ bool isOwnedCell(Object* object)
    {
        return object->type == ObjectType_Cell && !object->isGhost();
    }

    // Head cells take precedence, then the cell with the lowest id decides where a creature belongs
    __device__ __inline__ uint64_t calcReferenceKey(Object* cell)
    {
        auto priority = cell->typeData.cell.headCell ? 0ull : (1ull << 63);
        return priority | (cell->id & ~(1ull << 63));
    }

    __device__ __inline__ int decideOwnerByPosition(SimulationData const& data, float x, bool initialAssignment)
    {
        auto const& domain = data.domain;
        if (!initialAssignment && domain.getDistanceToStrip(domain.index, x) <= domain.layout->ownershipHysteresis) {
            return domain.index;
        }
        return domain.getStripOwner(x);
    }

    // A creature under construction belongs to the root of its chain of constructing creatures, since a creature can construct
    // offspring while it is still being constructed itself
    __device__ __inline__ int decideCreatureOwner(SimulationData const& data, Creature* creature, bool initialAssignment)
    {
        auto constexpr MaxChainLength = 16;
        auto reference = creature;
        for (int i = 0; i < MaxChainLength && reference->constructingCreature; ++i) {
            reference = reference->constructingCreature;
        }
        return decideOwnerByPosition(data, reference->referencePos.x, initialAssignment);
    }

    // A cell of a creature owned elsewhere is handed over to the owner of the creature
    __device__ __inline__ int decideObjectOwner(SimulationData const& data, Object* object, bool initialAssignment)
    {
        if (object->type == ObjectType_Cell) {
            auto creature = object->typeData.cell.creature;
            return creature->isReplica ? creature->ownerDomain : creature->newOwnerDomain;
        }
        return decideOwnerByPosition(data, object->pos.x, initialAssignment);
    }

    __device__ __inline__ uint64_t alignTo16(uint64_t value)
    {
        return (value + 15) / 16 * 16;
    }

    __device__ __inline__ bool hasSignalEntries(Node const& node)
    {
        return node.cellType == CellType_Memory && node.cellTypeData.memory.numSignalEntries > 0 && node.cellTypeData.memory.signalEntries != nullptr;
    }

    __device__ __inline__ uint64_t calcSerializedGenomeSize(Genome const& genome)
    {
        auto result = alignTo16(sizeof(Genome)) + alignTo16(sizeof(Gene) * genome.numGenes);
        for (int i = 0; i < genome.numGenes; ++i) {
            auto const& gene = genome.genes[i];
            result += alignTo16(sizeof(Node) * gene.numNodes);
            for (int j = 0; j < gene.numNodes; ++j) {
                auto const& node = gene.nodes[j];
                if (hasSignalEntries(node)) {
                    result += alignTo16(sizeof(SignalEntryGenome) * node.cellTypeData.memory.numSignalEntries);
                }
            }
        }
        return result;
    }

    template <typename T>
    __device__ __inline__ T* offsetToPointer(uint64_t offset)
    {
        return reinterpret_cast<T*>(offset);
    }

    template <typename T>
    __device__ __inline__ uint64_t pointerToOffset(T* pointer)
    {
        return reinterpret_cast<uint64_t>(pointer);
    }

    template <typename T>
    __device__ __inline__ void copyElements(T* target, T const* source, uint64_t numElements)
    {
        for (uint64_t i = 0; i < numElements; ++i) {
            target[i] = source[i];
        }
    }

    __device__ __inline__ void serializeGenome(SyncMessage const& message, Genome* genome)
    {
        auto numBytes = calcSerializedGenomeSize(*genome);
        auto byteIndex = message.reserve(&message.counters->numGenomeBytes, message.capacities.genomeBytes, numBytes);
        if (byteIndex < 0) {
            return;
        }
        auto entryIndex = message.reserve(&message.counters->numGenomeEntries, message.capacities.genomeEntries, 1);
        if (entryIndex < 0) {
            return;
        }
        auto block = message.genomeBytes + byteIndex;
        uint64_t offset = 0;
        auto takeBytes = [&](uint64_t size) {
            auto result = offset;
            offset += alignTo16(size);
            return result;
        };

        auto genomeOffset = takeBytes(sizeof(Genome));
        auto genesOffset = takeBytes(sizeof(Gene) * genome->numGenes);
        auto serializedGenome = reinterpret_cast<Genome*>(block + genomeOffset);
        *serializedGenome = *genome;
        serializedGenome->genes = offsetToPointer<Gene>(genesOffset);

        auto serializedGenes = reinterpret_cast<Gene*>(block + genesOffset);
        for (int i = 0; i < genome->numGenes; ++i) {
            auto const& gene = genome->genes[i];
            auto nodesOffset = takeBytes(sizeof(Node) * gene.numNodes);
            serializedGenes[i] = gene;
            serializedGenes[i].nodes = offsetToPointer<Node>(nodesOffset);

            auto serializedNodes = reinterpret_cast<Node*>(block + nodesOffset);
            for (int j = 0; j < gene.numNodes; ++j) {
                auto const& node = gene.nodes[j];
                serializedNodes[j] = node;
                if (hasSignalEntries(node)) {
                    auto numEntries = node.cellTypeData.memory.numSignalEntries;
                    auto entriesOffset = takeBytes(sizeof(SignalEntryGenome) * numEntries);
                    copyElements(reinterpret_cast<SignalEntryGenome*>(block + entriesOffset), node.cellTypeData.memory.signalEntries, numEntries);
                    serializedNodes[j].cellTypeData.memory.signalEntries = offsetToPointer<SignalEntryGenome>(entriesOffset);
                } else if (node.cellType == CellType_Memory) {
                    serializedNodes[j].cellTypeData.memory.signalEntries = nullptr;
                }
            }
        }
        message.genomeEntries[entryIndex] = GenomeEntry{.genomeId = genome->id, .byteOffset = static_cast<uint64_t>(byteIndex), .numBytes = numBytes};
    }

    // Fills target, which is either a new genome or a placeholder, from a serialized genome
    __device__ __inline__ void deserializeGenome(SimulationData& data, Genome* target, uint8_t const* block)
    {
        auto const& serializedGenome = *reinterpret_cast<Genome const*>(block);
        auto genes = data.entities.heap.getTypedSubArray<Gene>(serializedGenome.numGenes);
        auto serializedGenes = reinterpret_cast<Gene const*>(block + pointerToOffset(serializedGenome.genes));
        for (int i = 0; i < serializedGenome.numGenes; ++i) {
            auto const& serializedGene = serializedGenes[i];
            genes[i] = serializedGene;
            auto nodes = data.entities.heap.getTypedSubArray<Node>(serializedGene.numNodes);
            auto serializedNodes = reinterpret_cast<Node const*>(block + pointerToOffset(serializedGene.nodes));
            for (int j = 0; j < serializedGene.numNodes; ++j) {
                auto const& serializedNode = serializedNodes[j];
                nodes[j] = serializedNode;
                if (serializedNode.cellType == CellType_Memory) {
                    auto numEntries = serializedNode.cellTypeData.memory.numSignalEntries;
                    if (numEntries > 0 && serializedNode.cellTypeData.memory.signalEntries != nullptr) {
                        auto entries = data.entities.heap.getTypedSubArray<SignalEntryGenome>(numEntries);
                        copyElements(
                            entries,
                            reinterpret_cast<SignalEntryGenome const*>(block + pointerToOffset(serializedNode.cellTypeData.memory.signalEntries)),
                            numEntries);
                        nodes[j].cellTypeData.memory.signalEntries = entries;
                    } else {
                        nodes[j].cellTypeData.memory.signalEntries = nullptr;
                    }
                }
            }
            genes[i].nodes = nodes;
        }
        auto genomeId = serializedGenome.id;
        auto genomeIndex = target->genomeIndex;
        *target = serializedGenome;
        target->id = genomeId;
        target->genomeIndex = genomeIndex;
        target->genes = genes;
        target->isPlaceholder = false;
    }

    __device__ __inline__ Genome* createPlaceholderGenome(SimulationData& data, uint64_t genomeId)
    {
        auto genome = data.entities.heap.getTypedSubArray<Genome>(1);
        *genome = Genome{};
        genome->id = genomeId;
        genome->numGenes = 0;
        genome->genes = nullptr;
        genome->isPlaceholder = true;
        genome->genomeIndex = NoId;
        return genome;
    }

    template <typename T>
    __device__ __inline__ void zeroBytes(T* target, uint64_t numElements)
    {
        auto bytes = reinterpret_cast<uint8_t*>(target);
        for (uint64_t i = 0; i < sizeof(T) * numElements; ++i) {
            bytes[i] = 0;
        }
    }

    __device__ __inline__ void fillObjectRecord(ObjectRecord& record, Object* object, int newOwner)
    {
        record.object = *object;
        record.object.ownerDomain = static_cast<uint8_t>(newOwner);
        for (int i = 0; i < object->numConnections; ++i) {
            record.connectionIds[i] = object->connections[i].object->id;
        }
        record.creatureId = object->type == ObjectType_Cell ? object->typeData.cell.creature->id : NoId;
        record.transferPayloadIndex = -1;
    }

    __device__ __inline__ void packCreatureIfNecessary(SyncMessage const& message, Creature* creature, int receiver, int newOwner, bool transfer)
    {
        auto bit = 1u << receiver;
        if (atomicOr(&creature->packedForDomains, bit) & bit) {
            return;
        }
        auto index = message.reserve(&message.counters->numCreatures, message.capacities.creatures, 1);
        if (index < 0) {
            return;
        }
        auto& record = message.creatures[index];
        record.creature = *creature;
        record.creature.ownerDomain = static_cast<uint8_t>(newOwner);
        record.genomeId = creature->genome->id;
        record.transfer = transfer;
    }
}

/************************************************************************/
/* Ownership                                                            */
/************************************************************************/

__global__ void cudaDomainOwnership_resetCreatures(SimulationData data)
{
    auto& objects = data.entities.objects;
    auto const partition = calcSystemThreadPartition(objects.getNumEntries());
    for (int index = partition.startIndex; index <= partition.endIndex; index += partition.step) {
        auto object = objects.at(index);
        if (!isOwnedCell(object)) {
            continue;
        }
        auto creature = object->typeData.cell.creature;
        creature->referenceKey = NoReferenceKey;
        creature->constructingCreatureId = NoId;
        creature->constructingCreature = nullptr;
        creature->packedForDomains = 0;
    }
}

__global__ void cudaDomainOwnership_calcReferenceKeys(SimulationData data)
{
    auto& objects = data.entities.objects;
    auto const partition = calcSystemThreadPartition(objects.getNumEntries());
    for (int index = partition.startIndex; index <= partition.endIndex; index += partition.step) {
        auto object = objects.at(index);
        if (!isOwnedCell(object)) {
            continue;
        }
        alienAtomicMin64(&object->typeData.cell.creature->referenceKey, calcReferenceKey(object));
    }
}

__global__ void cudaDomainOwnership_calcReferencePositions(SimulationData data)
{
    auto& objects = data.entities.objects;
    auto const partition = calcSystemThreadPartition(objects.getNumEntries());
    for (int index = partition.startIndex; index <= partition.endIndex; index += partition.step) {
        auto object = objects.at(index);
        if (!isOwnedCell(object)) {
            continue;
        }
        auto creature = object->typeData.cell.creature;
        if (calcReferenceKey(object) == creature->referenceKey) {
            creature->referencePos = object->pos;
        }
    }
}

namespace
{
    // Returns the creature under construction that the constructor builds, or nullptr
    __device__ __inline__ Creature* getOffspringUnderConstruction(Object* object)
    {
        if (!isOwnedCell(object) || !object->typeData.cell.constructorAvailable) {
            return nullptr;
        }
        auto lastConstructedCell = ConstructorHelper::getLastConstructedCell(object);
        if (!lastConstructedCell || lastConstructedCell->type != ObjectType_Cell || lastConstructedCell->isGhost()) {
            return nullptr;
        }
        auto offspring = lastConstructedCell->typeData.cell.creature;
        if (offspring == object->typeData.cell.creature || lastConstructedCell->typeData.cell.cellState != CellState_UnderConstruction) {
            return nullptr;
        }
        return offspring;
    }
}

__global__ void cudaDomainOwnership_findConstructingCreatureIds(SimulationData data)
{
    auto& objects = data.entities.objects;
    auto const partition = calcSystemThreadPartition(objects.getNumEntries());
    for (int index = partition.startIndex; index <= partition.endIndex; index += partition.step) {
        auto object = objects.at(index);
        if (auto offspring = getOffspringUnderConstruction(object)) {
            alienAtomicMin64(&offspring->constructingCreatureId, object->typeData.cell.creature->id);
        }
    }
}

__global__ void cudaDomainOwnership_findConstructingCreatures(SimulationData data)
{
    auto& objects = data.entities.objects;
    auto const partition = calcSystemThreadPartition(objects.getNumEntries());
    for (int index = partition.startIndex; index <= partition.endIndex; index += partition.step) {
        auto object = objects.at(index);
        auto offspring = getOffspringUnderConstruction(object);
        if (offspring && offspring->constructingCreatureId == object->typeData.cell.creature->id) {
            offspring->constructingCreature = object->typeData.cell.creature;
        }
    }
}

__global__ void cudaDomainOwnership_decideCreatureOwners(SimulationData data, bool initialAssignment)
{
    auto& objects = data.entities.objects;
    auto const partition = calcSystemThreadPartition(objects.getNumEntries());
    for (int index = partition.startIndex; index <= partition.endIndex; index += partition.step) {
        auto object = objects.at(index);
        if (!isOwnedCell(object)) {
            continue;
        }
        auto creature = object->typeData.cell.creature;
        creature->newOwnerDomain = static_cast<uint8_t>(decideCreatureOwner(data, creature, initialAssignment));
    }
}

/************************************************************************/
/* Region of interest                                                   */
/************************************************************************/

__global__ void cudaDomainRoi_clearBaseBitmap(SimulationData data, DomainSyncData syncData)
{
    auto const partition = calcSystemThreadPartition(data.domain.layout->numBitmapWords);
    for (int index = partition.startIndex; index <= partition.endIndex; index += partition.step) {
        syncData.baseBitmap[index] = 0;
    }
}

__global__ void cudaDomainRoi_markOwnedObjects(SimulationData data, DomainSyncData syncData)
{
    auto& objects = data.entities.objects;
    auto const partition = calcSystemThreadPartition(objects.getNumEntries());
    for (int index = partition.startIndex; index <= partition.endIndex; index += partition.step) {
        auto object = objects.at(index);
        if (object->isGhost()) {
            continue;
        }
        auto tileIndex = data.domain.getTileIndex(object->pos);
        auto bit = 1u << (tileIndex % 32);
        auto& word = syncData.baseBitmap[tileIndex / 32];
        if (!(word & bit)) {
            atomicOr(&word, bit);
        }
    }
}

// Each thread computes one word of the dilated bitmap, so no atomics are needed
__global__ void cudaDomainRoi_dilate(SimulationData data, DomainSyncData syncData)
{
    auto const& domain = data.domain;
    auto const& layout = *domain.layout;
    auto radius = static_cast<int>(ceilf(layout.haloWidth / DomainLayout::TileSize)) + 1;
    auto stripReach = layout.haloWidth + DomainLayout::TileSize;
    auto ownBitmap = domain.getRoiBitmap(domain.index);

    auto const partition = calcSystemThreadPartition(layout.numBitmapWords);
    for (int wordIndex = partition.startIndex; wordIndex <= partition.endIndex; wordIndex += partition.step) {
        uint32_t word = 0;
        for (int bitIndex = 0; bitIndex < 32; ++bitIndex) {
            auto tileIndex = wordIndex * 32 + bitIndex;
            if (tileIndex >= layout.numTiles.x * layout.numTiles.y) {
                break;
            }
            auto tileX = tileIndex % layout.numTiles.x;
            auto tileY = tileIndex / layout.numTiles.x;
            auto tileCenterX = (toFloat(tileX) + 0.5f) * DomainLayout::TileSize;
            auto inRoi = domain.getDistanceToStrip(domain.index, tileCenterX) <= stripReach;
            for (int dy = -radius; dy <= radius && !inRoi; ++dy) {
                for (int dx = -radius; dx <= radius && !inRoi; ++dx) {
                    auto x = ((tileX + dx) % layout.numTiles.x + layout.numTiles.x) % layout.numTiles.x;
                    auto y = ((tileY + dy) % layout.numTiles.y + layout.numTiles.y) % layout.numTiles.y;
                    auto otherTileIndex = x + y * layout.numTiles.x;
                    if ((syncData.baseBitmap[otherTileIndex / 32] >> (otherTileIndex % 32)) & 1) {
                        inRoi = true;
                    }
                }
            }
            if (inRoi) {
                word |= 1u << bitIndex;
            }
        }
        ownBitmap[wordIndex] = word;
    }
}

/************************************************************************/
/* Id maps                                                              */
/************************************************************************/

__global__ void cudaDomainMaps_clear(DomainSyncData syncData)
{
    syncData.objectMap.clear_system();
    syncData.particleMap.clear_system();
    syncData.creatureMap.clear_system();
    syncData.genomeMap.clear_system();
}

__global__ void cudaDomainMaps_insert(SimulationData data, DomainSyncData syncData)
{
    {
        auto& objects = data.entities.objects;
        auto const partition = calcSystemThreadPartition(objects.getNumEntries());
        for (int index = partition.startIndex; index <= partition.endIndex; index += partition.step) {
            auto object = objects.at(index);
            syncData.objectMap.insert(object->id, object);
            if (object->type == ObjectType_Cell) {
                auto creature = object->typeData.cell.creature;
                syncData.creatureMap.insert(creature->id, creature);
                syncData.genomeMap.insert(creature->genome->id, creature->genome);
            }
        }
    }
    {
        auto& particles = data.entities.energies;
        auto const partition = calcSystemThreadPartition(particles.getNumEntries());
        for (int index = partition.startIndex; index <= partition.endIndex; index += partition.step) {
            auto particle = particles.at(index);
            syncData.particleMap.insert(particle->id, particle);
        }
    }
}

/************************************************************************/
/* Packing                                                              */
/************************************************************************/

__global__ void cudaDomainPack_reset(SimulationData data, DomainSyncData syncData)
{
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        for (int receiver = 0; receiver < data.domain.numDomains; ++receiver) {
            if (receiver == data.domain.index) {
                continue;
            }
            auto& counters = *syncData.outgoingMessages[receiver].counters;
            counters = SyncMessageCounters{};
        }
    }
    auto& objects = data.entities.objects;
    auto const partition = calcSystemThreadPartition(objects.getNumEntries());
    for (int index = partition.startIndex; index <= partition.endIndex; index += partition.step) {
        auto object = objects.at(index);
        if (isOwnedCell(object)) {
            object->typeData.cell.creature->packedForDomains = 0;
        }
    }
}

__global__ void cudaDomainPack_roiBitmaps(SimulationData data, DomainSyncData syncData)
{
    auto const& domain = data.domain;
    auto ownBitmap = domain.getRoiBitmap(domain.index);
    auto numWords = domain.layout->numBitmapWords;
    for (int receiver = 0; receiver < domain.numDomains; ++receiver) {
        if (receiver == domain.index) {
            continue;
        }
        auto const& message = syncData.outgoingMessages[receiver];
        auto const partition = calcSystemThreadPartition(numWords);
        for (int index = partition.startIndex; index <= partition.endIndex; index += partition.step) {
            message.roiBitmap[index] = ownBitmap[index];
        }
    }
}

__global__ void cudaDomainPack_objects(SimulationData data, DomainSyncData syncData)
{
    auto const& domain = data.domain;
    auto& objects = data.entities.objects;
    auto const partition = calcSystemThreadPartition(objects.getNumEntries());
    for (int index = partition.startIndex; index <= partition.endIndex; index += partition.step) {
        auto object = objects.at(index);
        if (object->isGhost()) {
            continue;
        }
        auto newOwner = decideObjectOwner(data, object, false);
        auto transfer = newOwner != domain.index;

        for (int receiver = 0; receiver < domain.numDomains; ++receiver) {
            if (receiver == domain.index) {
                continue;
            }
            auto isTransferReceiver = transfer && receiver == newOwner;
            if (!isTransferReceiver && !domain.isInRoi(receiver, object->pos)) {
                continue;
            }
            auto const& message = syncData.outgoingMessages[receiver];
            auto recordIndex = message.reserve(&message.counters->numObjects, message.capacities.objects, 1);
            if (recordIndex < 0) {
                continue;
            }
            auto& record = message.objects[recordIndex];
            fillObjectRecord(record, object, newOwner);

            if (isTransferReceiver) {
                auto payloadIndex = message.reserve(&message.counters->numTransferPayloads, message.capacities.transferPayloads, 1);
                if (payloadIndex < 0) {
                    continue;
                }
                record.transferPayloadIndex = payloadIndex;
                if (object->type == ObjectType_Cell) {
                    auto& payload = message.transferPayloads[payloadIndex];
                    payload.neuralNet = *object->typeData.cell.neuralNetwork;
                    if (object->typeData.cell.cellType == CellType_Memory) {
                        auto const& memory = object->typeData.cell.cellTypeData.memory;
                        copyElements(payload.memoryEntries, memory.signalEntries, memory.numSignalEntries);
                    }
                }
            }
            if (object->type == ObjectType_Cell) {
                packCreatureIfNecessary(message, object->typeData.cell.creature, receiver, newOwner, isTransferReceiver);
            }
        }
    }
}

__global__ void cudaDomainPack_particles(SimulationData data, DomainSyncData syncData)
{
    auto const& domain = data.domain;
    auto& particles = data.entities.energies;
    auto const partition = calcSystemThreadPartition(particles.getNumEntries());
    for (int index = partition.startIndex; index <= partition.endIndex; index += partition.step) {
        auto particle = particles.at(index);
        if (particle->ghost) {
            continue;
        }
        auto newOwner = decideOwnerByPosition(data, particle->pos.x, false);
        for (int receiver = 0; receiver < domain.numDomains; ++receiver) {
            if (receiver == domain.index) {
                continue;
            }
            auto transfer = receiver == newOwner;
            if (!transfer && !domain.isInRoi(receiver, particle->pos)) {
                continue;
            }
            auto const& message = syncData.outgoingMessages[receiver];
            auto recordIndex = message.reserve(&message.counters->numParticles, message.capacities.particles, 1);
            if (recordIndex < 0) {
                continue;
            }
            auto& record = message.particles[recordIndex];
            record.particle = *particle;
            record.particle.ownerDomain = static_cast<uint8_t>(newOwner);
            record.transfer = transfer;
        }
    }
}

__global__ void cudaDomainPack_genomeRequests(SimulationData data, DomainSyncData syncData)
{
    auto const& domain = data.domain;
    for (int receiver = 0; receiver < domain.numDomains; ++receiver) {
        if (receiver == domain.index) {
            continue;
        }
        auto const& message = syncData.outgoingMessages[receiver];
        auto numRequests = min(syncData.numGenomeRequestsToSend[receiver], DomainSyncData::MaxGenomeRequests);
        auto const partition = calcSystemThreadPartition(numRequests);
        for (int index = partition.startIndex; index <= partition.endIndex; index += partition.step) {
            auto requestIndex = message.reserve(&message.counters->numGenomeRequests, message.capacities.genomeRequests, 1);
            if (requestIndex >= 0) {
                message.genomeRequests[requestIndex] = syncData.genomeRequestsToSend[receiver * DomainSyncData::MaxGenomeRequests + index];
            }
        }
    }
}

__global__ void cudaDomainPack_requestedGenomes(SimulationData data, DomainSyncData syncData)
{
    auto const& domain = data.domain;
    for (int receiver = 0; receiver < domain.numDomains; ++receiver) {
        if (receiver == domain.index) {
            continue;
        }
        auto const& message = syncData.outgoingMessages[receiver];
        auto numRequests = min(syncData.numGenomesToServe[receiver], DomainSyncData::MaxGenomeRequests);
        auto const partition = calcSystemThreadPartition(numRequests);
        for (int index = partition.startIndex; index <= partition.endIndex; index += partition.step) {
            auto genomeId = syncData.genomesToServe[receiver * DomainSyncData::MaxGenomeRequests + index];
            auto genome = syncData.genomeMap.find(genomeId);
            if (genome && !genome->isPlaceholder) {
                serializeGenome(message, genome);
            }
        }
    }
}

// The receiver of a creature becomes its owner and therefore needs its genome right away
__global__ void cudaDomainPack_transferredGenomes(SimulationData data, DomainSyncData syncData)
{
    auto const& domain = data.domain;
    for (int receiver = 0; receiver < domain.numDomains; ++receiver) {
        if (receiver == domain.index) {
            continue;
        }
        auto const& message = syncData.outgoingMessages[receiver];
        auto numCreatures = min(static_cast<uint64_t>(message.counters->numCreatures), message.capacities.creatures);
        auto const partition = calcSystemThreadPartition(numCreatures);
        for (int index = partition.startIndex; index <= partition.endIndex; index += partition.step) {
            auto const& record = message.creatures[index];
            if (record.transfer && !record.creature.genome->isPlaceholder) {
                serializeGenome(message, record.creature.genome);
            }
        }
    }
}

__global__ void cudaDomainPack_ops(SimulationData data, DomainSyncData syncData)
{
    auto& outbox = data.domainOps;
    auto const partition = calcSystemThreadPartition(outbox.getNumEntries());
    for (int index = partition.startIndex; index <= partition.endIndex; index += partition.step) {
        auto const& op = outbox.at(index);
        if (op.targetDomain == data.domain.index) {
            continue;
        }
        auto const& message = syncData.outgoingMessages[op.targetDomain];
        auto opIndex = message.reserve(&message.counters->numOps, message.capacities.ops, 1);
        if (opIndex >= 0) {
            message.ops[opIndex] = op;
        }
    }
}

__global__ void cudaDomainPack_sensorScans(SimulationData data, DomainSyncData syncData)
{
    {
        auto& requests = data.sensorScanRequests;
        auto const partition = calcSystemThreadPartition(requests.getNumEntries());
        for (int index = partition.startIndex; index <= partition.endIndex; index += partition.step) {
            auto const& request = requests.at(index);
            auto const& message = syncData.outgoingMessages[request.targetDomain];
            auto requestIndex = message.reserve(&message.counters->numSensorScanRequests, message.capacities.sensorScanRequests, 1);
            if (requestIndex >= 0) {
                message.sensorScanRequests[requestIndex] = request;
            }
        }
    }
    {
        auto& responses = data.sensorScanResponses;
        auto const partition = calcSystemThreadPartition(responses.getNumEntries());
        for (int index = partition.startIndex; index <= partition.endIndex; index += partition.step) {
            auto const& response = responses.at(index);
            auto const& message = syncData.outgoingMessages[response.targetDomain];
            auto responseIndex = message.reserve(&message.counters->numSensorScanResponses, message.capacities.sensorScanResponses, 1);
            if (responseIndex >= 0) {
                message.sensorScanResponses[responseIndex] = response;
            }
        }
    }
}

/************************************************************************/
/* Committing                                                           */
/************************************************************************/

__global__ void cudaDomainCommit_objects(SimulationData data, uint8_t round)
{
    auto const& domain = data.domain;
    auto& objects = data.entities.objects;
    auto const partition = calcSystemThreadPartition(objects.getNumEntries());
    for (int index = partition.startIndex; index <= partition.endIndex; index += partition.step) {
        auto object = objects.at(index);
        if (object->isGhost()) {
            continue;
        }
        auto newOwner = decideObjectOwner(data, object, false);
        if (newOwner == domain.index) {
            continue;
        }
        object->setGhost(true);
        object->ownerDomain = static_cast<uint8_t>(newOwner);
        object->syncRound = round;
        if (object->type == ObjectType_Cell && object->typeData.cell.constructorAvailable) {
            object->typeData.cell.constructor.offspring = nullptr;
        }
    }
}

// Separated from the handover of the cells, since a cell that sees its creature half updated would stay with the wrong domain
__global__ void cudaDomainCommit_creatures(SimulationData data)
{
    auto const& domain = data.domain;
    auto& objects = data.entities.objects;
    auto const partition = calcSystemThreadPartition(objects.getNumEntries());
    for (int index = partition.startIndex; index <= partition.endIndex; index += partition.step) {
        auto object = objects.at(index);
        if (object->type != ObjectType_Cell) {
            continue;
        }
        auto creature = object->typeData.cell.creature;
        if (!creature->isReplica && creature->newOwnerDomain != domain.index) {
            creature->isReplica = true;
            creature->ownerDomain = creature->newOwnerDomain;
        }
    }
}

__global__ void cudaDomainCommit_particles(SimulationData data, uint8_t round)
{
    auto const& domain = data.domain;
    auto& particles = data.entities.energies;
    auto const partition = calcSystemThreadPartition(particles.getNumEntries());
    for (int index = partition.startIndex; index <= partition.endIndex; index += partition.step) {
        auto& particle = particles.at(index);
        if (particle->ghost) {
            continue;
        }
        auto newOwner = decideOwnerByPosition(data, particle->pos.x, false);
        if (newOwner != domain.index) {
            particle->ghost = true;
            particle->ownerDomain = static_cast<uint8_t>(newOwner);
            particle->syncRound = round;
        }
    }
}

__global__ void cudaDomainCommit_resetQueues(SimulationData data, DomainSyncData syncData)
{
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        for (int domain = 0; domain < data.domain.numDomains; ++domain) {
            syncData.numGenomeRequestsToSend[domain] = 0;
            syncData.numGenomesToServe[domain] = 0;
        }
        data.domainOps.reset();
        data.sensorContinuations.reset();
        data.sensorScanRequests.reset();
        data.sensorScanResponses.reset();

        // Processed in the last time step
        data.receivedShockWaves.reset();
        data.receivedSensorScanRequests.reset();
    }
}

/************************************************************************/
/* Unpacking                                                            */
/************************************************************************/

__global__ void cudaDomainUnpack_roiBitmap(SimulationData data, DomainSyncData syncData, int sender)
{
    auto const& message = syncData.incomingMessages[sender];
    auto senderBitmap = data.domain.getRoiBitmap(sender);
    auto const partition = calcSystemThreadPartition(data.domain.layout->numBitmapWords);
    for (int index = partition.startIndex; index <= partition.endIndex; index += partition.step) {
        senderBitmap[index] = message.roiBitmap[index];
    }
}

__global__ void cudaDomainUnpack_genomes(SimulationData data, DomainSyncData syncData, int sender)
{
    auto const& message = syncData.incomingMessages[sender];
    auto const partition = calcSystemThreadPartition(message.counters->numGenomeEntries);
    for (int index = partition.startIndex; index <= partition.endIndex; index += partition.step) {
        auto const& entry = message.genomeEntries[index];
        auto genome = syncData.genomeMap.find(entry.genomeId);
        if (genome && !genome->isPlaceholder) {
            continue;
        }
        if (!genome) {
            genome = createPlaceholderGenome(data, entry.genomeId);
            syncData.genomeMap.insert(entry.genomeId, genome);
        }
        deserializeGenome(data, genome, message.genomeBytes + entry.byteOffset);
    }
}

__global__ void cudaDomainUnpack_genomeRequests(SimulationData data, DomainSyncData syncData, int sender)
{
    auto const& message = syncData.incomingMessages[sender];
    auto const partition = calcSystemThreadPartition(message.counters->numGenomeRequests);
    for (int index = partition.startIndex; index <= partition.endIndex; index += partition.step) {
        syncData.queueGenomeToServe(sender, message.genomeRequests[index]);
    }
}

__global__ void cudaDomainUnpack_creatures(SimulationData data, DomainSyncData syncData, int sender)
{
    auto const& domain = data.domain;
    auto const& message = syncData.incomingMessages[sender];
    auto const partition = calcSystemThreadPartition(message.counters->numCreatures);
    for (int index = partition.startIndex; index <= partition.endIndex; index += partition.step) {
        auto const& record = message.creatures[index];
        auto creature = syncData.creatureMap.find(record.creature.id);
        if (creature && !creature->isReplica) {
            continue;
        }
        if (!creature) {
            creature = data.entities.heap.getTypedSubArray<Creature>(1);
            syncData.creatureMap.insert(record.creature.id, creature);
        }

        auto genome = syncData.genomeMap.find(record.genomeId);
        if (!genome) {
            genome = createPlaceholderGenome(data, record.genomeId);
            syncData.genomeMap.insert(record.genomeId, genome);
            syncData.queueGenomeRequest(sender, record.genomeId);
        } else if (genome->isPlaceholder) {
            syncData.queueGenomeRequest(sender, record.genomeId);
        }

        *creature = record.creature;
        creature->genome = genome;
        creature->isReplica = !record.transfer;
        creature->ownerDomain = record.transfer ? static_cast<uint8_t>(domain.index) : record.creature.ownerDomain;
        creature->newOwnerDomain = creature->ownerDomain;
        creature->constructingCreature = nullptr;
        creature->packedForDomains = 0;
    }
}

__global__ void cudaDomainUnpack_objects(SimulationData data, DomainSyncData syncData, int sender, uint8_t round)
{
    auto const& domain = data.domain;
    auto const& message = syncData.incomingMessages[sender];
    auto const partition = calcSystemThreadPartition(message.counters->numObjects);
    for (int index = partition.startIndex; index <= partition.endIndex; index += partition.step) {
        auto& record = message.objects[index];
        auto transfer = record.transferPayloadIndex >= 0;

        auto object = syncData.objectMap.find(record.object.id);
        record.applied = !object || object->isGhost();
        if (!record.applied) {
            continue;
        }

        NeuralNet* neuralNetwork = nullptr;
        SignalEntry* signalEntries = nullptr;
        int numSignalEntries = -1;
        if (!object) {
            object = data.entities.heap.getTypedSubArray<Object>(1);
            *data.entities.objects.getNewElement() = object;
            syncData.objectMap.insert(record.object.id, object);
        } else if (object->type == ObjectType_Cell) {
            neuralNetwork = object->typeData.cell.neuralNetwork;
            if (object->typeData.cell.cellType == CellType_Memory) {
                signalEntries = object->typeData.cell.cellTypeData.memory.signalEntries;
                numSignalEntries = object->typeData.cell.cellTypeData.memory.numSignalEntries;
            }
        }

        *object = record.object;
        object->locked = 0;
        object->numConnections = 0;
        object->setGhost(!transfer);
        object->setRemovedGhost(false);
        object->ownerDomain = transfer ? static_cast<uint8_t>(domain.index) : record.object.ownerDomain;
        object->syncRound = round;

        if (object->type == ObjectType_Cell) {
            auto& cell = object->typeData.cell;
            cell.creature = syncData.creatureMap.find(record.creatureId);

            if (!neuralNetwork) {
                neuralNetwork = data.entities.heap.getTypedSubArray<NeuralNet>(1);
                zeroBytes(neuralNetwork, 1);
            }
            cell.neuralNetwork = neuralNetwork;
            if (transfer) {
                *neuralNetwork = message.transferPayloads[record.transferPayloadIndex].neuralNet;
            }

            if (cell.cellType == CellType_Memory) {
                auto& memory = cell.cellTypeData.memory;
                if (!signalEntries || numSignalEntries != memory.numSignalEntries) {
                    signalEntries = memory.numSignalEntries > 0 ? data.entities.heap.getTypedSubArray<SignalEntry>(memory.numSignalEntries) : nullptr;
                    if (signalEntries) {
                        zeroBytes(signalEntries, memory.numSignalEntries);
                    }
                }
                memory.signalEntries = signalEntries;
                if (transfer && signalEntries) {
                    copyElements(signalEntries, message.transferPayloads[record.transferPayloadIndex].memoryEntries, memory.numSignalEntries);
                }
            }
            if (cell.constructorAvailable) {
                cell.constructor.offspring = nullptr;
            }
        }
    }
}

__global__ void cudaDomainUnpack_particles(SimulationData data, DomainSyncData syncData, int sender, uint8_t round)
{
    auto const& domain = data.domain;
    auto const& message = syncData.incomingMessages[sender];
    auto const partition = calcSystemThreadPartition(message.counters->numParticles);
    for (int index = partition.startIndex; index <= partition.endIndex; index += partition.step) {
        auto const& record = message.particles[index];
        auto particle = syncData.particleMap.find(record.particle.id);
        if (particle && !particle->ghost) {
            continue;
        }
        if (!particle) {
            particle = data.entities.heap.getTypedSubArray<Energy>(1);
            *data.entities.energies.getNewElement() = particle;
            syncData.particleMap.insert(record.particle.id, particle);
        }
        *particle = record.particle;
        particle->locked = 0;
        particle->selected = 0;
        particle->lastAbsorbedObject = nullptr;
        particle->ghost = !record.transfer;
        particle->ownerDomain = record.transfer ? static_cast<uint8_t>(domain.index) : record.particle.ownerDomain;
        particle->syncRound = round;
    }
}

__global__ void cudaDomainUnpack_resolveObjects(SimulationData data, DomainSyncData syncData, int sender)
{
    auto const& message = syncData.incomingMessages[sender];
    auto const partition = calcSystemThreadPartition(message.counters->numObjects);
    for (int index = partition.startIndex; index <= partition.endIndex; index += partition.step) {
        auto const& record = message.objects[index];
        if (!record.applied) {
            continue;
        }
        auto object = syncData.objectMap.find(record.object.id);
        if (!object || object->isRemovedGhost()) {
            continue;
        }

        if (object->type == ObjectType_Cell && object->typeData.cell.creature == nullptr) {
            object->setGhost(true);
            object->setRemovedGhost(true);
            continue;
        }

        auto numConnections = 0;
        auto pendingAngle = 0.0f;
        for (int i = 0; i < record.object.numConnections; ++i) {
            auto const& connection = record.object.connections[i];
            auto connectedObject = syncData.objectMap.find(record.connectionIds[i]);
            if (!connectedObject || connectedObject->isRemovedGhost()) {
                pendingAngle += connection.angleFromPrevious;
                continue;
            }
            object->connections[numConnections].object = connectedObject;
            object->connections[numConnections].distance = connection.distance;
            object->connections[numConnections].angleFromPrevious = connection.angleFromPrevious + pendingAngle;
            pendingAngle = 0;
            ++numConnections;
        }
        if (numConnections > 0 && pendingAngle > 0) {
            object->connections[0].angleFromPrevious += pendingAngle;
        }
        object->numConnections = numConnections;
    }
}

__global__ void cudaDomainUnpack_applyOps(SimulationData data, SimulationStatistics statistics, DomainSyncData syncData, int sender)
{
    auto const& message = syncData.incomingMessages[sender];
    auto const partition = calcSystemThreadPartition(message.counters->numOps);
    for (int index = partition.startIndex; index <= partition.endIndex; index += partition.step) {
        DomainOpProcessor::apply(data, statistics, syncData, message.ops[index]);
    }
}

__global__ void cudaDomainUnpack_sensorScanRequests(SimulationData data, DomainSyncData syncData, int sender)
{
    auto const& message = syncData.incomingMessages[sender];
    auto const partition = calcSystemThreadPartition(message.counters->numSensorScanRequests);
    for (int index = partition.startIndex; index <= partition.endIndex; index += partition.step) {
        if (auto request = data.receivedSensorScanRequests.tryGetNewElement()) {
            *request = message.sensorScanRequests[index];
        }
    }
}

__global__ void cudaDomainUnpack_sensorScanResponses(SimulationData data, DomainSyncData syncData, int sender)
{
    auto const& message = syncData.incomingMessages[sender];
    auto const partition = calcSystemThreadPartition(message.counters->numSensorScanResponses * SensorScan::NumRays);
    for (int index = partition.startIndex; index <= partition.endIndex; index += partition.step) {
        SensorProcessor::addScanResponse(data, message.sensorScanResponses[index / SensorScan::NumRays], index % SensorScan::NumRays);
    }
}

/************************************************************************/
/* Finishing a sync round                                               */
/************************************************************************/

// Stale ghost particles are deleted right away since nothing points to them
__global__ void cudaDomainSync_markStaleGhosts(SimulationData data, uint8_t round)
{
    {
        auto& objects = data.entities.objects;
        auto const partition = calcSystemThreadPartition(objects.getNumEntries());
        for (int index = partition.startIndex; index <= partition.endIndex; index += partition.step) {
            auto object = objects.at(index);
            if (object->isGhost() && object->syncRound != round) {
                object->setRemovedGhost(true);
            }
        }
    }
    {
        auto& particles = data.entities.energies;
        auto const partition = calcSystemThreadPartition(particles.getNumEntries());
        for (int index = partition.startIndex; index <= partition.endIndex; index += partition.step) {
            auto& particle = particles.at(index);
            if (particle->ghost && particle->syncRound != round) {
                particle = nullptr;
            }
        }
    }
}

__global__ void cudaDomainSync_removeConnectionsToRemovedGhosts(SimulationData data)
{
    auto& objects = data.entities.objects;
    auto const partition = calcSystemThreadPartition(objects.getNumEntries());
    for (int index = partition.startIndex; index <= partition.endIndex; index += partition.step) {
        auto object = objects.at(index);
        if (object->isRemovedGhost()) {
            continue;
        }
        for (int i = object->numConnections - 1; i >= 0; --i) {
            auto connectedObject = object->connections[i].object;
            if (connectedObject->isRemovedGhost()) {
                ObjectConnectionProcessor::deleteConnectionOneWay(object, connectedObject);
            }
        }
    }
}

__global__ void cudaDomainSync_deleteRemovedGhosts(SimulationData data)
{
    auto& objects = data.entities.objects;
    auto const partition = calcSystemThreadPartition(objects.getNumEntries());
    for (int index = partition.startIndex; index <= partition.endIndex; index += partition.step) {
        auto& object = objects.at(index);
        if (object->isRemovedGhost()) {
            object = nullptr;
        }
    }
}

// The scans of the time step before the last one have received all responses by now
__global__ void cudaDomainSync_publishSensorScans(SimulationData data, DomainSyncData syncData)
{
    auto& scans = data.pendingSensorScans[*data.timestep % 2];
    auto const partition = calcSystemThreadPartition(scans.getNumEntries());
    for (int index = partition.startIndex; index <= partition.endIndex; index += partition.step) {
        auto const& scan = scans.at(index);
        if (!scan.active) {
            continue;
        }
        auto object = syncData.objectMap.find(scan.sensorId);
        if (object && !object->isGhost() && object->type == ObjectType_Cell && object->typeData.cell.cellType == CellType_Sensor) {
            SensorProcessor::publishPendingScan(data, object, scan);
        }
    }
}

__global__ void cudaDomainSync_resetPendingSensorScans(SimulationData data)
{
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        data.pendingSensorScans[*data.timestep % 2].reset();
    }
}

/************************************************************************/
/* Initial distribution                                                 */
/************************************************************************/

__global__ void cudaDomainDistribute_assignOwners(SimulationData data, uint8_t round)
{
    auto const& domain = data.domain;
    auto& objects = data.entities.objects;
    auto const partition = calcSystemThreadPartition(objects.getNumEntries());
    for (int index = partition.startIndex; index <= partition.endIndex; index += partition.step) {
        auto object = objects.at(index);
        auto owner = decideObjectOwner(data, object, true);
        if (owner == domain.index) {
            continue;
        }
        object->setGhost(true);
        object->ownerDomain = static_cast<uint8_t>(owner);
        object->syncRound = round;
        if (object->type == ObjectType_Cell && object->typeData.cell.constructorAvailable) {
            object->typeData.cell.constructor.offspring = nullptr;
        }
    }
}

__global__ void cudaDomainDistribute_removeObjectsOutsideRoi(SimulationData data)
{
    auto const& domain = data.domain;
    auto& objects = data.entities.objects;
    auto const partition = calcSystemThreadPartition(objects.getNumEntries());
    for (int index = partition.startIndex; index <= partition.endIndex; index += partition.step) {
        auto object = objects.at(index);
        if (object->isGhost() && !domain.isInRoi(domain.index, object->pos)) {
            object->setRemovedGhost(true);
        }
    }
}

__global__ void cudaDomainDistribute_assignParticleOwners(SimulationData data, uint8_t round)
{
    auto const& domain = data.domain;
    auto& particles = data.entities.energies;
    auto const partition = calcSystemThreadPartition(particles.getNumEntries());
    for (int index = partition.startIndex; index <= partition.endIndex; index += partition.step) {
        auto& particle = particles.at(index);
        auto owner = decideOwnerByPosition(data, particle->pos.x, true);
        if (owner == domain.index) {
            continue;
        }
        if (domain.isInRoi(domain.index, particle->pos)) {
            particle->ghost = true;
            particle->ownerDomain = static_cast<uint8_t>(owner);
            particle->syncRound = round;
        } else {
            particle = nullptr;
        }
    }
}
