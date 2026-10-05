#include "DomainSyncService.cuh"

#include <algorithm>
#include <array>
#include <bit>
#include <cstdlib>
#include <set>

#include <Data/SimulationParametersTypes.h>

#include <EngineKernels/DomainSyncKernels.cuh>
#include <EngineKernels/KernelLauncher.cuh>
#include <EngineKernels/SimulationData.cuh>

#include "GarbageCollectorKernelsService.cuh"

namespace
{
    auto constexpr OwnershipHysteresis = 2.0f;

    SyncMessageCapacities const InitialCapacities{
        .objects = 4096,
        .transferPayloads = 256,
        .creatures = 1024,
        .genomeEntries = 64,
        .genomeBytes = 1024 * 1024,
        .genomeRequests = 1024,
        .particles = 4096,
        .ops = 16384,
        .sensorScanRequests = 256,
        .sensorScanResponses = 256,
        .bitmapWords = 0};

    auto constexpr InitialOutboxCapacity = 65536;
    auto constexpr InitialSensorScanCapacity = 4096;

    void activateDevice(Domain const& domain)
    {
        CHECK_FOR_DEVICE_ERRORS(cudaSetDevice(domain.device));
    }

    // Messages between devices that can access each other's memory are copied without a detour over the host
    void enablePeerAccess(std::vector<Domain> const& domains)
    {
        std::set<int> devices;
        for (auto const& domain : domains) {
            devices.insert(domain.device);
        }
        for (auto device : devices) {
            for (auto peerDevice : devices) {
                if (device == peerDevice) {
                    continue;
                }
                int canAccessPeer = 0;
                CHECK_FOR_DEVICE_ERRORS(cudaDeviceCanAccessPeer(&canAccessPeer, device, peerDevice));
                if (!canAccessPeer) {
                    continue;
                }
                DeviceScope deviceScope(device);
                auto result = cudaDeviceEnablePeerAccess(peerDevice, 0);
                if (result == cudaErrorPeerAccessAlreadyEnabled) {
                    cudaGetLastError();
                } else {
                    CHECK_FOR_DEVICE_ERRORS(result);
                }
            }
        }
    }

    uint64_t calcMapCapacity(uint64_t numEntries)
    {
        return std::bit_ceil(std::max<uint64_t>(1024, numEntries * 2));
    }

    uint64_t grow(uint64_t required, uint64_t current)
    {
        return required <= current ? current : std::max(required + required / 2, current * 2);
    }

    // The entries of a time step are collected before they can be sent, so an empty outbox is kept at twice the last usage
    template <typename T>
    void growOutboxIfNecessary(Array<T> const& outbox, uint64_t lastUsage)
    {
        auto capacity = outbox.getCapacity_host();
        if (lastUsage * 2 > capacity) {
            outbox.resize(std::max(capacity * 2, lastUsage * 4));
        }
    }

    // Only for empty arrays, since resizing discards the content
    template <typename T>
    void ensureCapacity(Array<T> const& array, uint64_t numEntries)
    {
        auto capacity = array.getCapacity_host();
        if (numEntries > capacity) {
            array.resize(grow(numEntries, capacity));
        }
    }
}

void DomainSyncService::init(std::vector<Domain>& domains, int2 const& worldSize, float haloWidth)
{
    enablePeerAccess(domains);

    auto numDomains = static_cast<int>(domains.size());

    DomainLayout layout;
    layout.numDomains = numDomains;
    layout.worldSize = worldSize;
    for (int index = 0; index <= numDomains; ++index) {
        layout.stripStarts[index] = static_cast<float>(worldSize.x) * static_cast<float>(index) / static_cast<float>(numDomains);
    }
    layout.haloWidth = haloWidth;
    layout.ownershipHysteresis = OwnershipHysteresis;
    layout.numTiles = {
        (worldSize.x + DomainLayout::TileSize - 1) / DomainLayout::TileSize, (worldSize.y + DomainLayout::TileSize - 1) / DomainLayout::TileSize};
    layout.numBitmapWords = (layout.numTiles.x * layout.numTiles.y + 31) / 32;
    for (int index = 0; index < numDomains; ++index) {
        layout.externalEnergyShares[index] = 1.0f / static_cast<float>(numDomains);
    }
    _layout = layout;

    auto capacities = InitialCapacities;
    capacities.bitmapWords = layout.numBitmapWords;

    for (auto& domain : domains) {
        activateDevice(domain);
        domain.sync = std::make_shared<DomainSyncState>();
        auto& sync = *domain.sync;

        CudaMemoryManager::getInstance().acquireMemory<DomainLayout>(1, sync.layout);
        copyToDevice(sync.layout, &layout);
        CudaMemoryManager::getInstance().acquireMemory<uint32_t>(static_cast<uint64_t>(numDomains) * layout.numBitmapWords, sync.roiBitmaps);
        CHECK_FOR_DEVICE_ERRORS(cudaMemset(sync.roiBitmaps, 0, sizeof(uint32_t) * numDomains * layout.numBitmapWords));

        auto& data = *domain.data;
        data.domain.index = domain.index;
        data.domain.numDomains = numDomains;
        data.domain.layout = sync.layout;
        data.domain.roiBitmaps = sync.roiBitmaps;
        data.domainOps.resize(InitialOutboxCapacity);
        data.receivedShockWaves.resize(InitialCapacities.ops);
        data.sensorContinuations.resize(InitialSensorScanCapacity);
        data.sensorScanRequests.resize(InitialSensorScanCapacity);
        data.receivedSensorScanRequests.resize(InitialSensorScanCapacity);
        data.sensorScanResponses.resize(InitialSensorScanCapacity);
        for (auto const& scans : data.pendingSensorScans) {
            scans.resize(InitialSensorScanCapacity);
        }

        sync.data.objectMap.init();
        sync.data.particleMap.init();
        sync.data.creatureMap.init();
        sync.data.genomeMap.init();
        CudaMemoryManager::getInstance().acquireMemory<SyncMessage>(DomainLayout::MaxDomains, sync.data.outgoingMessages);
        CudaMemoryManager::getInstance().acquireMemory<SyncMessage>(DomainLayout::MaxDomains, sync.data.incomingMessages);
        CudaMemoryManager::getInstance().acquireMemory<uint32_t>(layout.numBitmapWords, sync.data.baseBitmap);
        CudaMemoryManager::getInstance().acquireMemory<uint64_t>(numDomains * DomainSyncData::MaxGenomeRequests, sync.data.genomeRequestsToSend);
        CudaMemoryManager::getInstance().acquireMemory<int>(numDomains, sync.data.numGenomeRequestsToSend);
        CudaMemoryManager::getInstance().acquireMemory<uint64_t>(numDomains * DomainSyncData::MaxGenomeRequests, sync.data.genomesToServe);
        CudaMemoryManager::getInstance().acquireMemory<int>(numDomains, sync.data.numGenomesToServe);
        CHECK_FOR_DEVICE_ERRORS(cudaMemset(sync.data.numGenomeRequestsToSend, 0, sizeof(int) * numDomains));
        CHECK_FOR_DEVICE_ERRORS(cudaMemset(sync.data.numGenomesToServe, 0, sizeof(int) * numDomains));

        sync.outgoing.resize(numDomains);
        sync.incoming.resize(numDomains);
        sync.isIncomingAliased.resize(numDomains, false);
        sync.outgoingCounters.resize(numDomains);
        for (int receiver = 0; receiver < numDomains; ++receiver) {
            if (receiver != domain.index) {
                allocateMessage(sync.outgoing.at(receiver), capacities);
            }
        }
    }
    connectIncomingMessages(domains);
}

void DomainSyncService::shutdown(std::vector<Domain>& domains)
{
    for (auto& domain : domains) {
        if (!domain.sync) {
            continue;
        }
        activateDevice(domain);
        auto& sync = *domain.sync;
        for (auto const& sender : domains) {
            if (sender.index != domain.index && !sync.isIncomingAliased.at(sender.index)) {
                freeMessage(sync.incoming.at(sender.index));
            }
        }
        for (auto& message : sync.outgoing) {
            freeMessage(message);
        }
        sync.data.objectMap.free();
        sync.data.particleMap.free();
        sync.data.creatureMap.free();
        sync.data.genomeMap.free();
        CudaMemoryManager::getInstance().freeMemory(sync.data.outgoingMessages);
        CudaMemoryManager::getInstance().freeMemory(sync.data.incomingMessages);
        CudaMemoryManager::getInstance().freeMemory(sync.data.baseBitmap);
        CudaMemoryManager::getInstance().freeMemory(sync.data.genomeRequestsToSend);
        CudaMemoryManager::getInstance().freeMemory(sync.data.numGenomeRequestsToSend);
        CudaMemoryManager::getInstance().freeMemory(sync.data.genomesToServe);
        CudaMemoryManager::getInstance().freeMemory(sync.data.numGenomesToServe);
        CudaMemoryManager::getInstance().freeMemory(sync.layout);
        CudaMemoryManager::getInstance().freeMemory(sync.roiBitmaps);
        domain.sync.reset();
    }
}

void DomainSyncService::distribute(std::vector<Domain>& domains, KernelLaunchSettings const& launchSettings)
{
    ++_round;
    for (auto& domain : domains) {
        activateDevice(domain);
        auto const& data = *domain.data;
        calcOwnership(domain, launchSettings, true);
        launchKernelOnDefaultStream(KERNEL(cudaDomainDistribute_assignOwners), LaunchConfig{launchSettings.numBlocks, 8}, data, _round);
        calcRoi(domain, launchSettings);
        launchKernelOnDefaultStream(KERNEL(cudaDomainDistribute_removeObjectsOutsideRoi), LaunchConfig{launchSettings.numBlocks, 8}, data);
        launchKernelOnDefaultStream(KERNEL(cudaDomainDistribute_assignParticleOwners), LaunchConfig{launchSettings.numBlocks, 8}, data, _round);
        launchKernelOnDefaultStream(KERNEL(cudaDomainSync_removeConnectionsToRemovedGhosts), LaunchConfig{launchSettings.numBlocks, 8}, data);
        launchKernelOnDefaultStream(KERNEL(cudaDomainSync_deleteRemovedGhosts), LaunchConfig{launchSettings.numBlocks, 8}, data);
        GarbageCollectorKernelsService::get().cleanupAfterDataManipulation(launchSettings, data);
        CHECK_FOR_DEVICE_ERRORS(cudaDeviceSynchronize());
    }
    exchangeRoiBitmaps(domains);
}

void DomainSyncService::sync(std::vector<Domain>& domains, KernelLaunchSettings const& launchSettings, EnsureCapacityFunc const& ensureCapacity)
{
    ++_round;

    for (auto& domain : domains) {
        activateDevice(domain);
        rebuildMaps(domain, launchSettings, ArraySizesForGpuEntities{0, 0, 0});
        calcOwnership(domain, launchSettings, false);
        calcRoi(domain, launchSettings);
    }

    do {
        for (auto& domain : domains) {
            activateDevice(domain);
            pack(domain, launchSettings);
        }
    } while (readOutgoingCountersAndGrowOnOverflow(domains));

    for (auto& domain : domains) {
        activateDevice(domain);
        auto const& data = *domain.data;
        auto numOps = data.domainOps.getNumEntries_host();
        auto numSensorContinuations = data.sensorContinuations.getNumEntries_host();
        auto numSensorScanRequests = data.sensorScanRequests.getNumEntries_host();
        commit(domain, launchSettings);
        growOutboxIfNecessary(data.domainOps, numOps);
        growOutboxIfNecessary(data.sensorContinuations, numSensorContinuations);
        growOutboxIfNecessary(data.sensorScanRequests, numSensorScanRequests);
    }

    transport(domains);

    for (auto& domain : domains) {
        activateDevice(domain);
        auto incomingSizes = calcIncomingSizes(domain, domains);
        ensureCapacity(domain, incomingSizes);
        rebuildMaps(domain, launchSettings, incomingSizes);
        unpack(domain, domains, launchSettings);
    }
    for (auto& domain : domains) {
        activateDevice(domain);
        finishRound(domain, domains, launchSettings);
    }
}

// The domains draw from the external energy pool like a single simulation: the radiation sources first and the constructors from the
// rest, each in proportion to their demand. The constructors only draw in time steps with cell functions.
void DomainSyncService::updateExternalEnergyShares(
    std::vector<Domain>& domains,
    double externalEnergy,
    bool constructorDemandsMeasured,
    bool constructorsDrawNext)
{
    if (externalEnergy <= 0 || externalEnergy >= Infinity<float>::value) {
        return;
    }
    std::vector<double> sourceDemands;
    auto totalSourceDemand = 0.0;
    auto totalConstructorDemand = 0.0;
    for (auto& domain : domains) {
        activateDevice(domain);
        std::array<double, ExternalEnergyDemand_Count> demands;
        copyToHost(demands.data(), domain.data->externalEnergyDemands, ExternalEnergyDemand_Count);
        if (constructorDemandsMeasured) {
            domain.sync->constructorEnergyDemand = demands.at(ExternalEnergyDemand_Constructors);
        }
        sourceDemands.emplace_back(demands.at(ExternalEnergyDemand_Sources));
        totalSourceDemand += demands.at(ExternalEnergyDemand_Sources);
        totalConstructorDemand += constructorsDrawNext ? domain.sync->constructorEnergyDemand : 0.0;
    }

    auto sourceSupply = std::min(externalEnergy, totalSourceDemand);
    auto constructorSupply = std::min(externalEnergy - sourceSupply, totalConstructorDemand);
    auto unclaimedSupply = (externalEnergy - sourceSupply - constructorSupply) / static_cast<double>(domains.size());
    for (auto const& domain : domains) {
        auto supply = unclaimedSupply;
        if (totalSourceDemand > 0) {
            supply += sourceSupply * sourceDemands.at(domain.index) / totalSourceDemand;
        }
        if (totalConstructorDemand > 0) {
            supply += constructorSupply * domain.sync->constructorEnergyDemand / totalConstructorDemand;
        }
        _layout.externalEnergyShares[domain.index] = static_cast<float>(supply / externalEnergy);
    }
    for (auto const& domain : domains) {
        activateDevice(domain);
        copyToDevice(domain.sync->layout, &_layout);
    }
}

void DomainSyncService::rebuildMaps(Domain& domain, KernelLaunchSettings const& launchSettings, ArraySizesForGpuEntities const& incomingSizes)
{
    auto& sync = *domain.sync;
    auto const& data = *domain.data;
    auto objectMapCapacity = calcMapCapacity(data.entities.objects.getNumEntries_host() + incomingSizes.objectArray);
    if (sync.data.objectMap.getCapacity_host() < objectMapCapacity) {
        sync.data.objectMap.resize(objectMapCapacity);
        sync.data.creatureMap.resize(objectMapCapacity);
        sync.data.genomeMap.resize(objectMapCapacity);
    }
    auto particleMapCapacity = calcMapCapacity(data.entities.energies.getNumEntries_host() + incomingSizes.energyArray);
    if (sync.data.particleMap.getCapacity_host() < particleMapCapacity) {
        sync.data.particleMap.resize(particleMapCapacity);
    }
    launchKernelOnDefaultStream(KERNEL(cudaDomainMaps_clear), LaunchConfig{launchSettings.numBlocks, 64}, sync.data);
    launchKernelOnDefaultStream(KERNEL(cudaDomainMaps_insert), LaunchConfig{launchSettings.numBlocks, 8}, data, sync.data);
}

void DomainSyncService::calcOwnership(Domain& domain, KernelLaunchSettings const& launchSettings, bool initialAssignment)
{
    auto const& data = *domain.data;
    launchKernelOnDefaultStream(KERNEL(cudaDomainOwnership_resetCreatures), LaunchConfig{launchSettings.numBlocks, 8}, data);
    launchKernelOnDefaultStream(KERNEL(cudaDomainOwnership_calcReferenceKeys), LaunchConfig{launchSettings.numBlocks, 8}, data);
    launchKernelOnDefaultStream(KERNEL(cudaDomainOwnership_calcReferencePositions), LaunchConfig{launchSettings.numBlocks, 8}, data);
    launchKernelOnDefaultStream(KERNEL(cudaDomainOwnership_findConstructingCreatures), LaunchConfig{launchSettings.numBlocks, 8}, data);
    launchKernelOnDefaultStream(KERNEL(cudaDomainOwnership_decideCreatureOwners), LaunchConfig{launchSettings.numBlocks, 8}, data, initialAssignment);
}

void DomainSyncService::calcRoi(Domain& domain, KernelLaunchSettings const& launchSettings)
{
    auto const& data = *domain.data;
    auto const& syncData = domain.sync->data;
    launchKernelOnDefaultStream(KERNEL(cudaDomainRoi_clearBaseBitmap), LaunchConfig{launchSettings.numBlocks, 8}, data, syncData);
    launchKernelOnDefaultStream(KERNEL(cudaDomainRoi_markOwnedObjects), LaunchConfig{launchSettings.numBlocks, 8}, data, syncData);
    launchKernelOnDefaultStream(KERNEL(cudaDomainRoi_dilate), LaunchConfig{launchSettings.numBlocks, 8}, data, syncData);
}

void DomainSyncService::pack(Domain& domain, KernelLaunchSettings const& launchSettings)
{
    auto const& data = *domain.data;
    auto const& syncData = domain.sync->data;
    launchKernelOnDefaultStream(KERNEL(cudaDomainPack_reset), LaunchConfig{launchSettings.numBlocks, 8}, data, syncData);
    launchKernelOnDefaultStream(KERNEL(cudaDomainPack_roiBitmaps), LaunchConfig{launchSettings.numBlocks, 8}, data, syncData);
    launchKernelOnDefaultStream(KERNEL(cudaDomainPack_objects), LaunchConfig{launchSettings.numBlocks, 8}, data, syncData);
    launchKernelOnDefaultStream(KERNEL(cudaDomainPack_particles), LaunchConfig{launchSettings.numBlocks, 8}, data, syncData);
    launchKernelOnDefaultStream(KERNEL(cudaDomainPack_genomeRequests), LaunchConfig{launchSettings.numBlocks, 8}, data, syncData);
    launchKernelOnDefaultStream(KERNEL(cudaDomainPack_requestedGenomes), LaunchConfig{launchSettings.numBlocks, 8}, data, syncData);
    launchKernelOnDefaultStream(KERNEL(cudaDomainPack_transferredGenomes), LaunchConfig{launchSettings.numBlocks, 8}, data, syncData);
    launchKernelOnDefaultStream(KERNEL(cudaDomainPack_ops), LaunchConfig{launchSettings.numBlocks, 8}, data, syncData);
    launchKernelOnDefaultStream(KERNEL(cudaDomainPack_sensorScans), LaunchConfig{launchSettings.numBlocks, 8}, data, syncData);
}

void DomainSyncService::commit(Domain& domain, KernelLaunchSettings const& launchSettings)
{
    auto const& data = *domain.data;
    auto const& syncData = domain.sync->data;
    launchKernelOnDefaultStream(KERNEL(cudaDomainCommit_objects), LaunchConfig{launchSettings.numBlocks, 8}, data, _round);
    launchKernelOnDefaultStream(KERNEL(cudaDomainCommit_particles), LaunchConfig{launchSettings.numBlocks, 8}, data, _round);
    launchKernelOnDefaultStream(KERNEL(cudaDomainCommit_resetQueues), LaunchConfig{1, 1}, data, syncData);
}

void DomainSyncService::unpack(Domain& receiver, std::vector<Domain>& domains, KernelLaunchSettings const& launchSettings)
{
    auto const& data = *receiver.data;
    auto const& syncData = receiver.sync->data;

    // Each received scan request is answered in the next time step
    uint64_t numIncomingOps = 0;
    uint64_t numIncomingSensorScanRequests = 0;
    for (auto const& sender : domains) {
        if (sender.index != receiver.index) {
            auto const& counters = sender.sync->outgoingCounters.at(receiver.index);
            numIncomingOps += counters.numOps;
            numIncomingSensorScanRequests += counters.numSensorScanRequests;
        }
    }
    ensureCapacity(data.receivedShockWaves, numIncomingOps);
    ensureCapacity(data.receivedSensorScanRequests, numIncomingSensorScanRequests);
    ensureCapacity(data.sensorScanResponses, numIncomingSensorScanRequests);

    for (auto const& sender : domains) {
        if (sender.index == receiver.index) {
            continue;
        }
        launchKernelOnDefaultStream(KERNEL(cudaDomainUnpack_roiBitmap), LaunchConfig{launchSettings.numBlocks, 8}, data, syncData, sender.index);
        launchKernelOnDefaultStream(KERNEL(cudaDomainUnpack_genomes), LaunchConfig{launchSettings.numBlocks, 8}, data, syncData, sender.index);
        launchKernelOnDefaultStream(KERNEL(cudaDomainUnpack_genomeRequests), LaunchConfig{launchSettings.numBlocks, 8}, data, syncData, sender.index);
        launchKernelOnDefaultStream(KERNEL(cudaDomainUnpack_creatures), LaunchConfig{launchSettings.numBlocks, 8}, data, syncData, sender.index);
        launchKernelOnDefaultStream(KERNEL(cudaDomainUnpack_objects), LaunchConfig{launchSettings.numBlocks, 8}, data, syncData, sender.index, _round);
        launchKernelOnDefaultStream(KERNEL(cudaDomainUnpack_particles), LaunchConfig{launchSettings.numBlocks, 8}, data, syncData, sender.index, _round);
        launchKernelOnDefaultStream(KERNEL(cudaDomainUnpack_sensorScanRequests), LaunchConfig{launchSettings.numBlocks, 8}, data, syncData, sender.index);
        launchKernelOnDefaultStream(KERNEL(cudaDomainUnpack_sensorScanResponses), LaunchConfig{launchSettings.numBlocks, 64}, data, syncData, sender.index);
    }
}

void DomainSyncService::finishRound(Domain& domain, std::vector<Domain>& domains, KernelLaunchSettings const& launchSettings)
{
    auto const& data = *domain.data;
    auto const& syncData = domain.sync->data;
    launchKernelOnDefaultStream(KERNEL(cudaDomainSync_markStaleGhosts), LaunchConfig{launchSettings.numBlocks, 8}, data, _round);
    for (auto const& sender : domains) {
        if (sender.index != domain.index) {
            launchKernelOnDefaultStream(KERNEL(cudaDomainUnpack_resolveObjects), LaunchConfig{launchSettings.numBlocks, 8}, data, syncData, sender.index);
        }
    }
    for (auto const& sender : domains) {
        if (sender.index != domain.index) {
            launchKernelOnDefaultStream(
                KERNEL(cudaDomainUnpack_applyOps), LaunchConfig{launchSettings.numBlocks, 8}, data, *domain.statistics, syncData, sender.index);
        }
    }

    // The scans that are published now are replaced by the scans of the next time step
    auto const& pendingSensorScans = data.pendingSensorScans[copyToHost(data.timestep) % 2];
    auto numPendingSensorScans = pendingSensorScans.getNumEntries_host();
    launchKernelOnDefaultStream(KERNEL(cudaDomainSync_publishSensorScans), LaunchConfig{launchSettings.numBlocks, 8}, data, syncData);
    launchKernelOnDefaultStream(KERNEL(cudaDomainSync_resetPendingSensorScans), LaunchConfig{1, 1}, data);

    launchKernelOnDefaultStream(KERNEL(cudaDomainSync_removeConnectionsToRemovedGhosts), LaunchConfig{launchSettings.numBlocks, 8}, data);
    launchKernelOnDefaultStream(KERNEL(cudaDomainSync_deleteRemovedGhosts), LaunchConfig{launchSettings.numBlocks, 8}, data);
    GarbageCollectorKernelsService::get().compactPointerArrays(launchSettings, data);
    CHECK_FOR_DEVICE_ERRORS(cudaDeviceSynchronize());
    growOutboxIfNecessary(pendingSensorScans, numPendingSensorScans);
}

bool DomainSyncService::readOutgoingCountersAndGrowOnOverflow(std::vector<Domain>& domains)
{
    auto overflow = false;
    for (auto& domain : domains) {
        activateDevice(domain);
        CHECK_FOR_DEVICE_ERRORS(cudaDeviceSynchronize());
        auto& sync = *domain.sync;
        auto domainOverflow = false;
        for (auto const& receiver : domains) {
            if (receiver.index == domain.index) {
                continue;
            }
            auto& message = sync.outgoing.at(receiver.index);
            auto& counters = sync.outgoingCounters.at(receiver.index);
            copyToHost(&counters, message.counters);
            if (!counters.overflow) {
                continue;
            }
            auto capacities = message.capacities;
            capacities.objects = grow(counters.numObjects, capacities.objects);
            capacities.transferPayloads = grow(counters.numTransferPayloads, capacities.transferPayloads);
            capacities.creatures = grow(counters.numCreatures, capacities.creatures);
            capacities.genomeEntries = grow(counters.numGenomeEntries, capacities.genomeEntries);
            capacities.genomeBytes = grow(counters.numGenomeBytes, capacities.genomeBytes);
            capacities.genomeRequests = grow(counters.numGenomeRequests, capacities.genomeRequests);
            capacities.particles = grow(counters.numParticles, capacities.particles);
            capacities.ops = grow(counters.numOps, capacities.ops);
            capacities.sensorScanRequests = grow(counters.numSensorScanRequests, capacities.sensorScanRequests);
            capacities.sensorScanResponses = grow(counters.numSensorScanResponses, capacities.sensorScanResponses);
            freeMessage(message);
            allocateMessage(message, capacities);
            domainOverflow = true;
        }
        overflow |= domainOverflow;
    }
    if (overflow) {
        connectIncomingMessages(domains);
    }
    return overflow;
}

void DomainSyncService::transport(std::vector<Domain>& domains)
{
    for (auto& receiver : domains) {
        auto& receiverSync = *receiver.sync;
        for (auto const& sender : domains) {
            if (sender.index == receiver.index || receiverSync.isIncomingAliased.at(sender.index)) {
                continue;
            }
            auto const& source = sender.sync->outgoing.at(receiver.index);
            auto const& target = receiverSync.incoming.at(sender.index);
            for (auto const& range : source.getUsedRanges(sender.sync->outgoingCounters.at(receiver.index))) {
                if (range.size > 0) {
                    CHECK_FOR_DEVICE_ERRORS(
                        cudaMemcpyPeer(target.buffer + range.offset, receiver.device, source.buffer + range.offset, sender.device, range.size));
                }
            }
        }
    }
}

void DomainSyncService::exchangeRoiBitmaps(std::vector<Domain>& domains)
{
    for (auto& receiver : domains) {
        for (auto const& sender : domains) {
            if (sender.index == receiver.index) {
                continue;
            }
            auto const& senderSync = *sender.sync;
            auto numWords = senderSync.outgoing.at(receiver.index).capacities.bitmapWords;
            auto source = senderSync.roiBitmaps + numWords * sender.index;
            auto target = receiver.sync->roiBitmaps + numWords * sender.index;
            CHECK_FOR_DEVICE_ERRORS(cudaMemcpyPeer(target, receiver.device, source, sender.device, numWords * sizeof(uint32_t)));
        }
    }
}

ArraySizesForGpuEntities DomainSyncService::calcIncomingSizes(Domain const& receiver, std::vector<Domain> const& domains) const
{
    ArraySizesForGpuEntities result{0, 0, 0};
    for (auto const& sender : domains) {
        if (sender.index == receiver.index) {
            continue;
        }
        auto const& counters = sender.sync->outgoingCounters.at(receiver.index);
        result.objectArray += counters.numObjects;
        result.energyArray += counters.numParticles;
        result.heap += counters.numObjects * (sizeof(Object) + sizeof(NeuralNet) + sizeof(SignalEntry) * MAX_CELL_MEMORY_ENTRIES + 3 * GpuMemoryAlignmentBytes)
            + counters.numCreatures * (sizeof(Creature) + sizeof(Genome) + 2 * GpuMemoryAlignmentBytes) + counters.numGenomeBytes * 2
            + counters.numGenomeEntries * 64 * GpuMemoryAlignmentBytes + counters.numParticles * (sizeof(Energy) + GpuMemoryAlignmentBytes);
    }
    return result;
}

void DomainSyncService::allocateMessage(SyncMessage& message, SyncMessageCapacities const& capacities)
{
    uint8_t* buffer = nullptr;
    auto size = SyncMessage::calcBufferSize(capacities);
    CudaMemoryManager::getInstance().acquireMemory<uint8_t>(size, buffer);
    CHECK_FOR_DEVICE_ERRORS(cudaMemset(buffer, 0, sizeof(SyncMessageCounters)));
    message.assignBuffer(buffer, capacities);
}

void DomainSyncService::freeMessage(SyncMessage& message)
{
    if (message.buffer) {
        CudaMemoryManager::getInstance().freeMemory(message.buffer);
    }
    message = SyncMessage();
}

namespace
{
    // Developer switch that copies the messages between domains on the same device as if they were on different devices
    bool isMessageCopyForced()
    {
        static auto result = std::getenv("ALIEN_COPY_DOMAIN_MESSAGES") != nullptr;
        return result;
    }
}

// A message between domains on the same device is read directly from the buffer of the sender
void DomainSyncService::connectIncomingMessages(std::vector<Domain>& domains)
{
    for (auto& receiver : domains) {
        activateDevice(receiver);
        auto& receiverSync = *receiver.sync;
        for (auto const& sender : domains) {
            if (sender.index == receiver.index) {
                continue;
            }
            auto const& outgoing = sender.sync->outgoing.at(receiver.index);
            auto& incoming = receiverSync.incoming.at(sender.index);
            if (sender.device == receiver.device && !isMessageCopyForced()) {
                incoming = outgoing;
                receiverSync.isIncomingAliased.at(sender.index) = true;
            } else {
                if (receiverSync.isIncomingAliased.at(sender.index) || incoming.capacities != outgoing.capacities) {
                    if (!receiverSync.isIncomingAliased.at(sender.index)) {
                        freeMessage(incoming);
                    }
                    allocateMessage(incoming, outgoing.capacities);
                }
                receiverSync.isIncomingAliased.at(sender.index) = false;
            }
        }
    }
    for (auto& domain : domains) {
        activateDevice(domain);
        uploadMessageDescriptors(domain);
    }
}

void DomainSyncService::uploadMessageDescriptors(Domain& domain)
{
    auto& sync = *domain.sync;
    copyToDevice(sync.data.outgoingMessages, sync.outgoing.data(), static_cast<int>(sync.outgoing.size()));
    copyToDevice(sync.data.incomingMessages, sync.incoming.data(), static_cast<int>(sync.incoming.size()));
}
