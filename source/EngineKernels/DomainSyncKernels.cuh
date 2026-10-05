#pragma once

#include "DomainSync.cuh"
#include "SimulationData.cuh"
#include "SimulationStatistics.cuh"

__global__ void cudaDomainSync_gatherControl(SimulationData data, DomainSyncControl* control);

// Ownership decisions at the end of a time step
__global__ void cudaDomainOwnership_resetCreatures(SimulationData data);
__global__ void cudaDomainOwnership_calcReferenceKeys(SimulationData data);
__global__ void cudaDomainOwnership_calcReferencePositions(SimulationData data);
__global__ void cudaDomainOwnership_findConstructingCreatureIds(SimulationData data);
__global__ void cudaDomainOwnership_findConstructingCreatures(SimulationData data);
__global__ void cudaDomainOwnership_decideCreatureOwners(SimulationData data, bool initialAssignment);

// Region of interest of the own domain
__global__ void cudaDomainRoi_clearBaseBitmap(SimulationData data, DomainSyncData syncData);
__global__ void cudaDomainRoi_markOwnedObjects(SimulationData data, DomainSyncData syncData);
__global__ void cudaDomainRoi_dilate(SimulationData data, DomainSyncData syncData);

// Id maps
__global__ void cudaDomainMaps_clear(DomainSyncData syncData);
__global__ void cudaDomainMaps_insert(SimulationData data, DomainSyncData syncData);

// Packing the messages to the other domains
__global__ void cudaDomainPack_reset(SimulationData data, DomainSyncData syncData);
__global__ void cudaDomainPack_roiBitmaps(SimulationData data, DomainSyncData syncData);
__global__ void cudaDomainPack_objects(SimulationData data, DomainSyncData syncData);
__global__ void cudaDomainPack_particles(SimulationData data, DomainSyncData syncData);
__global__ void cudaDomainPack_genomeRequests(SimulationData data, DomainSyncData syncData);
__global__ void cudaDomainPack_requestedGenomes(SimulationData data, DomainSyncData syncData);
__global__ void cudaDomainPack_transferredGenomes(SimulationData data, DomainSyncData syncData);
__global__ void cudaDomainPack_ops(SimulationData data, DomainSyncData syncData);
__global__ void cudaDomainPack_sensorScans(SimulationData data, DomainSyncData syncData);

// Handing over the ownership once the messages are complete
__global__ void cudaDomainCommit_objects(SimulationData data, uint8_t round);
__global__ void cudaDomainCommit_creatures(SimulationData data);
__global__ void cudaDomainCommit_particles(SimulationData data, uint8_t round);
__global__ void cudaDomainCommit_resetQueues(SimulationData data, DomainSyncData syncData);

// Unpacking the message of another domain
__global__ void cudaDomainUnpack_roiBitmap(SimulationData data, DomainSyncData syncData, int sender);
__global__ void cudaDomainUnpack_genomes(SimulationData data, DomainSyncData syncData, int sender);
__global__ void cudaDomainUnpack_genomeRequests(SimulationData data, DomainSyncData syncData, int sender);
__global__ void cudaDomainUnpack_creatures(SimulationData data, DomainSyncData syncData, int sender);
__global__ void cudaDomainUnpack_objects(SimulationData data, DomainSyncData syncData, int sender, uint8_t round);
__global__ void cudaDomainUnpack_particles(SimulationData data, DomainSyncData syncData, int sender, uint8_t round);
__global__ void cudaDomainUnpack_resolveObjects(SimulationData data, DomainSyncData syncData, int sender);
__global__ void cudaDomainUnpack_applyOps(SimulationData data, SimulationStatistics statistics, DomainSyncData syncData, int sender);
__global__ void cudaDomainUnpack_sensorScanRequests(SimulationData data, DomainSyncData syncData, int sender);
__global__ void cudaDomainUnpack_sensorScanResponses(SimulationData data, DomainSyncData syncData, int sender);

// Finishing a sync round
__global__ void cudaDomainSync_markStaleGhosts(SimulationData data, uint8_t round);
__global__ void cudaDomainSync_removeConnectionsToRemovedGhosts(SimulationData data);
__global__ void cudaDomainSync_deleteRemovedGhosts(SimulationData data);
__global__ void cudaDomainSync_publishSensorScans(SimulationData data, DomainSyncData syncData);
__global__ void cudaDomainSync_resetPendingSensorScans(SimulationData data);

// Initial distribution after every domain received the whole world
__global__ void cudaDomainDistribute_assignOwners(SimulationData data, uint8_t round);
__global__ void cudaDomainDistribute_removeObjectsOutsideRoi(SimulationData data);
__global__ void cudaDomainDistribute_assignParticleOwners(SimulationData data, uint8_t round);
