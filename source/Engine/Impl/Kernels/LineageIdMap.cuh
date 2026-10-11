#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "Base.cuh"
#include "CudaMemoryManager.cuh"

// Fixed-size hash map in global device memory with open addressing and linear probing, keyed by lineage id.
// Entries are never removed; a map is either reset as a whole or rebuilt into a second map (see SimulationStatistics).
// The entry type must be trivially copyable and start with the key member `uint32_t lineageId`, because free slots
// are marked by writing EmptyLineageId into the leading four bytes of every entry.
template <typename Entry>
class LineageIdMap
{
public:
    static auto constexpr EmptyLineageId = 0xffffffffu;  // Marks an unused slot
    static auto constexpr NoLineageSlot = -1;            // Returned if a lineage id is not present or the map is full
    static auto constexpr Capacity = 1 << 18;

    __host__ void init()
    {
        static_assert(std::is_trivially_copyable_v<Entry>);
        static_assert(offsetof(Entry, lineageId) == 0);
        static_assert((Capacity & (Capacity - 1)) == 0, "Capacity must be a power of two so that probing can use a bit mask");

        CudaMemoryManager::getInstance().acquireMemory<Entry>(Capacity, _entries);
        CudaMemoryManager::getInstance().acquireMemory<uint32_t>(1, _numUsedSlots);
        clear();
    }

    __host__ void free()
    {
        CudaMemoryManager::getInstance().freeMemory(_entries);
        CudaMemoryManager::getInstance().freeMemory(_numUsedSlots);
    }

    // All values start at zero; only the leading key member is set to EmptyLineageId
    __host__ void clear()
    {
        CHECK_FOR_DEVICE_ERRORS(cudaMemset(_entries, 0, sizeof(Entry) * Capacity));
        CHECK_FOR_DEVICE_ERRORS(cudaMemset2D(_entries, sizeof(Entry), 0xff, sizeof(uint32_t), Capacity));
        CHECK_FOR_DEVICE_ERRORS(cudaMemset(_numUsedSlots, 0, sizeof(uint32_t)));
    }

    __host__ uint32_t readNumUsedSlots() const
    {
        uint32_t result;
        CHECK_FOR_DEVICE_ERRORS(cudaMemcpy(&result, _numUsedSlots, sizeof(uint32_t), cudaMemcpyDeviceToHost));
        return result;
    }

    __inline__ __device__ Entry& at(int index) { return _entries[index]; }
    __inline__ __device__ Entry const& at(int index) const { return _entries[index]; }

    __inline__ __device__ int insertOrFind(uint32_t lineageId)
    {
        auto index = toSlotIndex(lineageId);
        for (int i = 0; i < Capacity; ++i) {
            auto origLineageId = atomicCAS(&_entries[index].lineageId, EmptyLineageId, lineageId);
            if (origLineageId == EmptyLineageId) {
                atomicAdd(_numUsedSlots, 1u);
                return index;
            }
            if (origLineageId == lineageId) {
                return index;
            }
            index = toNextSlotIndex(index);
        }
        return NoLineageSlot;
    }

    __inline__ __device__ int find(uint32_t lineageId) const
    {
        auto index = toSlotIndex(lineageId);
        for (int i = 0; i < Capacity; ++i) {
            auto slotLineageId = _entries[index].lineageId;
            if (slotLineageId == lineageId) {
                return index;
            }
            if (slotLineageId == EmptyLineageId) {
                return NoLineageSlot;
            }
            index = toNextSlotIndex(index);
        }
        return NoLineageSlot;
    }

    // Not synchronized: may only be called while no other thread accesses the map
    __inline__ __device__ void resetSlot(int index)
    {
        auto& entry = _entries[index];
        entry = Entry{};
        entry.lineageId = EmptyLineageId;
    }
    __inline__ __device__ void resetNumUsedSlots() { *_numUsedSlots = 0; }

private:
    static auto constexpr KnuthHashFactor = 2654435761u;

    __inline__ __device__ static int toSlotIndex(uint32_t lineageId) { return toInt((lineageId * KnuthHashFactor) & (Capacity - 1)); }
    __inline__ __device__ static int toNextSlotIndex(int index) { return (index + 1) & (Capacity - 1); }

    Entry* _entries;
    uint32_t* _numUsedSlots;
};
