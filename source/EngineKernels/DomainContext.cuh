#pragma once

#include <cstdint>

#include <cuda_runtime.h>

// The world is split into vertical strips, one per domain. A domain simulates the objects in its strip, except for cells, which
// belong to the domain of their creature. It also holds ghost copies of the objects of other domains within its region of
// interest: the tiles of its strip and of its own objects, dilated by the halo width.
struct DomainLayout
{
    static auto constexpr MaxDomains = 16;
    static auto constexpr TileSize = 16;

    int numDomains = 1;
    int2 worldSize = {0, 0};
    float stripStarts[MaxDomains + 1] = {};  // stripStarts[numDomains] = stripStarts[0] + worldSize.x
    float haloWidth = 0;
    float ownershipHysteresis = 0;
    int2 numTiles = {0, 0};
    int numBitmapWords = 0;  // Size of the region-of-interest bitmap of one domain

    // Fraction of the external energy pool each domain may draw from in a time step
    float externalEnergyShares[MaxDomains] = {};
};

// Identifies a domain of the domain decomposition; its index is also the owner index stored in objects and creatures
struct DomainContext
{
    int index = 0;
    int numDomains = 1;
    DomainLayout* layout = nullptr;  // Device memory
    uint32_t* roiBitmaps = nullptr;  // Device memory, the bitmaps of all domains one after another
    uint8_t* syncRound = nullptr;    // Device memory

    __device__ __inline__ bool isDecomposed() const { return numDomains > 1; }

    __device__ __inline__ int getStripOwner(float x) const
    {
        auto const& layoutRef = *layout;
        auto relX = x - layoutRef.stripStarts[0];
        auto worldWidth = toFloatValue(layoutRef.worldSize.x);
        if (relX < 0) {
            relX += worldWidth;
        }
        if (relX >= worldWidth) {
            relX -= worldWidth;
        }
        for (int domain = 1; domain < numDomains; ++domain) {
            if (relX < layoutRef.stripStarts[domain] - layoutRef.stripStarts[0]) {
                return domain - 1;
            }
        }
        return numDomains - 1;
    }

    // Distance along the x axis from x to the strip of the given domain, 0 if x lies within the strip
    __device__ __inline__ float getDistanceToStrip(int domain, float x) const
    {
        auto const& layoutRef = *layout;
        auto worldWidth = toFloatValue(layoutRef.worldSize.x);
        auto start = layoutRef.stripStarts[domain];
        auto width = layoutRef.stripStarts[domain + 1] - start;
        auto relX = x - start;
        relX -= floorf(relX / worldWidth) * worldWidth;
        if (relX < width) {
            return 0;
        }
        return fminf(relX - width, worldWidth - relX);
    }

    __device__ __inline__ int getTileIndex(float2 const& pos) const
    {
        auto const& layoutRef = *layout;
        auto tileX = static_cast<int>(floorf(pos.x)) / DomainLayout::TileSize;
        auto tileY = static_cast<int>(floorf(pos.y)) / DomainLayout::TileSize;
        tileX = ((tileX % layoutRef.numTiles.x) + layoutRef.numTiles.x) % layoutRef.numTiles.x;
        tileY = ((tileY % layoutRef.numTiles.y) + layoutRef.numTiles.y) % layoutRef.numTiles.y;
        return tileX + tileY * layoutRef.numTiles.x;
    }

    __device__ __inline__ uint32_t* getRoiBitmap(int domain) const { return roiBitmaps + static_cast<uint64_t>(domain) * layout->numBitmapWords; }

    __device__ __inline__ bool isInRoi(int domain, float2 const& pos) const
    {
        auto tileIndex = getTileIndex(pos);
        return (getRoiBitmap(domain)[tileIndex / 32] >> (tileIndex % 32)) & 1;
    }

private:
    __device__ __inline__ static float toFloatValue(int value) { return static_cast<float>(value); }
};
