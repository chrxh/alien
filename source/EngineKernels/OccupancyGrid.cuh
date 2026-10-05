#pragma once

#include "Base.cuh"
#include "CudaMemoryManager.cuh"

// One bit per integer position of the world. The bits are stored in tiles of 8x8 positions, so a small area is covered by few memory transactions.
//
// An area can have up to 17x17 positions and may reach beyond the world boundaries. To avoid splitting it there, the positions near the lower
// boundaries are copied beyond the upper boundaries.
//
// A coarse level marks blocks of 32x32 positions that contain set positions inside or at most one position outside. It allows to skip empty
// regions quickly.
class OccupancyGrid
{
public:
    static int constexpr TileSize = 8;  // A tile is stored in an uint64_t: bit (y % 8) * 8 + (x % 8) belongs to position (x, y)
    static int constexpr BlockSize = 32;
    static int constexpr MaxAreaSize = 17;

    __host__ __inline__ void init(int2 const& worldSize)
    {
        _worldSize = worldSize;
        _numTiles = {(worldSize.x + MaxAreaSize + TileSize - 1) / TileSize, (worldSize.y + MaxAreaSize + TileSize - 1) / TileSize};
        CudaMemoryManager::getInstance().acquireMemory<uint64_t>(_numTiles.x * _numTiles.y, _tiles);
        CHECK_FOR_DEVICE_ERRORS(cudaMemset(_tiles, 0, sizeof(uint64_t) * _numTiles.x * _numTiles.y));

        _numBlocks = {(worldSize.x + BlockSize - 1) / BlockSize, (worldSize.y + BlockSize - 1) / BlockSize};
        CudaMemoryManager::getInstance().acquireMemory<uint32_t>(getNumBlockWords(), _blocks);
        CHECK_FOR_DEVICE_ERRORS(cudaMemset(_blocks, 0, sizeof(uint32_t) * getNumBlockWords()));
    }

    __host__ __inline__ void free()
    {
        CudaMemoryManager::getInstance().freeMemory(_tiles);
        CudaMemoryManager::getInstance().freeMemory(_blocks);
    }

    __device__ __inline__ void clear_system()
    {
        auto partition = calcSystemThreadPartition(_numTiles.x * _numTiles.y);
        for (int index = partition.startIndex; index <= partition.endIndex; index += partition.step) {
            _tiles[index] = 0;
        }
        auto blockPartition = calcSystemThreadPartition(getNumBlockWords());
        for (int index = blockPartition.startIndex; index <= blockPartition.endIndex; index += blockPartition.step) {
            _blocks[index] = 0;
        }
    }

    // The position must lie inside the world
    __device__ __inline__ void set(int2 const& pos)
    {
        markBlocksNear(pos);
        setBit(pos);
        auto isNearLowerBoundaryX = pos.x < MaxAreaSize;
        auto isNearLowerBoundaryY = pos.y < MaxAreaSize;
        if (isNearLowerBoundaryX) {
            setBit({pos.x + _worldSize.x, pos.y});
        }
        if (isNearLowerBoundaryY) {
            setBit({pos.x, pos.y + _worldSize.y});
        }
        if (isNearLowerBoundaryX && isNearLowerBoundaryY) {
            setBit({pos.x + _worldSize.x, pos.y + _worldSize.y});
        }
    }

    // The position must lie inside the world
    __device__ __inline__ bool isSet(int2 const& pos) const { return (_tiles[getTileIndex(getTile(pos))] & getBit(pos)) != 0; }

    // Uses the coarse level, the position must lie inside the world or on its upper boundaries. The blocks at the upper world boundaries can be
    // cut off by them and are therefore never reported empty.
    __device__ __inline__ bool isAnySetNearBlockOf(int2 const& pos) const
    {
        int2 block{pos.x / BlockSize, pos.y / BlockSize};
        if (block.x >= _numBlocks.x - 1 || block.y >= _numBlocks.y - 1) {
            return true;
        }
        auto blockIndex = getBlockIndex(block);
        return ((_blocks[blockIndex / 32] >> (blockIndex % 32)) & 1) != 0;
    }

    __device__ __inline__ bool isAnySet(int2 const& minPos, int2 const& maxPos) const
    {
        auto area = getArea(minPos, maxPos);
        for (int tileY = area.firstTile.y; tileY <= area.lastTile.y; ++tileY) {
            for (int tileX = area.firstTile.x; tileX <= area.lastTile.x; ++tileX) {
                int2 tile{tileX, tileY};
                if ((_tiles[getTileIndex(tile)] & getAreaMask(area, tile)) != 0) {
                    return true;
                }
            }
        }
        return false;
    }

private:
    struct Area
    {
        int2 start;  // Inside the world
        int2 end;    // May lie in the copied positions beyond the upper world boundaries
        int2 firstTile;
        int2 lastTile;
    };

    // An area larger than the world is reduced to the world, so that each area fits into the world and the copied positions
    __device__ __inline__ Area getArea(int2 minPos, int2 maxPos) const
    {
        if (maxPos.x - minPos.x >= _worldSize.x) {
            minPos.x = 0;
            maxPos.x = _worldSize.x - 1;
        }
        if (maxPos.y - minPos.y >= _worldSize.y) {
            minPos.y = 0;
            maxPos.y = _worldSize.y - 1;
        }
        int2 start{wrap(minPos.x, _worldSize.x), wrap(minPos.y, _worldSize.y)};
        int2 end{start.x + maxPos.x - minPos.x, start.y + maxPos.y - minPos.y};
        return {start, end, getTile(start), getTile(end)};
    }

    __device__ __inline__ void setBit(int2 const& pos)
    {
        auto& tile = _tiles[getTileIndex(getTile(pos))];
        auto bit = getBit(pos);
        if ((tile & bit) == 0) {
            alienAtomicOr64(&tile, bit);
        }
    }

    // Marks the blocks of the position and of its neighbor positions
    __device__ __inline__ void markBlocksNear(int2 const& pos)
    {
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
                int2 neighborPos{wrap(pos.x + dx, _worldSize.x), wrap(pos.y + dy, _worldSize.y)};
                markBlock({neighborPos.x / BlockSize, neighborPos.y / BlockSize});
            }
        }
    }

    __device__ __inline__ void markBlock(int2 const& block)
    {
        auto blockIndex = getBlockIndex(block);
        auto& blocks = _blocks[blockIndex / 32];
        auto bit = 1u << (blockIndex % 32);
        if ((blocks & bit) == 0) {
            atomicOr(&blocks, bit);
        }
    }

    __host__ __device__ __inline__ int getNumBlockWords() const { return (_numBlocks.x * _numBlocks.y + 31) / 32; }

    __device__ __inline__ int getBlockIndex(int2 const& block) const { return block.y * _numBlocks.x + block.x; }

    __device__ __inline__ int getTileIndex(int2 const& tile) const { return tile.y * _numTiles.x + tile.x; }

    __device__ __inline__ static int2 getTile(int2 const& pos) { return {pos.x / TileSize, pos.y / TileSize}; }

    __device__ __inline__ static uint64_t getBit(int2 const& pos) { return 1ull << (pos.y % TileSize * TileSize + pos.x % TileSize); }

    // Bits of the tile that belong to positions of the area
    __device__ __inline__ static uint64_t getAreaMask(Area const& area, int2 const& tile)
    {
        auto firstRow = max(area.start.y - tile.y * TileSize, 0);
        auto lastRow = min(area.end.y - tile.y * TileSize, TileSize - 1);
        auto rowsMask = (~0ull >> (64 - TileSize * (lastRow - firstRow + 1))) << (TileSize * firstRow);

        auto firstColumn = max(area.start.x - tile.x * TileSize, 0);
        auto lastColumn = min(area.end.x - tile.x * TileSize, TileSize - 1);
        auto columnsInRow = (0xffull >> (TileSize - 1 - lastColumn + firstColumn)) << firstColumn;
        auto columnsMask = columnsInRow * 0x0101010101010101ull;  // Repeats the columns for every row of the tile

        return rowsMask & columnsMask;
    }

    // The coordinates lie less than one world size outside of the world, a general modulo operation would slow down the scan loops considerably
    __device__ __inline__ static int wrap(int coordinate, int size)
    {
        return coordinate < 0 ? coordinate + size : (coordinate >= size ? coordinate - size : coordinate);
    }

    int2 _worldSize;
    int2 _numTiles;
    uint64_t* _tiles;
    int2 _numBlocks;
    uint32_t* _blocks;
};
