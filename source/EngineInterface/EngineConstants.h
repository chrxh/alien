#pragma once

auto constexpr VALUE_NOT_SET_UINT64 = 0x7fffffffffffffff;
auto constexpr VALUE_NOT_SET_UINT32 = 0x7fffffff;
auto constexpr VALUE_NOT_SET_FLOAT = 1e16f;

auto constexpr MAX_OBJECT_CONNECTIONS = 6;
auto constexpr STANDARD_NEURONS_PER_CELL = 8;
auto constexpr MEMORY_NEURONS_PER_CELL = 4;
auto constexpr TELEMETRY_INPUTS_PER_CELL = 4;
auto constexpr NEURAL_NET_INPUTS = STANDARD_NEURONS_PER_CELL + MEMORY_NEURONS_PER_CELL + TELEMETRY_INPUTS_PER_CELL;
auto constexpr NEURAL_NET_OUTPUTS = STANDARD_NEURONS_PER_CELL + MEMORY_NEURONS_PER_CELL;
static_assert(NEURAL_NET_INPUTS == 16);
auto constexpr MAX_COLORS = 10;
auto constexpr MAX_CELL_MEMORY_ENTRIES = 32;

auto constexpr WARP_SIZE = 32;

// Kernels that give each warp an object of its own are launched with this block size. It decides how many
// objects an SM works on at once, since one warp per block left the Blackwell consumer GPUs at a single
// resident block per SM, which is 32 of 1536 threads.
auto constexpr FLUID_WARPS_PER_BLOCK = 16;

auto constexpr TIMESTEPS_PER_CELL_FUNCTION = 3;

auto constexpr MAX_LAYERS = 20;
auto constexpr MAX_SOURCES = 40;

auto constexpr MAX_HISTOGRAM_SLOTS = 20;

auto constexpr TRIGGER_THRESHOLD = 0.1f;

auto constexpr CELL_UPDATE_INTERVAL = 100;

auto constexpr PREVIEW_WIDTH = 10000;
auto constexpr PREVIEW_HEIGHT = 200;
auto constexpr PREVIEW_MAX_CELLS = 500;
