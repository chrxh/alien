#pragma once

auto constexpr VALUE_NOT_SET_UINT64 = 0x7fffffffffffffff;
auto constexpr VALUE_NOT_SET_FLOAT = 1e16f;

auto constexpr MAX_OBJECT_CONNECTIONS = 6;
auto constexpr MAX_CHANNELS = 8;
auto constexpr MAX_COLORS = 7;
auto constexpr MAX_CELL_MEMORY_ENTRIES = 32;
auto constexpr MAX_TARGETS_PER_CREATURE = 8;

auto constexpr MAX_LAYERS = 20;
auto constexpr MAX_SOURCES = 40;

auto constexpr MAX_HISTOGRAM_SLOTS = 20;

auto constexpr MAX_ACTIVATION_TIME = 256 * 4;
auto constexpr TRIGGER_THRESHOLD = 0.1f;

auto constexpr PREVIEW_WIDTH = 10000;
auto constexpr PREVIEW_HEIGHT = 200;
auto constexpr PREVIEW_MAX_CELLS = 500;
