#pragma once

struct CudaSettings
{
    int numBlocks = 16384;

    bool operator==(CudaSettings const& other) const = default;
};

