#pragma once

#include "Definitions.h"


class _TemporalControlWindow
{
public:
    _TemporalControlWindow(StyleRepository const& styleRepository);

    void process();

private:
    StyleRepository _styleRepository;

    TextureData _runTexture;
    TextureData _pauseTexture;
    TextureData _stepBackwardTexture;
    TextureData _stepForwardTexture;
    TextureData _snapshotTexture;
    TextureData _restoreTexture;
};