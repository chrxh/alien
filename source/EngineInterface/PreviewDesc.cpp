#include "PreviewDesc.h"

#include "EngineConstants.h"

SignalPreviewDesc::SignalPreviewDesc()
{
    _channels.resize(STANDARD_NEURONS_PER_CELL, 0);
}
