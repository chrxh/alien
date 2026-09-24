#pragma once

#include <optional>

#include <Base/Singleton.h>

#include "LayerColorPalette.h"

// Edits the layers and radiation sources of the current simulation together with the reference parameters
class LocationEditService
{
    MAKE_SINGLETON(LocationEditService);

public:
    // The new location is placed behind the given one. Returns its order number or nullopt if the maximum number of locations has been reached.
    std::optional<int> insertDefaultLayer(int orderNumber);
    std::optional<int> insertDefaultSource(int orderNumber);
    std::optional<int> cloneLocation(int orderNumber);

    void deleteLocation(int orderNumber);
    void moveLocationUpwards(int orderNumber);
    void moveLocationDownwards(int orderNumber);

private:
    RealVector2D calcPositionForNewLocation() const;

    LayerColorPalette _layerColorPalette;
    int _insertedLocationCounter = 0;
};
