#pragma once

#include <string>
#include <vector>

#include <imgui.h>

#include <Network/NetworkResourceRawTO.h>

#include "Definitions.h"

enum GallerySorting
{
    GallerySorting_MostReactions,
    GallerySorting_Newest,
    GallerySorting_MostDownloads
};

enum LocalGallerySorting
{
    LocalGallerySorting_Newest,
    LocalGallerySorting_Name
};

// Shows the resources of the current workspace as tiles with preview pictures. A click selects a tile, a double click opens its resource.
class _BrowserGalleryWidget
{
public:
    static BrowserGalleryWidget create(BrowserData const& data, BrowserPictureCache const& pictureCache);

    void shutdown();

    void processSorting();
    void process();
    void processPlaceholderTiles();
    void processPaging();

    float getSortingWidth() const;
    float getPagerWidth() const;
    void resetPage();

private:
    _BrowserGalleryWidget(BrowserData const& data, BrowserPictureCache const& pictureCache);

    struct TileLayout
    {
        float tileWidth = 0;
        std::vector<float> columnOffsets;  // Relative to the left border of a row
        int numColumns = 1;
    };
    TileLayout calcTileLayout() const;
    float calcTileHeight(float tileWidth) const;

    void processTile(NetworkResourceRawTO const& rawTO, float tileWidth);
    void processPlaceholderTile(float tileWidth);
    void processBadges(NetworkResourceRawTO const& rawTO, ImVec2 const& picturePos);
    void processEmptyHint();
    void processKeyboardNavigation(std::vector<NetworkResourceRawTO> const& pageEntries);

    std::string getPageText() const;
    std::vector<NetworkResourceRawTO> getSortedEntries() const;

    void onSelectEntry(NetworkResourceRawTO const& rawTO);

    BrowserData _data;
    BrowserPictureCache _pictureCache;

    int _sorting = GallerySorting_MostReactions;
    int _localSorting = LocalGallerySorting_Newest;
    int _page = 0;
    int _numEntries = 0;
    int _numPages = 1;
    int _numColumns = 1;
};
