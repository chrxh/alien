#pragma once

#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include <Network/NetworkResourceRawTO.h>

#include <PersisterInterface/Definitions.h>

#include "Definitions.h"

enum GallerySorting
{
    GallerySorting_MostReactions,
    GallerySorting_Newest,
    GallerySorting_MostDownloads
};

class _BrowserGalleryWidget
{
public:
    static BrowserGalleryWidget create(BrowserData const& data);

    void shutdown();

    void processSorting();
    void process();
    void processPaging();

    void processPendingRequests();

    float getPagerWidth() const;
    void resetPage();

private:
    _BrowserGalleryWidget(BrowserData const& data);

    void processTile(NetworkResourceRawTO const& rawTO, float tileWidth);
    void processPicture(NetworkResourceRawTO const& rawTO, float width);
    void processReactionButton(NetworkResourceRawTO const& rawTO);
    void processReactionTooltip(NetworkResourceRawTO const& rawTO);

    std::string getPageText() const;
    std::vector<NetworkResourceRawTO> getSortedEntries() const;
    void requestMissingPictures(std::vector<NetworkResourceRawTO> const& pageEntries);
    bool hasPreviewPictures() const;

    void onSelectEntry(NetworkResourceRawTO const& rawTO);

    BrowserData _data;
    TaskProcessor _pictureProcessor;

    int _sorting = GallerySorting_MostReactions;
    int _page = 0;
    int _numEntries = 0;
    int _numPages = 1;
    std::unordered_map<std::string, std::optional<TextureData>> _pictureBySimId;
};
