#include "BrowserDataService.h"

#include "BrowserDataTO.h"

std::vector<BrowserDataTO> BrowserDataService::createBrowserData(std::vector<NetworkDataTO> const& remoteData)
{
    std::vector<BrowserDataTO> result;
    result.reserve(remoteData.size());
    for (auto const& entry : remoteData) {
        auto browserData = std::make_shared<_BrowserDataTO>();
        browserData->id = entry->id;
        browserData->timestamp = entry->timestamp;
        browserData->userName = entry->userName;
        browserData->simName = entry->simName;
        browserData->numLikesByEmojiType = entry->numLikesByEmojiType;
        browserData->numDownloads = entry->numDownloads;
        browserData->width = entry->width;
        browserData->height = entry->height;
        browserData->particles = entry->particles;
        browserData->contentSize = entry->contentSize;
        browserData->description = entry->description;
        browserData->version = entry->version;
        browserData->fromRelease = entry->fromRelease;
        browserData->type = entry->type;

        result.emplace_back(browserData);
    }
    return result;
}
