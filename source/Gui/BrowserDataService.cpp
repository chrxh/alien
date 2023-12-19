#include "BrowserDataService.h"

#include "BrowserDataTO.h"

std::vector<BrowserDataTO> BrowserDataService::createBrowserData(std::vector<NetworkDataTO> const& remoteData)
{
    std::vector<BrowserDataTO> result;
    result.reserve(remoteData.size());
    for (auto const& entry : remoteData) {
        auto browserData = std::make_shared<_BrowserDataTO>();

        BrowserLeaf leaf{
            .id = entry->id,
            .timestamp = entry->timestamp,
            .userName = entry->userName,
            .simName = entry->simName,
            .numLikesByEmojiType = entry->numLikesByEmojiType,
            .numDownloads = entry->numDownloads,
            .width = entry->width,
            .height = entry->height,
            .particles = entry->particles,
            .contentSize = entry->contentSize,
            .description = entry->description,
            .version = entry->version
        };

        browserData->type = entry->type;
        browserData->level = 0;
        browserData->node = leaf;

        result.emplace_back(browserData);
    }
    return result;
}
