#include "BrowserDataService.h"

#include <boost/algorithm/string/split.hpp>
#include <boost/algorithm/string.hpp>

#include "BrowserDataTO.h"

std::vector<BrowserDataTO> BrowserDataService::createBrowserData(std::vector<NetworkDataTO> const& networkTOs)
{
    std::list<BrowserDataTO> result;
    for (auto const& entry : networkTOs) {

        //parse folder
        std::vector<std::string> location;
        boost::split(location, entry->simName, boost::is_any_of("/"));
        if (!location.empty()) {
            location.pop_back();
        }

        if (!result.empty()) {

            //find matching node
            auto searchIter = result.end();
            auto bestMatchIter = searchIter;
            int bestMatchEqualFolders = -1;
            for (int i = 0; i < result.size(); ++i) {
                --searchIter;
                auto otherEntry = *searchIter;

                int equalFolders = 0;
                int numFolders = std::min(location.size(), otherEntry->location.size());
                for (int i = 0; i < numFolders; ++i) {
                    if (location[i] == otherEntry->location[i]) {
                        ++equalFolders;
                    } else {
                        break;
                    }
                }

                if (equalFolders < bestMatchEqualFolders) {
                    break;
                }
                if (equalFolders > bestMatchEqualFolders) {
                    bestMatchIter = searchIter;
                    bestMatchEqualFolders = equalFolders;
                }
            }

            //insert folders
            for (int i = bestMatchEqualFolders; i < location.size(); ++i) {
                auto browserData = std::make_shared<_BrowserDataTO>();
                browserData->location = std::vector(location.begin(), location.begin() + i + 1);
                browserData->type = entry->type;
                browserData->node = BrowserFolder();
                result.insert(bestMatchIter, browserData);
            }
        }

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
        browserData->location = location;
        browserData->node = leaf;

        result.emplace_back(browserData);
    }
    return std::vector(result.begin(), result.end());
}
