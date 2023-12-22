#include "BrowserDataService.h"

#include <boost/algorithm/string/split.hpp>
#include <boost/algorithm/string.hpp>

#include "BrowserDataTO.h"

std::vector<BrowserDataTO> BrowserDataService::createBrowserData(std::vector<NetworkDataTO> const& networkTOs)
{
    std::list<BrowserDataTO> result;
    for (auto const& entry : networkTOs) {

        //parse folder
        std::vector<std::string> folders;
        std::string nameWithoutFolders;
        boost::split(folders, entry->simName, boost::is_any_of("/"));
        if (!folders.empty()) {
            nameWithoutFolders = folders.back();
            folders.pop_back();
        }

        std::list<BrowserDataTO>::iterator bestMatchIter;
        int bestMatchEqualFolders;
        if (!result.empty()) {

            //find matching node
            auto searchIter = result.end();
            bestMatchIter = searchIter;
            bestMatchEqualFolders = -1;
            for (int i = 0; i < result.size(); ++i) {
                --searchIter;
                auto otherEntry = *searchIter;

                int equalFolders = 0;
                int numFolders = std::min(folders.size(), otherEntry->folders.size());
                for (int i = 0; i < numFolders; ++i) {
                    if (folders[i] == otherEntry->folders[i]) {
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
            ++bestMatchIter;
        } else {
            bestMatchIter = result.begin();
            bestMatchEqualFolders = 0;
        }
        //insert folders
        for (int i = bestMatchEqualFolders; i < folders.size(); ++i) {
            auto browserData = std::make_shared<_BrowserDataTO>();
            browserData->folders = std::vector(folders.begin(), folders.begin() + i + 1);
            browserData->type = entry->type;
            browserData->node = BrowserFolder();
            bestMatchIter = result.insert(bestMatchIter, browserData);
            ++bestMatchIter;
        }

        auto browserData = std::make_shared<_BrowserDataTO>();
        BrowserLeaf leaf{
            .id = entry->id,
            .timestamp = entry->timestamp,
            .userName = entry->userName,
            .simName = nameWithoutFolders,
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
        browserData->folders = folders;
        browserData->node = leaf;

        result.insert(bestMatchIter, browserData);
    }
    return std::vector(result.begin(), result.end());
}
