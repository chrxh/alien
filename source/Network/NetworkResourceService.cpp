#include "NetworkResourceService.h"

#include <boost/algorithm/string/split.hpp>
#include <boost/algorithm/string.hpp>

#include "NetworkResourceTreeTO.h"

namespace
{
    int getNumEqualFolders(std::vector<std::string> const& folderNames, std::vector<std::string> const& otherfolderNames)
    {
        auto equalFolders = 0;
        auto numFolders = std::min(folderNames.size(), otherfolderNames.size());
        for (int i = 0; i < numFolders; ++i) {
            if (folderNames[i] == otherfolderNames[i]) {
                ++equalFolders;
            } else {
                return equalFolders;
            }
        }
        return equalFolders;
    }
}

std::vector<NetworkResourceTreeTO> NetworkResourceService::createBrowserData(std::vector<NetworkResourceRawTO> const& networkTOs)
{
    std::list<NetworkResourceTreeTO> browserDataToList;
    for (auto const& entry : networkTOs) {

        //parse folder names
        std::vector<std::string> folderNames;
        std::string nameWithoutFolders;
        boost::split(folderNames, entry->simName, boost::is_any_of("/"));
        if (!folderNames.empty()) {
            nameWithoutFolders = folderNames.back();
            folderNames.pop_back();
        }

        std::list<NetworkResourceTreeTO>::iterator bestMatchIter;
        int bestMatchEqualFolders;
        if (!browserDataToList.empty()) {

            //find matching node
            auto searchIter = browserDataToList.end();
            bestMatchIter = searchIter;
            bestMatchEqualFolders = -1;
            for (int i = 0; i < browserDataToList.size(); ++i) {
                --searchIter;
                auto otherEntry = *searchIter;
                auto equalFolders = getNumEqualFolders(folderNames, otherEntry->folderNames);
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
            bestMatchIter = browserDataToList.begin();
            bestMatchEqualFolders = 0;
        }

        //insert folders
        for (int i = bestMatchEqualFolders; i < folderNames.size(); ++i) {
            auto browserData = std::make_shared<_NetworkResourceTreeTO>();
            browserData->folderNames = std::vector(folderNames.begin(), folderNames.begin() + i + 1);
            browserData->type = entry->type;
            browserData->node = BrowserFolder();
            bestMatchIter = browserDataToList.insert(bestMatchIter, browserData);
            ++bestMatchIter;
        }

        //insert leaf
        auto browserData = std::make_shared<_NetworkResourceTreeTO>();
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
        browserData->folderNames = folderNames;
        browserData->node = leaf;
        browserDataToList.insert(bestMatchIter, browserData);
    }

    //calc folder lines
    std::vector result(browserDataToList.begin(), browserDataToList.end());
    for (int i = 0; i < result.size(); ++i) {
        auto& entry = result.at(i);

        if (i == 0) {
            if (!entry->isLeaf()) {
                entry->folderLines.emplace_back(FolderLine::Start);
            }
        } else {
            auto const& prevEntry = result.at(i - 1);
            auto numEqualFolders = getNumEqualFolders(entry->folderNames, prevEntry->folderNames);

            entry->folderLines.resize(entry->folderNames.size(), FolderLine::None);

            //process until numEqualFolders - 1
            if (numEqualFolders > 0) {
                int f = numEqualFolders - 1;
                if (prevEntry->folderLines.at(f) == FolderLine::Start) {
                    entry->folderLines.at(f) = FolderLine::End;
                } else if (prevEntry->folderLines.at(f) == FolderLine::End) {
                    prevEntry->folderLines.at(f) = FolderLine::Branch;
                    entry->folderLines.at(f) = FolderLine::End;
                } else if (prevEntry->folderLines.at(f) == FolderLine::Branch) {
                    entry->folderLines.at(f) = FolderLine::End;
                } else if (prevEntry->folderLines.at(f) == FolderLine::None) {
                    for (int j = i - 1; j >= 0; --j) {
                        auto& otherEntry = result.at(j);
                        if (otherEntry->folderLines.at(f) == FolderLine::None) {
                            otherEntry->folderLines.at(f) = FolderLine::Continue;
                        } else if (otherEntry->folderLines.at(f) == FolderLine::End) {
                            otherEntry->folderLines.at(f) = FolderLine::Branch;
                        } else {
                            break;
                        }
                    }

                }

            }

            for (int f = 0; f < numEqualFolders - 1; ++f) {
                if (prevEntry->folderLines.at(f) == FolderLine::Branch) {
                    entry->folderLines.at(f) = FolderLine::None;
                }
            }

            if (numEqualFolders < entry->folderNames.size()) {
                CHECK(numEqualFolders + 1 == entry->folderNames.size());
                entry->folderLines.back() = FolderLine::Start;

                if (numEqualFolders > 0 && numEqualFolders < prevEntry->folderNames.size()) {
                    entry->folderLines.at(numEqualFolders - 1) = FolderLine::End;
                    bool noneFound = false;
                    for (int j = i - 1; j >= 0; --j) {
                        auto& otherEntry = result.at(j);
                        if (otherEntry->folderLines.at(numEqualFolders - 1) == FolderLine::None) {
                            otherEntry->folderLines.at(numEqualFolders - 1) = FolderLine::Continue;
                            noneFound = true;
                        } else if (noneFound && otherEntry->folderLines.at(numEqualFolders - 1) == FolderLine::End) {
                            otherEntry->folderLines.at(numEqualFolders - 1) = FolderLine::Branch;
                        } else {
                            break;
                        }
                    }
                }
            }
            if (numEqualFolders > 0 && numEqualFolders < prevEntry->folderNames.size() && numEqualFolders == entry->folderNames.size()) {
                entry->folderLines.back() = FolderLine::End;
            }
        }
    }
    return result;
}
