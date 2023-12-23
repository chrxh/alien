#include "NetworkResourceService.h"

#include <boost/algorithm/string/split.hpp>
#include <boost/algorithm/string.hpp>

#include "NetworkResourceTreeTO.h"

namespace
{
    int getNumEqualFolders(std::vector<std::string> const& folderNames, std::vector<std::string> const& otherFolderNames)
    {
        auto equalFolders = 0;
        auto numFolders = std::min(folderNames.size(), otherFolderNames.size());
        for (int i = 0; i < numFolders; ++i) {
            if (folderNames[i] == otherFolderNames[i]) {
                ++equalFolders;
            } else {
                return equalFolders;
            }
        }
        return equalFolders;
    }
}

std::vector<NetworkResourceTreeTO> NetworkResourceService::createTreeTOs(
    std::vector<NetworkResourceRawTO> const& networkTOs,
    std::vector<std::vector<std::string>> const& expandedFolderNames)
{
    std::list<NetworkResourceTreeTO> treeToList;
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
        if (!treeToList.empty()) {

            //find matching node
            auto searchIter = treeToList.end();
            bestMatchIter = searchIter;
            bestMatchEqualFolders = -1;
            for (int i = 0; i < treeToList.size(); ++i) {
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
            bestMatchIter = treeToList.begin();
            bestMatchEqualFolders = 0;
        }

        //insert folders
        for (int i = bestMatchEqualFolders; i < folderNames.size(); ++i) {
            auto treeTO = std::make_shared<_NetworkResourceTreeTO>();
            treeTO->folderNames = std::vector(folderNames.begin(), folderNames.begin() + i + 1);
            treeTO->type = entry->type;
            treeTO->node = BrowserFolder();
            bestMatchIter = treeToList.insert(bestMatchIter, treeTO);
            ++bestMatchIter;
        }

        //insert leaf
        auto treeTO = std::make_shared<_NetworkResourceTreeTO>();
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
        treeTO->type = entry->type;
        treeTO->folderNames = folderNames;
        treeTO->node = leaf;
        treeToList.insert(bestMatchIter, treeTO);
    }

    //calc folder lines
    std::vector treeTOs(treeToList.begin(), treeToList.end());
    for (int i = 0; i < treeTOs.size(); ++i) {
        auto& entry = treeTOs.at(i);

        if (i == 0) {
            if (!entry->isLeaf()) {
                entry->folderLines.emplace_back(FolderSymbols::ExpandedFolder);
            }
        } else {
            auto const& prevEntry = treeTOs.at(i - 1);
            auto numEqualFolders = getNumEqualFolders(entry->folderNames, prevEntry->folderNames);

            entry->folderLines.resize(entry->folderNames.size(), FolderSymbols::None);

            //process until numEqualFolders - 1
            if (numEqualFolders > 0) {
                int f = numEqualFolders - 1;
                if (prevEntry->folderLines.at(f) == FolderSymbols::ExpandedFolder) {
                    entry->folderLines.at(f) = FolderSymbols::EndFolder;
                } else if (prevEntry->folderLines.at(f) == FolderSymbols::EndFolder) {
                    prevEntry->folderLines.at(f) = FolderSymbols::BranchFolder;
                    entry->folderLines.at(f) = FolderSymbols::EndFolder;
                } else if (prevEntry->folderLines.at(f) == FolderSymbols::BranchFolder) {
                    entry->folderLines.at(f) = FolderSymbols::EndFolder;
                } else if (prevEntry->folderLines.at(f) == FolderSymbols::None) {
                    for (int j = i - 1; j >= 0; --j) {
                        auto& otherEntry = treeTOs.at(j);
                        if (otherEntry->folderLines.at(f) == FolderSymbols::None) {
                            otherEntry->folderLines.at(f) = FolderSymbols::ContinueFolder;
                        } else if (otherEntry->folderLines.at(f) == FolderSymbols::EndFolder) {
                            otherEntry->folderLines.at(f) = FolderSymbols::BranchFolder;
                        } else {
                            break;
                        }
                    }

                }

            }

            for (int f = 0; f < numEqualFolders - 1; ++f) {
                if (prevEntry->folderLines.at(f) == FolderSymbols::BranchFolder) {
                    entry->folderLines.at(f) = FolderSymbols::None;
                }
            }

            if (numEqualFolders < entry->folderNames.size()) {
                CHECK(numEqualFolders + 1 == entry->folderNames.size());
                entry->folderLines.back() = FolderSymbols::ExpandedFolder;

                if (numEqualFolders > 0 && numEqualFolders < prevEntry->folderNames.size()) {
                    entry->folderLines.at(numEqualFolders - 1) = FolderSymbols::EndFolder;
                    bool noneFound = false;
                    for (int j = i - 1; j >= 0; --j) {
                        auto& otherEntry = treeTOs.at(j);
                        if (otherEntry->folderLines.at(numEqualFolders - 1) == FolderSymbols::None) {
                            otherEntry->folderLines.at(numEqualFolders - 1) = FolderSymbols::ContinueFolder;
                            noneFound = true;
                        } else if (noneFound && otherEntry->folderLines.at(numEqualFolders - 1) == FolderSymbols::EndFolder) {
                            otherEntry->folderLines.at(numEqualFolders - 1) = FolderSymbols::BranchFolder;
                        } else {
                            break;
                        }
                    }
                }
            }
            if (numEqualFolders > 0 && numEqualFolders < prevEntry->folderNames.size() && numEqualFolders == entry->folderNames.size()) {
                entry->folderLines.back() = FolderSymbols::EndFolder;
            }
        }
    }

    //collapse items
    std::unordered_set<std::string> expandedFolderStrings;
    for(auto const& folderNames : expandedFolderNames) {
        expandedFolderStrings.insert(boost::join(folderNames, "/"));
    }

    std::vector<NetworkResourceTreeTO> result;
    result.reserve(treeTOs.size());
    for (auto const& entry : treeTOs) {
        auto folderString = boost::join(entry->folderNames, "/");
        if (!expandedFolderStrings.contains(folderString)) {
            
        }
    }
    return treeTOs;
}
