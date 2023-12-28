#include "NetworkResourceService.h"

#include <ranges>

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
    std::set<std::vector<std::string>> const& collapsedFolderNames)
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
                entry->treeSymbols.emplace_back(FolderTreeSymbols::Expanded);
            }
        } else {
            auto const& prevEntry = treeTOs.at(i - 1);
            auto numEqualFolders = getNumEqualFolders(entry->folderNames, prevEntry->folderNames);

            entry->treeSymbols.resize(entry->folderNames.size(), FolderTreeSymbols::None);

            //calc symbols at position numEqualFolders - 1
            if (numEqualFolders > 0) {
                int f = numEqualFolders - 1;
                if (prevEntry->treeSymbols.at(f) == FolderTreeSymbols::Expanded) {
                    entry->treeSymbols.at(f) = FolderTreeSymbols::End;
                } else if (prevEntry->treeSymbols.at(f) == FolderTreeSymbols::End) {
                    prevEntry->treeSymbols.at(f) = FolderTreeSymbols::Branch;
                    entry->treeSymbols.at(f) = FolderTreeSymbols::End;
                } else if (prevEntry->treeSymbols.at(f) == FolderTreeSymbols::Branch) {
                    entry->treeSymbols.at(f) = FolderTreeSymbols::End;
                } else if (prevEntry->treeSymbols.at(f) == FolderTreeSymbols::None) {
                    for (int j = i - 1; j >= 0; --j) {
                        auto& otherEntry = treeTOs.at(j);
                        if (otherEntry->treeSymbols.at(f) == FolderTreeSymbols::None) {
                            otherEntry->treeSymbols.at(f) = FolderTreeSymbols::Continue;
                        } else if (otherEntry->treeSymbols.at(f) == FolderTreeSymbols::End) {
                            otherEntry->treeSymbols.at(f) = FolderTreeSymbols::Branch;
                        } else {
                            break;
                        }
                    }

                }

            }

            //calc symbols before position numEqualFolders - 1
            for (int f = 0; f < numEqualFolders - 1; ++f) {
                if (prevEntry->treeSymbols.at(f) == FolderTreeSymbols::Branch) {
                    entry->treeSymbols.at(f) = FolderTreeSymbols::None;
                }
            }

            if (numEqualFolders < entry->folderNames.size()) {
                CHECK(numEqualFolders + 1 == entry->folderNames.size());
                entry->treeSymbols.back() = FolderTreeSymbols::Expanded;

                if (numEqualFolders > 0 && numEqualFolders < prevEntry->folderNames.size()) {
                    entry->treeSymbols.at(numEqualFolders - 1) = FolderTreeSymbols::End;
                    bool noneFound = false;
                    for (int j = i - 1; j >= 0; --j) {
                        auto& otherEntry = treeTOs.at(j);
                        if (otherEntry->treeSymbols.at(numEqualFolders - 1) == FolderTreeSymbols::None) {
                            otherEntry->treeSymbols.at(numEqualFolders - 1) = FolderTreeSymbols::Continue;
                            noneFound = true;
                        } else if (noneFound && otherEntry->treeSymbols.at(numEqualFolders - 1) == FolderTreeSymbols::End) {
                            otherEntry->treeSymbols.at(numEqualFolders - 1) = FolderTreeSymbols::Branch;
                        } else {
                            break;
                        }
                    }
                }
            }
            if (numEqualFolders > 0 && numEqualFolders < prevEntry->folderNames.size() && numEqualFolders == entry->folderNames.size()) {
                entry->treeSymbols.back() = FolderTreeSymbols::End;
            }
        }
    }

    //calc numLeafs for folders
    for (int i = toInt(treeTOs.size()) - 1; i > 0; --i) {
        auto& entry = treeTOs.at(i);
        if (entry->isLeaf()) {
            for (int j = i - 1; j >= 0; --j) {
                auto& otherEntry = treeTOs.at(j);
                auto numEqualFolders = getNumEqualFolders(entry->folderNames, otherEntry->folderNames);
                if (numEqualFolders == 0) {
                    break;
                }
                if (numEqualFolders == otherEntry->folderNames.size() && !otherEntry->isLeaf()) {
                    ++otherEntry->getFolder().numLeafs;
                }
            }
        }
    }

    //collapse items
    std::unordered_set<std::string> collapsedFolderStrings;
    for(auto const& folderNames : collapsedFolderNames) {
        collapsedFolderStrings.insert(boost::join(folderNames, "/"));
    }

    std::vector<NetworkResourceTreeTO> result;
    result.reserve(treeTOs.size());
    for (auto const& entry : treeTOs) {
        auto isVisible = true;

        std::string folderString;
        auto numSolderToCheck = entry->isLeaf() ? entry->folderNames.size() : entry->folderNames.size() - 1;
        for (size_t i = 0; i < numSolderToCheck; ++i) {
            folderString.append(entry->folderNames.at(i));
            if (collapsedFolderStrings.contains(folderString)) {
                isVisible = false;
            }
            if (i < numSolderToCheck) {
                folderString.append("/");
            }
        }

        if (!entry->isLeaf()) {
            auto folderString = boost::join(entry->folderNames, "/");
            if (collapsedFolderStrings.contains(folderString)) {
                entry->treeSymbols.back() = FolderTreeSymbols::Collapsed;
            }
        }
        if (isVisible) {
            result.emplace_back(entry);
        }
    }
    return result;
}
